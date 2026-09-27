"""Online Retail -> sale lines with cancellations netted out.

The Phase 3 cleaner dropped every cancellation row (`InvoiceNo` starting with 'C') and kept the orders they cancelled, so a
sale that was reversed minutes later still counted as demand. The two clearest cases: order 581483 (80,995 units of 23843) was
cancelled by C581484 twelve minutes later, and order 541431 (74,215 units of 23166) by C541433 sixteen minutes later. The second
was 95% of everything ever "demanded" of product 23166.

This module pairs each cancellation with the sale it reverses, so net demand is what customers actually kept:

* the key is (CustomerID, StockCode): a cancellation carries no reference to its original invoice;
* it reverses the MOST RECENT earlier sale first (a customer cancels what they just ordered), spilling into older sales if it is
  larger, and never into a sale made after it;
* the reversal is booked on the original sale's date (assumption A01).

Every raw row ends with exactly one `disposition`, and the counts of dispositions sum to the workbook's row count, so nothing
disappears without being counted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backend.services.preprocessing.cleaned import assumptions as A
from backend.services.preprocessing.utils import iqr_outlier_bounds

RAW_COLUMNS = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"]

DUPLICATE = "DUPLICATE_ROW"
NON_PRODUCT = "NON_PRODUCT_CODE"
UNSPECIFIED = "UNSPECIFIED_COUNTRY"
MALFORMED_CANCELLATION = "MALFORMED_CANCELLATION"
CANCELLATION = "CANCELLATION"
STOCK_ADJUSTMENT = "STOCK_ADJUSTMENT"
NON_POSITIVE_PRICE = "NON_POSITIVE_PRICE"
ZERO_QUANTITY = "ZERO_QUANTITY"
SALE = "SALE"

MATCHED_FULL = "MATCHED_FULL"
MATCHED_PARTIAL = "MATCHED_PARTIAL"
UNMATCHED_NO_PRIOR_SALE = "UNMATCHED_NO_PRIOR_SALE"
UNMATCHED_NO_CUSTOMER = "UNMATCHED_NO_CUSTOMER"


def classify_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalises the workbook and gives every row one `disposition`. `raw_row` is the row's position in the workbook."""
    df = raw[RAW_COLUMNS].copy()
    df.insert(0, "raw_row", np.arange(len(df)))
    is_duplicate = df.duplicated(subset=RAW_COLUMNS, keep="first")  # NaN == NaN here, like drop_duplicates in Phase 3

    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    df["Country"] = df["Country"].astype(str).str.strip()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["CustomerID"] = df["CustomerID"].astype("Int64")

    is_cancel_invoice = df["InvoiceNo"].str.startswith("C")
    is_non_product = df["StockCode"].isin(A.NON_PRODUCT_CODES) | df["StockCode"].str.startswith(A.NON_PRODUCT_PREFIXES)
    qty, price = df["Quantity"], df["UnitPrice"]

    df["disposition"] = np.select(
        [
            is_duplicate,
            is_non_product,
            df["Country"].isin(A.UNSPECIFIED_COUNTRIES),
            is_cancel_invoice & (qty >= 0),
            is_cancel_invoice,
            qty < 0,
            qty == 0,
            price <= 0,
        ],
        [DUPLICATE, NON_PRODUCT, UNSPECIFIED, MALFORMED_CANCELLATION, CANCELLATION, STOCK_ADJUSTMENT, ZERO_QUANTITY, NON_POSITIVE_PRICE],
        default=SALE,
    )
    df["line_no"] = df.groupby("InvoiceNo").cumcount() + 1  # position within the invoice in the workbook: stable, and part of the key
    return df


# The Phase 3 exclusion rules, reproduced ONLY so the difference between the old and the new demand can be accounted for unit by
# unit; a test checks that they reproduce data/processed/demand.csv exactly.
_LEGACY_NON_PRODUCT_CODES = frozenset({"POST", "DOT", "M", "MANUAL", "BANK CHARGES", "PADS", "CRUK", "AMAZONFEE", "C2"})
_LEGACY_UNSPECIFIED = frozenset({"Unspecified", "European Community", "nan"})


def bridge_to_legacy(rows: pd.DataFrame, cancelled_units_removed: int, net_units: int) -> dict:
    """Old demand units -> new net demand units, every unit accounted for.

    The old cleaner kept a row unless it was a duplicate, a cancellation or non-positive quantity, a legacy non-product code or an
    unspecified country. The new layer keeps a strict subset of those rows (SALE), then removes the cancelled units. So
    old - (units the old cleaner kept that the new one does not, by reason) - cancelled = new, exactly.
    """
    not_kept_by_legacy = (
        (rows["disposition"] == DUPLICATE)
        | rows["InvoiceNo"].str.startswith("C")
        | (rows["Quantity"] <= 0)
        | rows["StockCode"].isin(_LEGACY_NON_PRODUCT_CODES)
        | rows["Country"].isin(_LEGACY_UNSPECIFIED)
    )
    legacy = rows[~not_kept_by_legacy]
    legacy_units = int(legacy["Quantity"].sum())
    removed = legacy[legacy["disposition"] != SALE].groupby("disposition")["Quantity"].sum()
    removed_by_reason = {k: int(v) for k, v in removed.items()}
    bridged = legacy_units - sum(removed_by_reason.values()) - cancelled_units_removed
    assert bridged == net_units, f"demand bridge does not close: {bridged} != {net_units}"
    return {
        "legacy_units": legacy_units,
        "removed_because_not_a_priced_sale_or_not_a_product": removed_by_reason,
        "removed_because_cancelled": int(cancelled_units_removed),
        "new_net_units": int(net_units),
    }


def match_cancellations(sales: pd.DataFrame, cancellations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairs cancellations with the sales they reverse.

    Returns (pairs, outcomes). `pairs` has one row per (cancellation, sale) it touches; `outcomes` has one row per cancellation
    saying whether it was fully matched, partly matched or not matched and why. `sales` needs raw_row, InvoiceNo, line_no,
    CustomerID, StockCode, InvoiceDate, Quantity; `cancellations` the same with a negative Quantity.
    """
    # entries are [timestamp, raw_row, invoice, line_no, quantity still uncancelled], oldest first
    pool: dict[tuple[int, str], list[list]] = {}
    for r in sales.itertuples(index=False):
        if pd.isna(r.CustomerID):
            continue
        pool.setdefault((int(r.CustomerID), r.StockCode), []).append([r.InvoiceDate, r.raw_row, r.InvoiceNo, r.line_no, int(r.Quantity)])
    for entries in pool.values():
        entries.sort(key=lambda e: (e[0], e[1]))

    pairs: list[dict] = []
    outcomes: list[dict] = []
    for c in cancellations.sort_values(["InvoiceDate", "raw_row"]).itertuples(index=False):
        need = int(-c.Quantity)
        matched = 0
        if pd.isna(c.CustomerID):
            outcome = UNMATCHED_NO_CUSTOMER
        else:
            for entry in reversed(pool.get((int(c.CustomerID), c.StockCode), [])):  # newest sale first
                if entry[0] > c.InvoiceDate or entry[4] <= 0:
                    continue  # made after the cancellation, or already fully cancelled
                take = min(entry[4], need - matched)
                entry[4] -= take
                matched += take
                pairs.append(
                    {
                        "cancel_order_id": c.InvoiceNo, "cancel_raw_row": c.raw_row, "sale_order_id": entry[2], "sale_line_no": entry[3],
                        "sale_raw_row": entry[1], "customer_id": int(c.CustomerID), "product_id": c.StockCode, "matched_quantity": take,
                        "cancel_ts": c.InvoiceDate, "sale_ts": entry[0], "lag_days": round((c.InvoiceDate - entry[0]).total_seconds() / 86400, 4),
                    }
                )
                if matched == need:
                    break
            outcome = MATCHED_FULL if matched == need else MATCHED_PARTIAL if matched else UNMATCHED_NO_PRIOR_SALE
        outcomes.append(
            {"cancel_order_id": c.InvoiceNo, "cancel_raw_row": c.raw_row, "product_id": c.StockCode, "customer_id": c.CustomerID,
             "cancel_quantity": need, "matched_quantity": matched, "unmatched_quantity": need - matched, "outcome": outcome}
        )

    pair_columns = ["cancel_order_id", "cancel_raw_row", "sale_order_id", "sale_line_no", "sale_raw_row", "customer_id", "product_id",
                    "matched_quantity", "cancel_ts", "sale_ts", "lag_days"]
    outcome_columns = ["cancel_order_id", "cancel_raw_row", "product_id", "customer_id", "cancel_quantity", "matched_quantity", "unmatched_quantity", "outcome"]
    return pd.DataFrame(pairs, columns=pair_columns), pd.DataFrame(outcomes, columns=outcome_columns)


def build_sale_lines(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """The whole step: (sale_lines, pairs, outcomes, report). `sale_lines` carries gross, cancelled and net quantity, and keeps
    `raw_row` (the workbook position) so any line can be traced back to the source."""
    rows = classify_rows(raw)
    counts = rows["disposition"].value_counts().to_dict()
    assert sum(counts.values()) == len(raw), "every workbook row must have exactly one disposition"

    sales = rows[rows["disposition"] == SALE]
    cancels = rows[rows["disposition"] == CANCELLATION]
    pairs, outcomes = match_cancellations(sales, cancels)

    cancelled = pairs.groupby("sale_raw_row")["matched_quantity"].sum() if len(pairs) else pd.Series(dtype="int64")
    lines = pd.DataFrame(
        {
            "raw_row": sales["raw_row"].to_numpy(),
            "order_id": sales["InvoiceNo"].to_numpy(),
            "line_no": sales["line_no"].to_numpy(),
            "product_id": sales["StockCode"].to_numpy(),
            "quantity_gross": sales["Quantity"].astype("int64").to_numpy(),
            "unit_price_gbp": sales["UnitPrice"].to_numpy(),
            "order_ts": sales["InvoiceDate"].to_numpy(),
            "customer_id": sales["CustomerID"].to_numpy(),
            "location_id": sales["Country"].to_numpy(),
            "description": sales["Description"].to_numpy(),
        }
    )
    lines["quantity_cancelled"] = lines["raw_row"].map(cancelled).fillna(0).astype("int64")
    lines["quantity_net"] = lines["quantity_gross"] - lines["quantity_cancelled"]
    assert (lines["quantity_net"] >= 0).all(), "a line cannot be cancelled by more than it sold"

    positive = lines.loc[lines["quantity_net"] > 0, "quantity_net"]
    _, high = iqr_outlier_bounds(positive, k=A.OUTLIER_IQR_K)
    lines["is_quantity_outlier"] = lines["quantity_net"] > high

    lag = pairs["lag_days"] if len(pairs) else pd.Series(dtype="float64")
    weights = pairs["matched_quantity"] if len(pairs) else pd.Series(dtype="int64")
    units_by_lag = {
        "within_1_day": int(weights[lag <= 1].sum()), "1_to_30_days": int(weights[(lag > 1) & (lag <= 30)].sum()), "over_30_days": int(weights[lag > 30].sum()),
    }
    by_outcome = outcomes.groupby("outcome").agg(cancellations=("cancel_quantity", "size"), units=("cancel_quantity", "sum")) if len(outcomes) else pd.DataFrame()
    report = {
        "workbook_rows": len(raw),
        "dispositions": {k: int(v) for k, v in sorted(counts.items())},
        "sale_lines": len(lines),
        "sale_lines_fully_cancelled": int((lines["quantity_net"] == 0).sum()),
        "sale_lines_partly_cancelled": int(((lines["quantity_cancelled"] > 0) & (lines["quantity_net"] > 0)).sum()),
        "units_gross": int(lines["quantity_gross"].sum()),
        "units_cancelled_and_removed": int(lines["quantity_cancelled"].sum()),
        "units_net": int(lines["quantity_net"].sum()),
        "cancellation_outcomes": {k: {"cancellations": int(v["cancellations"]), "units": int(v["units"])} for k, v in by_outcome.iterrows()},
        "cancelled_units_by_lag": units_by_lag,
        "median_lag_days": round(float(lag.median()), 2) if len(lag) else None,
        "quantity_outlier_upper_fence": float(high),
        "quantity_outlier_lines": int(lines["is_quantity_outlier"].sum()),
        "bridge_to_phase3_demand": bridge_to_legacy(rows, int(lines["quantity_cancelled"].sum()), int(lines["quantity_net"].sum())),
    }
    return lines, pairs, outcomes, report
