"""Everything the cleaned layer assumes rather than measures.

Each constant lives here once, is imported by the module that uses it, and appears in `ASSUMPTIONS`, which the readiness report
prints verbatim. A number that shapes the data but is not written down here is a bug: it is exactly how the earlier layer ended
up with sea distances nobody could explain.

Where a value is unchanged from the Phase 3 modules it is imported from them, so the two layers cannot drift apart silently.
"""
from __future__ import annotations

from dataclasses import dataclass

from backend.services.preprocessing.cleaned import AUTHORED, SYNTHETIC
from backend.services.preprocessing.inventory import (  # noqa: F401  (re-exported: the one place these numbers are defined)
    REPLENISH_DAYS,
    SAFETY_DAYS,
    TARGET_DAYS_OF_COVER,
    WAREHOUSE_PROFILES,
)

# --------------------------------------------------------------------------- demand
# Stock codes that are fees, postage, adjustments or vouchers, not goods that were demanded. The Phase 3 list was
# {POST, DOT, M, MANUAL, BANK CHARGES, PADS, CRUK, AMAZONFEE, C2}; the audit of the raw file found D (discount), S (samples),
# B (bad-debt adjustment) and the GIFT_0001_* vouchers as well. DCGS* are real products and stay.
NON_PRODUCT_CODES = frozenset({"POST", "DOT", "M", "MANUAL", "C2", "D", "S", "B", "BANK CHARGES", "CRUK", "AMAZONFEE", "PADS"})
NON_PRODUCT_PREFIXES = ("GIFT_",)
UNSPECIFIED_COUNTRIES = frozenset({"Unspecified", "European Community"})  # no destination -> not demand at a location
OUTLIER_IQR_K = 3.0  # Tukey's far fence, as in Phase 3: flagged, never dropped (a large wholesale order is real demand)

# The modeling panel: same selection as ml/training/prepare_modeling_data.py (a test asserts the two agree)
MODELING_TOP_K = 40
MODELING_MIN_ACTIVE_DAYS = 100

# --------------------------------------------------------------------------- inventory
REPLENISHMENT_LEAD_TIME_DAYS = 0  # Phase 3 simplification kept: a replenishment arrives the day it is triggered
WAREHOUSE_ORDER = tuple(WAREHOUSE_PROFILES)  # allocation tie-break order, fixed so the split is reproducible

# --------------------------------------------------------------------------- routes
VESSEL_SPEED_KNOTS = 18.0
KM_PER_NAUTICAL_MILE = 1.852
COST_PER_KM = 0.012  # abstract cost units per unit per km; a placeholder, as in Phase 3
RAIL_DISTANCE_KM = 11_000.0
RAIL_TRANSIT_DAYS = (15.0, 18.0)
RAIL_COST_MULTIPLIER = 2.0
AIR_SPEED_KMH = 800.0
AIR_HANDLING_DAYS = 1.5
AIR_COST_MULTIPLIER = 6.0
CAPACITY_BY_MODE = {"sea": 8000, "rail": 2000, "air": 200}  # abstract capacity units; no public source

# --------------------------------------------------------------------------- suppliers
SUPPLIER_SEED = 42
BASE_CAPACITY_BY_COUNTRY = {"China": 10000, "India": 8000, "Vietnam": 6000, "Turkey": 5000, "Netherlands": 4000}
CAPACITY_RANGE = (0.6, 1.1)
UNIT_COST_RANGE = (85.0, 120.0)
LEAD_TIME_RANGE_DAYS = (4, 11)  # inclusive
RELIABILITY_BAND_BY_RISK = {"LOW": (0.90, 0.99), "MEDIUM": (0.80, 0.92), "HIGH": (0.65, 0.85)}
PRODUCT_CARRY_PROBABILITY = 0.65  # the Phase 3 roster skipped a (supplier, product) pair with probability 0.35
SUPPLIER_PRODUCT_COUNT = 3
REDUCED_CAPACITY_FACTOR = 0.5
DISRUPTED_CAPACITY_FACTOR = 0.0
RED_SEA_DURATION_DAYS = 90.0

# --------------------------------------------------------------------------- tariffs
TARIFF_BROAD_COVERAGE_MIN_COUNTRIES = 100  # the reference year is the latest year at least this many countries report
TARIFF_EXTREME_PCT = 50.0
TARIFF_JUMP_PP = 10.0

# --------------------------------------------------------------------------- products
ABC_CUTOFFS = (0.80, 0.95)


@dataclass(frozen=True)
class Assumption:
    id: str
    area: str
    statement: str
    label: str  # SYNTHETIC or AUTHORED: an assumption is never REAL
    consequence: str  # what changes if it is wrong


ASSUMPTIONS: tuple[Assumption, ...] = (
    Assumption("A01", "demand", "A cancellation invoice (InvoiceNo starting 'C') reverses the most recent earlier sale of the same product to "
               "the same CustomerID (last-in-first-out), spilling into older sales if it is larger than the latest; the reversal is booked "
               "on the ORIGINAL sale's date, so a cancelled order never appears as demand.", AUTHORED,
               "First-in-first-out would move some reversals to older orders; totals do not change, dates of a minority of units do."),
    Assumption("A02", "demand", "A cancellation with no earlier sale in the workbook (the sale pre-dates 2010-12-01) or without a CustomerID "
               "is not netted: there is no date to book it on, and booking it on the cancellation date would create negative demand.", AUTHORED,
               "Net demand is slightly overstated by the units of returns whose sale is outside the data window."),
    Assumption("A03", "demand", "Rows with a negative quantity on a non-'C' invoice (write-offs: 'damaged', 'check', 'thrown away') and rows "
               "with a price <= 0 are stock adjustments or give-aways, not customer demand, and are excluded.", AUTHORED,
               "Those units would otherwise count as demand; the demand bridge in the manifest gives their exact number."),
    Assumption("A04", "demand", "Exact duplicate rows (all eight columns equal) are one line entered twice and are dropped, as in Phase 3.", AUTHORED,
               "A genuine second scan of the same item on one invoice would be undercounted (0.97% of rows)."),
    Assumption("A05", "demand", "Codes for postage, fees, discounts, samples, bad-debt adjustments and gift vouchers are not products.", AUTHORED,
               "A code wrongly excluded removes real demand; the list is short and printed in the manifest."),
    Assumption("A06", "demand", "Rows whose Country is 'Unspecified' or 'European Community' have no destination and are excluded.", AUTHORED,
               "About 0.1% of rows; demand at a location cannot include them."),
    Assumption("A07", "demand", f"A sale line is flagged as a quantity outlier if its net quantity is above Q3 + {OUTLIER_IQR_K} x IQR of "
               "net line quantities. Flagged, never removed.", AUTHORED, "Models that want to cap large orders can use the flag."),
    Assumption("A08", "modeling", f"The modeling panel is the top {MODELING_TOP_K} (product, location) series by net volume among series with at "
               f"least {MODELING_MIN_ACTIVE_DAYS} active days, on a dense daily calendar (days without a sale are zero demand); split 70/15/15 "
               "by time.", AUTHORED, "Which products are in the panel changes if the thresholds change."),
    Assumption("A09", "inventory", f"Inventory is a simulation, not an observation. Start stock = {TARGET_DAYS_OF_COVER} days of cover, safety stock = "
               f"{SAFETY_DAYS} days (x a warehouse multiplier), replenish {REPLENISH_DAYS} days of demand whenever closing stock would fall below "
               f"safety stock; all sized on the FULL-YEAR mean of net demand (a simulation seed, not a forecast) and replenished with a "
               f"{REPLENISHMENT_LEAD_TIME_DAYS}-day lead time.", SYNTHETIC,
               "Instant replenishment flatters availability; stock-outs here are a lower bound. Do not use as a backtest of an inventory policy."),
    Assumption("A10", "inventory", f"Each product's demand is split across three warehouses in fixed shares {dict((k, v['demand_share']) for k, v in WAREHOUSE_PROFILES.items())} "
               "by largest remainder, so integer units are conserved exactly. The warehouses are in India while the demand is UK retail: the "
               "split is an illustration, not geography.", SYNTHETIC, "The Inventory Agent's transfer logic is exercised, not validated."),
    Assumption("A11", "routes", "A sea lane's distance is the sum of great-circle legs between the waypoints of scripts/build_lanes.py (lanes "
               "checked against the Natural Earth 110m coastline). Two lanes the map does not have (Mumbai and Chennai round the Cape) get waypoints "
               "authored here and checked the same way. The result is an approximation of a shipping lane, not a vessel track or a published "
               "sea-distance table.", AUTHORED, "Distances could be off by several percent either way; the Suez-versus-Cape ordering is what matters."),
    Assumption("A12", "routes", f"Sea transit = distance / {VESSEL_SPEED_KNOTS} knots. Rail (China-Europe Railway Express) = {RAIL_DISTANCE_KM:.0f} km, "
               f"{RAIL_TRANSIT_DAYS[0]:.0f}-{RAIL_TRANSIT_DAYS[1]:.0f} days. Air = great circle at {AIR_SPEED_KMH:.0f} km/h plus {AIR_HANDLING_DAYS} days handling. "
               f"Cost per unit = {COST_PER_KM} x sea km; rail costs {RAIL_COST_MULTIPLIER}x and air {AIR_COST_MULTIPLIER}x the same lane's Suez sea cost. Capacity "
               f"{CAPACITY_BY_MODE}. All carried over from Phase 3; none has a public source in this workspace.", SYNTHETIC,
               "Freight costs are placeholders in abstract cost units."),
    Assumption("A13", "suppliers", "The supplier roster (ids S001-S008, countries, risk tiers) is fictitious and labelled so. Capacity, unit cost, "
               "lead time and reliability are drawn from seeded uniform ranges; reliability follows the supplier's risk tier, not its current "
               "status. Each (supplier, product) pair has its own random stream, so adding a product never changes another pair's numbers.", SYNTHETIC,
               "Nothing about a real supplier is implied."),
    Assumption("A14", "suppliers", f"Supplier unit cost and route cost are in ABSTRACT COST UNITS. They are not pounds and are not commensurate "
               "with the retail price in product.avg_unit_price_gbp (a cost of ~100 against a retail price of ~1).", SYNTHETIC,
               "Only relative comparisons and scenario deltas are meaningful; an absolute cost is not."),
    Assumption("A15", "suppliers", f"The supplier-carrying products are the {SUPPLIER_PRODUCT_COUNT} highest-volume products in the modeling panel (the "
               "Phase 3 rule, re-applied to corrected demand), so that suppliers, forecasts and stock refer to the same products.", AUTHORED,
               "The products differ from Phase 3 because the old top three included an order that was cancelled."),
    Assumption("A16", "scenarios", f"A DISRUPTED supplier keeps {DISRUPTED_CAPACITY_FACTOR:.0%} of nominal capacity and a REDUCED one {REDUCED_CAPACITY_FACTOR:.0%}. "
               "The current agents act only on DISRUPTED; REDUCED is recorded for when they do.", SYNTHETIC, "REDUCED behaviour is a placeholder."),
    Assumption("A17", "scenarios", f"The Red Sea scenario makes every lane through Bab-el-Mandeb unavailable for {RED_SEA_DURATION_DAYS:.0f} days. The "
               "duration is illustrative; no real traffic, delay or cost figure is asserted.", SYNTHETIC, "It is a stress test, not a forecast."),
    Assumption("A18", "reference", "Port names in the route table are matched to World Port Index records by hand: Singapore -> 50000 'KEPPEL - (EAST SINGAPORE)' "
               "(the WPI has no port named Singapore), Mumbai -> 48840, Chennai -> 49450, Shanghai -> 59970, Rotterdam -> 31140.", AUTHORED,
               "A different Singapore terminal would change no distance by more than a few km."),
    Assumption("A19", "reference", "ISO-2 codes and the retail country labels ('EIRE', 'RSA', ...) are mapped to ISO-3 by hand for the countries the "
               "data uses. Only those countries have an ISO-2 code, so a port outside them has no ISO-3.", AUTHORED,
               "The country table is not a complete ISO 3166 list."),
    Assumption("A20", "tariffs", f"The tariff reference year is the latest year in which at least {TARIFF_BROAD_COVERAGE_MIN_COUNTRIES} countries report. A country whose "
               f"latest value is older is STALE, one with none is NO_DATA. Values above {TARIFF_EXTREME_PCT:.0f}% or moving by "
               f">= {TARIFF_JUMP_PP:.0f} points from the previous observation are flagged, not changed.", AUTHORED,
               "Flags are informational; no tariff value was altered or filled."),
    Assumption("A21", "products", f"ABC class is by cumulative net revenue: A up to {ABC_CUTOFFS[0]:.0%}, B up to {ABC_CUTOFFS[1]:.0%}, C the rest. A product's "
               "price is the median of its priced sale lines.", AUTHORED, "A revenue ranking is a proxy for criticality, not a measure of it."),
)
