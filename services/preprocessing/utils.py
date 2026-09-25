"""Shared preprocessing helpers — reusable across entity modules, per the
brief's "reusable preprocessing code rather than notebook-only transformations"
instruction (§6).
"""
from __future__ import annotations

import math
import re

_MONEY_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([KMB]?)\s*$", re.IGNORECASE)
_MONEY_MULTIPLIERS = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}


def parse_money_string(raw: object) -> float:
    """Parses NOAA Storm Events' DAMAGE_PROPERTY/DAMAGE_CROPS strings
    ("25.00K", "1.50M", "0.00K", "", NaN) into a plain float dollar amount.

    This is the "inconsistent units" case the brief asks preprocessing to
    handle (§6) — NOAA stores damage as a magnitude-suffixed string, not a
    number, and it's inconsistent even in whether the suffix is present.
    """
    if raw is None:
        return 0.0
    text = str(raw).strip()
    if text == "" or text.lower() == "nan":
        return 0.0
    match = _MONEY_RE.match(text)
    if not match:
        return 0.0
    amount, suffix = match.groups()
    return float(amount) * _MONEY_MULTIPLIERS[suffix.upper()]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon points (WGS84 mean
    Earth radius 6371 km). Used to compute route distances from real World
    Port Index coordinates — see ports_routes.py for why this is a documented
    proxy rather than a real shipping-lane distance dataset.
    """
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def iqr_outlier_bounds(series, k: float = 3.0) -> tuple[float, float]:
    """Tukey's-fences outlier bounds (k=3.0, the wider 'far outlier' fence
    rather than the standard 1.5, since demand data legitimately has bulk-
    order spikes that aren't data errors — see demand.py for how this is used
    to *flag*, not silently drop, rows).
    """
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return (q1 - k * iqr, q3 + k * iqr)
