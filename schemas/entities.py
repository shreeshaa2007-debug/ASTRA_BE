"""Internal data schemas — the seven entities defined in docs/data-plan.md §
"Internal schemas". Every preprocessing module in backend/services/preprocessing
adapts its source dataset onto these, so nothing downstream (agents, API,
frontend) ever sees a source-specific column name.

Every record carries `provenance` so a consumer can tell real data from a
derived/synthetic stand-in without reading documentation — this is the
concrete mechanism behind the "label it if it's simulated" rule (brief §30).
"""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Provenance(str, Enum):
    """Where a record's values actually came from — never guess, always tag."""

    REAL = "real"                # measured, from an acquired public dataset
    DERIVED = "derived"          # computed from real data via a documented rule
    SYNTHETIC = "synthetic"      # no real source exists; generated, clearly labeled


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ShipmentStatus(str, Enum):
    ON_TIME = "ON_TIME"
    DELAYED = "DELAYED"
    AT_RISK = "AT_RISK"
    CANCELLED = "CANCELLED"


class SupplierStatus(str, Enum):
    ACTIVE = "ACTIVE"
    REDUCED = "REDUCED"
    DISRUPTED = "DISRUPTED"


class RouteStatus(str, Enum):
    NORMAL = "NORMAL"
    DELAYED = "DELAYED"
    DISRUPTED = "DISRUPTED"
    ALTERNATIVE = "ALTERNATIVE"


class DisruptionStatus(str, Enum):
    ACTIVE = "ACTIVE"
    RESOLVED = "RESOLVED"
    HISTORICAL = "HISTORICAL"


# --------------------------------------------------------------------------- #
# SHIPMENTS — schema defined for completeness (brief §5); no processed table
# exists yet, since category D (transportation/logistics) has no acquired
# dataset — see data/dataset_registry.yaml. Populated once real or curated
# shipment records exist.
# --------------------------------------------------------------------------- #
class ShipmentRecord(BaseModel):
    shipment_id: str
    product_id: str
    origin: str
    destination: str
    route_id: str
    carrier_id: Optional[str] = None
    quantity: int = Field(gt=0)
    departure_date: date
    expected_arrival: date
    actual_arrival: Optional[date] = None
    transport_mode: str
    cost: float = Field(ge=0)
    status: ShipmentStatus
    provenance: Provenance


# --------------------------------------------------------------------------- #
# INVENTORY — derived from demand (backend/services/preprocessing/inventory.py)
# --------------------------------------------------------------------------- #
class InventoryRecord(BaseModel):
    warehouse_id: str
    product_id: str
    date: date
    opening_stock: int = Field(ge=0)
    inbound_quantity: int = Field(ge=0)
    outbound_quantity: int = Field(ge=0)
    closing_stock: int = Field(ge=0)
    safety_stock: int = Field(ge=0)
    stockout_flag: bool
    provenance: Provenance = Provenance.DERIVED


# --------------------------------------------------------------------------- #
# DEMAND — from data/raw/demand_uci_online_retail
# --------------------------------------------------------------------------- #
class DemandRecord(BaseModel):
    date: date
    product_id: str
    location_id: str
    demand_quantity: float = Field(ge=0)
    provenance: Provenance = Provenance.REAL

    # feature columns added by backend/services/preprocessing/features.py —
    # optional because the base record is valid before features are attached
    lag_1: Optional[float] = None
    lag_2: Optional[float] = None
    lag_7: Optional[float] = None
    lag_14: Optional[float] = None
    lag_28: Optional[float] = None
    rolling_mean_7: Optional[float] = None
    rolling_mean_14: Optional[float] = None
    rolling_mean_28: Optional[float] = None
    day_of_week: Optional[int] = Field(default=None, ge=0, le=6)
    month: Optional[int] = Field(default=None, ge=1, le=12)
    disruption_active: Optional[bool] = None
    split: Optional[str] = None  # "train" | "val" | "test" — set by the time-based split


# --------------------------------------------------------------------------- #
# SUPPLIERS — synthesized (backend/services/preprocessing/suppliers.py);
# no public dataset carries capacity/cost/lead-time — see dataset_registry.yaml
# --------------------------------------------------------------------------- #
class SupplierRecord(BaseModel):
    supplier_id: str
    supplier_name: str
    region: str
    product_id: str
    capacity: int = Field(ge=0)
    unit_cost: float = Field(gt=0)
    lead_time_days: int = Field(ge=0)
    reliability: float = Field(ge=0, le=1)
    risk_level: RiskLevel
    status: SupplierStatus
    provenance: Provenance = Provenance.SYNTHETIC


# --------------------------------------------------------------------------- #
# ROUTES — port endpoints are real (World Port Index); distances are computed
# (haversine great-circle) with a documented Suez/Cape detour heuristic, since
# no real route-distance dataset was acquired — see
# backend/services/preprocessing/ports_routes.py
# --------------------------------------------------------------------------- #
class RouteRecord(BaseModel):
    route_id: str
    origin: str
    destination: str
    transport_mode: str
    distance_km: float = Field(gt=0)
    capacity: int = Field(ge=0)
    transit_time_days: float = Field(gt=0)
    cost_per_unit: float = Field(gt=0)
    status: RouteStatus
    provenance: Provenance = Provenance.DERIVED


class PortRecord(BaseModel):
    """Port master data — real, from the World Port Index (2017 NGA snapshot
    via HDX). Feeds RouteRecord.origin/destination as coordinates for the
    haversine distance calculation."""

    index_no: int
    port_name: str
    country: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    harbor_size: Optional[str] = None
    harbor_type: Optional[str] = None
    provenance: Provenance = Provenance.REAL


# --------------------------------------------------------------------------- #
# DISRUPTIONS — NOAA Storm Events (real, weather only) + a small curated log
# of real historical non-weather events (Suez 2021, etc.) — see
# backend/services/preprocessing/disruptions.py
# --------------------------------------------------------------------------- #
class DisruptionRecord(BaseModel):
    event_id: str
    event_type: str
    location: str
    start_date: datetime
    end_date: Optional[datetime] = None
    severity: RiskLevel
    affected_route: Optional[str] = None
    affected_supplier: Optional[str] = None
    estimated_delay_days: Optional[float] = None
    status: DisruptionStatus
    provenance: Provenance = Provenance.REAL

    @field_validator("end_date")
    @classmethod
    def end_after_start(cls, v: Optional[datetime], info):
        start = info.data.get("start_date")
        if v is not None and start is not None and v < start:
            raise ValueError("end_date cannot be before start_date")
        return v


# --------------------------------------------------------------------------- #
# TARIFFS — World Bank WDI weighted-mean applied tariff. Country-year grain,
# NOT the country-pair x product grain the brief's schema implies — flagged
# honestly in dataset_registry.yaml and here via `is_country_level_proxy`.
# --------------------------------------------------------------------------- #
class TariffRecord(BaseModel):
    origin_country: str
    destination_country: str  # "ANY" for this proxy dataset — see class docstring
    product_category: str     # "ALL" for this proxy dataset
    effective_year: int = Field(ge=1960, le=2100)
    tariff_rate: float = Field(ge=0)
    is_country_level_proxy: bool = True
    provenance: Provenance = Provenance.REAL
