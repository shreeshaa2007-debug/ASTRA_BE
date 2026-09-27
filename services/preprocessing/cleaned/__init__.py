"""The cleaned data layer: `data/raw` -> `data/cleaned`, ready for downstream modeling and, later, SAP HANA Cloud.

This package sits beside the Phase 3 pipeline (`backend/services/preprocessing/pipeline.py`) and does not replace it:
`data/processed/` is what the running application reads today and is left exactly as it was. The cleaned layer fixes the
defects the data audit found (cancelled orders counted as demand, a non-unique inventory key, disruption state stored in the
supplier master, sea distances that were straight lines across land, missing Cape alternatives) and labels every record with
where it came from.

    python -m backend.services.preprocessing.cleaned.build

Provenance labels, one per record, in upper case (the Phase 3 files use lower-case `real | derived | synthetic`; the new
layer adds `AUTHORED` and is deliberately a different spelling so the two cannot be mixed up):

    REAL       measured, from a public dataset in data/raw, and carried through unchanged
    DERIVED    computed from REAL data by a documented rule (netting, aggregation, a formula)
    SYNTHETIC  generated because no public source exists; an assumption, never evidence
    AUTHORED   written by hand by the project about a publicly reported real-world event; not machine-verified here

A row carries the provenance of the fact it asserts. Where a REAL row also carries a DERIVED or SYNTHETIC column, that column
is named in `data_dictionary.csv`, which lists the provenance of every column.
"""
from __future__ import annotations

from pathlib import Path

CLEANED_DIR = Path("data/cleaned")
INTERIM_DIR = Path("data/interim/cleaning")  # audit artifacts that explain the cleaning but are not modeling inputs
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")  # read for comparison only; never written by this package

REAL = "REAL"
DERIVED = "DERIVED"
SYNTHETIC = "SYNTHETIC"
AUTHORED = "AUTHORED"
PROVENANCE_LABELS = (REAL, DERIVED, SYNTHETIC, AUTHORED)
