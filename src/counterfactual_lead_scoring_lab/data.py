from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa
from pandera import Check

from counterfactual_lead_scoring_lab.schemas import LeadRecord

NUMERIC_FEATURES = [
    "total_visits",
    "time_on_site_seconds",
    "page_views_per_visit",
    "activity_score",
    "profile_score",
]
CATEGORICAL_FEATURES = ["lead_origin", "lead_source", "occupation", "city"]
BOOLEAN_FEATURES = ["do_not_email"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES

LEAD_FRAME_SCHEMA = pa.DataFrameSchema(
    {
        "prospect_id": pa.Column(str, unique=True),
        "lead_origin": pa.Column(str),
        "lead_source": pa.Column(str),
        "do_not_email": pa.Column(bool),
        "total_visits": pa.Column(float, Check.ge(0)),
        "time_on_site_seconds": pa.Column(int, Check.ge(0)),
        "page_views_per_visit": pa.Column(float, Check.ge(0)),
        "occupation": pa.Column(str),
        "city": pa.Column(str),
        "activity_score": pa.Column(float, Check.in_range(0, 100)),
        "profile_score": pa.Column(float, Check.in_range(0, 100)),
        "converted": pa.Column(int, Check.isin([0, 1])),
    },
    strict=True,
)


def load_leads(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    frame = normalize_kaggle_leads(raw) if "Prospect ID" in raw.columns else raw
    records = [LeadRecord.model_validate(row) for row in frame.to_dict(orient="records")]
    validated = pd.DataFrame([record.model_dump() for record in records])
    return LEAD_FRAME_SCHEMA.validate(validated)


def lead_to_frame(lead: LeadRecord) -> pd.DataFrame:
    return pd.DataFrame([lead.model_dump()])[FEATURE_COLUMNS]


def normalize_kaggle_leads(raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prospect_id": raw["Prospect ID"].astype(str),
            "lead_origin": raw["Lead Origin"].map(_clean_label),
            "lead_source": raw["Lead Source"].map(_clean_label),
            "do_not_email": raw["Do Not Email"].map(_yes_no),
            "total_visits": _number(raw["TotalVisits"]),
            "time_on_site_seconds": _number(raw["Total Time Spent on Website"]).round().astype(int),
            "page_views_per_visit": _number(raw["Page Views Per Visit"]),
            "occupation": raw["What is your current occupation"].map(_clean_label),
            "city": raw["City"].map(_clean_label),
            "activity_score": _number(raw["Asymmetrique Activity Score"]).clip(0, 100),
            "profile_score": _number(raw["Asymmetrique Profile Score"]).clip(0, 100),
            "converted": raw["Converted"].fillna(0).astype(int),
        }
    )


def _number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(float)


def _clean_label(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    normalized = str(value).strip().lower().replace(" ", "_")
    return normalized if normalized and normalized != "select" else "unknown"


def _yes_no(value: Any) -> bool:
    return str(value).strip().lower() == "yes"
