from __future__ import annotations

from pathlib import Path

import pandas as pd

from counterfactual_lead_scoring_lab.schemas import LeadRecord

FEATURE_COLUMNS = [
    "company_size",
    "annual_revenue_usd",
    "website_visits_30d",
    "demo_requests",
    "email_opens_30d",
    "support_fit_score",
    "region_code",
]


def load_leads(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    records = [LeadRecord.model_validate(row) for row in raw.to_dict(orient="records")]
    frame = pd.DataFrame([record.model_dump() for record in records])
    region_codes = {region: idx for idx, region in enumerate(sorted(frame["region"].unique()))}
    frame["region_code"] = frame["region"].map(region_codes).astype(float)
    return frame

