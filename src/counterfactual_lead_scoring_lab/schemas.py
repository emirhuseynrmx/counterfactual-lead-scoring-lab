from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class LeadRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    lead_id: str = Field(min_length=1, max_length=64)
    email: EmailStr
    company_size: int = Field(ge=1, le=250_000)
    annual_revenue_usd: float = Field(ge=0, le=10_000_000_000)
    website_visits_30d: int = Field(ge=0, le=1_000_000)
    demo_requests: int = Field(ge=0, le=100)
    email_opens_30d: int = Field(ge=0, le=1_000)
    support_fit_score: float = Field(ge=0, le=1)
    region: str = Field(min_length=2, max_length=32)
    converted: int = Field(ge=0, le=1)

    @field_validator("region")
    @classmethod
    def normalize_region(cls, value: str) -> str:
        return value.lower()


class LeadTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_rows: int = Field(default=30, ge=10)
    test_size: float = Field(default=0.25, gt=0, lt=0.5)
    random_state: int = Field(default=11, ge=0)


class TrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    csv_path: Path
    config: LeadTrainingConfig = Field(default_factory=LeadTrainingConfig)


class LeadModelReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: int
    conversion_rate: float
    roc_auc: float
    top_drivers: list[dict[str, float | str]]
    counterfactual_playbook: list[str]

