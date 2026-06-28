from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LeadRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    prospect_id: str = Field(min_length=1, max_length=96)
    lead_origin: str = Field(min_length=1, max_length=80)
    lead_source: str = Field(min_length=1, max_length=80)
    do_not_email: bool
    total_visits: float = Field(ge=0, le=1_000_000)
    time_on_site_seconds: int = Field(ge=0, le=10_000_000)
    page_views_per_visit: float = Field(ge=0, le=10_000)
    occupation: str = Field(min_length=1, max_length=96)
    city: str = Field(min_length=1, max_length=96)
    activity_score: float = Field(ge=0, le=100)
    profile_score: float = Field(ge=0, le=100)
    converted: int = Field(ge=0, le=1)

    @field_validator("lead_origin", "lead_source", "occupation", "city")
    @classmethod
    def normalize_label(cls, value: str) -> str:
        normalized = value.strip().lower().replace(" ", "_")
        return normalized if normalized and normalized != "select" else "unknown"


class LeadTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_rows: int = Field(default=100, ge=30)
    test_size: float = Field(default=0.25, gt=0, lt=0.5)
    random_state: int = Field(default=11, ge=0)


class TrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    csv_path: Path
    config: LeadTrainingConfig = Field(default_factory=LeadTrainingConfig)


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    csv_path: Path
    lead: LeadRecord
    config: LeadTrainingConfig = Field(default_factory=LeadTrainingConfig)


class LeadScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prospect_id: str
    conversion_probability: float
    tier: str
    top_reasons: list[str]
    recommended_actions: list[str]


class LeadModelReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: int
    conversion_rate: float
    roc_auc: float
    top_drivers: list[dict[str, float | str]]
    sample_scores: list[LeadScore]
    counterfactual_playbook: list[str]
