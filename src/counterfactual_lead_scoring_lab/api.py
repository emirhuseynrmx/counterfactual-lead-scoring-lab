from __future__ import annotations

from litestar import Litestar, get, post

from counterfactual_lead_scoring_lab.data import load_leads
from counterfactual_lead_scoring_lab.schemas import (
    LeadModelReport,
    LeadScore,
    ScoreRequest,
    TrainRequest,
)
from counterfactual_lead_scoring_lab.trainer import score_lead, train_lead_model


@get("/health", sync_to_thread=False)
def health() -> dict[str, str]:
    return {"status": "ok"}


@post("/v1/train", sync_to_thread=False)
def train(data: TrainRequest) -> LeadModelReport:
    return train_lead_model(load_leads(data.csv_path), data.config)


@post("/v1/score", sync_to_thread=False)
def score(data: ScoreRequest) -> LeadScore:
    return score_lead(load_leads(data.csv_path), data.lead, data.config)


@get("/v1/explainability-contract", sync_to_thread=False)
def explainability_contract() -> dict[str, str]:
    return {
        "drivers": "Global model drivers are extracted from the trained feature pipeline.",
        "counterfactuals": (
            "Action hints keep immutable fields fixed and focus on behavior changes."
        ),
    }


app = Litestar(route_handlers=[health, train, score, explainability_contract], debug=False)
