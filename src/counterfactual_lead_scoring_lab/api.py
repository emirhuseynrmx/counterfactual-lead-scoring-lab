from __future__ import annotations

from litestar import Litestar, get, post

from counterfactual_lead_scoring_lab.data import load_leads
from counterfactual_lead_scoring_lab.schemas import LeadModelReport, TrainRequest
from counterfactual_lead_scoring_lab.trainer import train_lead_model


@get("/health", sync_to_thread=False)
def health() -> dict[str, str]:
    return {"status": "ok"}


@post("/v1/train", sync_to_thread=False)
def train(data: TrainRequest) -> LeadModelReport:
    return train_lead_model(load_leads(data.csv_path), data.config)


@get("/v1/explainability-contract", sync_to_thread=False)
def explainability_contract() -> dict[str, str]:
    return {
        "shap": "Global and per-lead driver explanations for score review.",
        "dice": "Counterfactual lead actions with immutable fields fixed.",
    }


app = Litestar(route_handlers=[health, train, explainability_contract], debug=False)
