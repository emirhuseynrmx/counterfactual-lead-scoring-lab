# Counterfactual Lead Scoring Lab

Litestar service for training lead scoring models with Pydantic v2 validation, SHAP-style driver review, and DiCE counterfactual planning.

```mermaid
flowchart LR
    A[Lead CSV] --> B[Pydantic validation]
    B --> C[Feature builder]
    C --> D[Gradient boosting model]
    D --> E[Holdout ROC-AUC]
    D --> F[SHAP lead drivers]
    D --> G[DiCE action search]
    E --> H[Lead scoring report]
    F --> H
    G --> H
```

## What It Solves

Lead scoring is weak when it only returns a probability. This project keeps the useful parts: validated input, trained model, explainable drivers, and counterfactual actions that can be reviewed before routing leads to sales.

## Stack

Litestar, Pydantic v2, pandas, scikit-learn, SHAP, DiCE, Typer, pytest, ruff.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
python -m pytest
python -m ruff check .
lead-score-lab train
```

Run the API:

```bash
uvicorn counterfactual_lead_scoring_lab.api:app --reload
```

## API

| Endpoint | Purpose |
|---|---|
| `POST /v1/train` | Train a lead scoring model from a local CSV path |
| `GET /v1/explainability-contract` | Document SHAP and DiCE use |
| `GET /health` | Health check |

