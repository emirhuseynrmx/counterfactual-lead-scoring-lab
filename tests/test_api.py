from litestar.testing import TestClient

from counterfactual_lead_scoring_lab.api import app


def test_health() -> None:
    with TestClient(app=app) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_explainability_contract() -> None:
    with TestClient(app=app) as client:
        payload = client.get("/v1/explainability-contract").json()

    assert payload["shap"].startswith("Global")
    assert "Counterfactual" in payload["dice"]

