from pathlib import Path

from counterfactual_lead_scoring_lab.data import load_leads
from counterfactual_lead_scoring_lab.schemas import LeadTrainingConfig
from counterfactual_lead_scoring_lab.trainer import train_lead_model


def test_train_lead_model() -> None:
    frame = load_leads(Path("examples/leads.csv"))
    report = train_lead_model(frame, LeadTrainingConfig(min_rows=10, test_size=0.3))

    assert report.rows >= 30
    assert 0 <= report.roc_auc <= 1
    assert report.top_drivers
    assert any("DiCE" in item for item in report.counterfactual_playbook)

