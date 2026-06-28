from counterfactual_lead_scoring_lab.reporting import build_report_typ


def test_report_contains_counterfactual_playbook() -> None:
    report = build_report_typ()

    assert "Counterfactual Lead Scoring Report" in report
    assert "Sample Lead Queue" in report
    assert "Counterfactual Playbook" in report
    assert "held-out split" in report
    assert "ROC-AUC" in report
