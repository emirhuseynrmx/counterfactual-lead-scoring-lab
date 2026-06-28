from counterfactual_lead_scoring_lab.reporting import build_report_typ


def test_report_contains_counterfactual_playbook() -> None:
    report = build_report_typ()

    assert "Counterfactual Lead Scoring Lab" in report
    assert "Counterfactual Playbook" in report
    assert "ROC-AUC" in report
