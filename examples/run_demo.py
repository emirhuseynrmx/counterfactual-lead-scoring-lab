"""
End-to-end demo: train a lead scoring model on sample leads and print tier breakdown.

Usage:
    python examples/run_demo.py
    python examples/run_demo.py --csv examples/leads.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from counterfactual_lead_scoring_lab.data import load_leads
from counterfactual_lead_scoring_lab.schemas import LeadTrainingConfig
from counterfactual_lead_scoring_lab.trainer import train_lead_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("examples/leads.csv"))
    args = parser.parse_args()

    print(f"\nLoading leads: {args.csv}")
    frame = load_leads(args.csv)
    config = LeadTrainingConfig()

    print("Training GradientBoosting lead scoring model…\n")
    report = train_lead_model(frame, config)

    print("Counterfactual Lead Scoring Lab — Full Report")
    print("=" * 55)
    print(f"  Leads           : {report.rows}")
    print(f"  Holdout rows    : {report.holdout_rows}")
    print(f"  Conversion rate : {report.conversion_rate:.1%}")
    print(f"  ROC-AUC         : {report.roc_auc:.4f}")
    print(f"  Features used   : {report.feature_count}")

    print("\nTop Conversion Drivers")
    print("-" * 50)
    print(f"  {'Feature':<32} Importance")
    print("  " + "-" * 44)
    for item in report.top_drivers:
        print(f"  {item['feature']!s:<32} {item['importance']!s:>10}")

    if report.sample_scores:
        print("\nSample Lead Scores (holdout, first 8)")
        print("-" * 60)
        tier_icons = {"hot": "[HOT]", "warm": "[WRM]", "nurture": "[NUR]"}
        print(f"  {'Tier':<7} {'Prospect ID':<38} {'p(conv)':>8}")
        print("  " + "-" * 60)
        for score in report.sample_scores:
            tag = tier_icons.get(score.tier, "     ")
            print(
                f"  {tag:<7} {score.prospect_id[:36]:<38} {score.conversion_probability:>8.4f}"
            )

        # Show recommended actions for the first hot lead
        hot = next((s for s in report.sample_scores if s.tier == "hot"), None)
        if hot:
            print(f"\n  Actions for {hot.prospect_id[:20]}:")
            for action in hot.recommended_actions:
                print(f"    - {action}")

    print("\nCounterfactual Playbook")
    print("-" * 55)
    for i, step in enumerate(report.counterfactual_playbook, 1):
        print(f"  {i}. {step}")

    print("\nEvidence Contract")
    print("-" * 55)
    for check in report.evidence_checks:
        icon = "OK" if check.status == "pass" else "!!" if check.status == "review" else "--"
        print(f"  [{icon}] {check.check}: {check.evidence}")
    print()


if __name__ == "__main__":
    main()
