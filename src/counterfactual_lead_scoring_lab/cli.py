from __future__ import annotations

from enum import Enum
from pathlib import Path

import typer

from counterfactual_lead_scoring_lab.data import load_leads
from counterfactual_lead_scoring_lab.schemas import LeadTrainingConfig
from counterfactual_lead_scoring_lab.trainer import train_lead_model

app = typer.Typer(help="Train a lead scoring model with counterfactual planning hooks.")


class OutputFormat(str, Enum):
    summary = "summary"
    json = "json"


@app.command()
def train(
    csv_path: Path = Path("examples/leads.csv"),
    output: OutputFormat = OutputFormat.summary,
) -> None:
    """Train lead scoring model and display tier breakdown and counterfactual playbook."""
    report = train_lead_model(load_leads(csv_path), LeadTrainingConfig(min_rows=10))

    if output == OutputFormat.json:
        typer.echo(report.model_dump_json(indent=2))
        return

    typer.echo("")
    typer.echo("Counterfactual Lead Scoring Lab")
    typer.echo("=" * 45)
    typer.echo(f"  Leads trained on : {report.rows}")
    typer.echo(f"  Holdout rows     : {report.holdout_rows}")
    typer.echo(f"  Conversion rate  : {report.conversion_rate:.1%}")
    typer.echo(f"  ROC-AUC          : {report.roc_auc:.4f}")
    typer.echo(f"  Features used    : {report.feature_count}")

    typer.echo("")
    typer.echo("Top Conversion Drivers")
    typer.echo("-" * 45)
    typer.echo(f"  {'Feature':<32} Importance")
    typer.echo("  " + "-" * 40)
    for item in report.top_drivers[:6]:
        typer.echo(f"  {item['feature']!s:<32} {item['importance']:.5f}")

    if report.sample_scores:
        typer.echo("")
        typer.echo("Sample Lead Scores (holdout)")
        typer.echo("-" * 45)
        tier_icons = {"hot": ">>", "warm": "~~", "nurture": ".."}
        for score in report.sample_scores[:6]:
            icon = tier_icons.get(score.tier, " ")
            typer.echo(
                f"  {icon} [{score.tier.upper():<7}] "
                f"{score.prospect_id[:20]:<20}  "
                f"p={score.conversion_probability:.3f}"
            )
            if score.recommended_actions:
                typer.echo(f"    → {score.recommended_actions[0]}")

    typer.echo("")
    typer.echo("Counterfactual Playbook")
    typer.echo("-" * 45)
    for i, step in enumerate(report.counterfactual_playbook, 1):
        typer.echo(f"  {i}. {step}")

    typer.echo("")
    typer.echo("Evidence Contract")
    typer.echo("-" * 45)
    for check in report.evidence_checks:
        icon = "OK" if check.status == "pass" else "!!" if check.status == "review" else "--"
        typer.echo(f"  [{icon}] {check.check}: {check.evidence}")
    typer.echo("")
