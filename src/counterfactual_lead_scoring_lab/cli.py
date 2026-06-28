from __future__ import annotations

from pathlib import Path

import typer

from counterfactual_lead_scoring_lab.data import load_leads
from counterfactual_lead_scoring_lab.schemas import LeadTrainingConfig
from counterfactual_lead_scoring_lab.trainer import train_lead_model

app = typer.Typer(help="Train a lead scoring model with counterfactual planning hooks.")


@app.command()
def train(csv_path: Path = Path("examples/leads.csv")) -> None:
    report = train_lead_model(load_leads(csv_path), LeadTrainingConfig(min_rows=10))
    typer.echo(report.model_dump_json(indent=2))

