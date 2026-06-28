from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from counterfactual_lead_scoring_lab.data import normalize_kaggle_leads

DATASET_SLUG = "amritachatterjee09/lead-scoring-dataset"
RAW_FILENAME = "Lead Scoring.csv"

app = typer.Typer(help="Download and prepare the Kaggle lead scoring dataset.")


def download_lead_dataset(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            DATASET_SLUG,
            "-p",
            str(data_dir),
            "--unzip",
        ],
        check=True,
    )
    raw_path = data_dir / RAW_FILENAME
    if not raw_path.exists():
        raise FileNotFoundError(f"Kaggle download completed but {RAW_FILENAME} was not found.")
    return raw_path


def prepare_lead_scoring(raw_path: Path, output_path: Path, max_rows: int = 1200) -> Path:
    normalized = normalize_kaggle_leads(pd.read_csv(raw_path)).head(max_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output_path, index=False, float_format="%.4f")
    return output_path


@app.command()
def prepare(
    data_dir: Annotated[Path, typer.Option(help="Directory for the Kaggle download.")] = Path(
        "data/raw/kaggle/lead-scoring"
    ),
    out: Annotated[Path, typer.Option(help="Prepared lead CSV path.")] = Path(
        "examples/leads.csv"
    ),
    rows: Annotated[int, typer.Option(help="Maximum rows to keep in the public sample.")] = 1200,
    skip_download: Annotated[
        bool,
        typer.Option(help="Use an existing raw Kaggle CSV in data_dir."),
    ] = False,
) -> None:
    raw_path = data_dir / RAW_FILENAME if skip_download else download_lead_dataset(data_dir)
    prepared = prepare_lead_scoring(raw_path, out, max_rows=rows)
    typer.echo(f"Prepared Kaggle lead scoring data at {prepared}")
