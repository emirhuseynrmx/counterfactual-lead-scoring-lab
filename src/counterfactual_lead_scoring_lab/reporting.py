from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from counterfactual_lead_scoring_lab.data import load_leads
from counterfactual_lead_scoring_lab.schemas import LeadTrainingConfig
from counterfactual_lead_scoring_lab.trainer import train_lead_model


def build_report_typ(csv_path: Path = Path("examples/leads.csv")) -> str:
    report = train_lead_model(load_leads(csv_path), LeadTrainingConfig())
    driver_rows = "\n".join(
        f'  [{item["feature"]}], [{item["importance"]}],'
        for item in report.top_drivers
    )
    playbook = "\n".join(f"- {item}" for item in report.counterfactual_playbook)
    rows_card = _metric_card("Rows", str(report.rows))
    conversion_card = _metric_card("Conversion", str(report.conversion_rate))
    auc_card = _metric_card("ROC-AUC", str(report.roc_auc))
    return f"""#set page(margin: 16mm)
#set text(font: "Arial", size: 10pt)

#text(size: 18pt, weight: "bold")[Counterfactual Lead Scoring Lab]

Model-training report for lead scoring with SHAP/DiCE planning.

#grid(columns: (1fr, 1fr, 1fr), gutter: 8pt)[
{rows_card}
][
{conversion_card}
][
{auc_card}
]

#v(10pt)
#text(size: 12pt, weight: "bold")[Top Lead Drivers]

#table(columns: (1fr, 1fr), [Feature], [Importance],
{driver_rows}
)

#v(8pt)
#text(size: 12pt, weight: "bold")[Counterfactual Playbook]
{playbook}
"""


def _metric_card(label: str, value: str) -> str:
    return (
        f'  #block(fill: rgb("#f3f6fb"), radius: 4pt, inset: 8pt)'
        f'[{label}\\ #text(size: 18pt, weight: "bold")[{value}]]'
    )


def write_report(output_dir: Path = Path("docs/samples")) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    typ_path = output_dir / "lead_scoring_report.typ"
    typ_path.write_text(build_report_typ(), encoding="utf-8")
    if shutil.which("typst"):
        subprocess.run(
            ["typst", "compile", str(typ_path), str(typ_path.with_suffix(".pdf"))],
            check=True,
        )
    return typ_path


def main() -> None:
    print(write_report())
