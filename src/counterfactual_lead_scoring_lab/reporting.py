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
        f'  [{_typ_text(item["feature"])}], [{float(item["importance"]):.4f}],'
        for item in report.top_drivers
    )
    score_rows = "\n".join(_score_row(score.model_dump()) for score in report.sample_scores[:6])
    playbook = "\n".join(f"- {item}" for item in report.counterfactual_playbook)
    rows_card = _metric_card("Rows", str(report.rows))
    conversion_card = _metric_card("Conversion", f"{report.conversion_rate:.1%}")
    auc_card = _metric_card("Holdout ROC-AUC", f"{report.roc_auc:.3f}")
    return f"""#set page(margin: 16mm)
#set text(font: "Arial", size: 10pt)

#let muted = rgb("#667085")
#let panel = rgb("#f6f8fb")

#text(size: 18pt, weight: "bold")[Counterfactual Lead Scoring Report]

#text(fill: muted)[
  Lead routing report for a provided CSV. The model ranks leads by conversion
  probability and keeps counterfactual guidance limited to fields a sales or
  marketing team can reasonably influence.
]

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
#text(size: 12pt, weight: "bold")[Sample Lead Queue]

#table(
  columns: (1.2fr, .8fr, .7fr, 1.6fr),
  [Prospect], [Probability], [Tier], [Recommended action],
{score_rows}
)

#v(8pt)
#text(size: 12pt, weight: "bold")[Counterfactual Playbook]
{playbook}

#v(8pt)
#text(size: 12pt, weight: "bold")[Model Risk Notes]

- Reported ROC-AUC is measured on a held-out split.
- Do not use post-sale fields, manually assigned sales outcomes, or future
  activity columns as model inputs.
- Counterfactual guidance should not change immutable or protected customer attributes.
"""


def _metric_card(label: str, value: str) -> str:
    return (
        f'  #block(fill: panel, radius: 4pt, inset: 8pt)'
        f'[{label}\\ #text(size: 18pt, weight: "bold")[{value}]]'
    )


def _score_row(score: dict[str, object]) -> str:
    actions = score.get("recommended_actions")
    if isinstance(actions, list) and actions:
        action = str(actions[0])
    else:
        action = "Review lead before routing."
    return (
        f'  [{_typ_text(score.get("prospect_id"))}],'
        f' [{float(score.get("conversion_probability", 0)):.1%}],'
        f' [{_typ_text(score.get("tier"))}],'
        f' [{_typ_text(action)}],'
    )


def _typ_text(value: object) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": "\\\\",
        "[": "\\[",
        "]": "\\]",
        "#": "\\#",
        "$": "\\$",
        "@": "\\@",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


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
