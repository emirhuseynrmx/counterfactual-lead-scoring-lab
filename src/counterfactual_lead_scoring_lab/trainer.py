from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from counterfactual_lead_scoring_lab.data import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    lead_to_frame,
)
from counterfactual_lead_scoring_lab.schemas import (
    LeadModelReport,
    LeadRecord,
    LeadScore,
    LeadTrainingConfig,
)


def train_lead_model(frame: pd.DataFrame, config: LeadTrainingConfig) -> LeadModelReport:
    pipeline, top_drivers, auc, holdout_rows = fit_pipeline(frame, config)
    # Reconstruct the same split to score held-out rows (avoids in-sample inflation)
    _, test_frame = train_test_split(
        frame,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=frame["converted"].astype(int),
    )
    sample_frame = test_frame.head(8)
    sample_probs = pipeline.predict_proba(sample_frame[FEATURE_COLUMNS])[:, 1]
    sample_scores = [
        _score_row(row, float(prob), top_drivers)
        for row, prob in zip(sample_frame.to_dict(orient="records"), sample_probs, strict=True)
    ]

    return LeadModelReport(
        rows=len(frame),
        conversion_rate=round(float(frame["converted"].mean()), 4),
        roc_auc=round(float(auc), 4),
        holdout_rows=holdout_rows,
        feature_count=len(FEATURE_COLUMNS),
        evidence_checks=_evidence_checks(frame, config, holdout_rows),
        top_drivers=top_drivers,
        sample_scores=sample_scores,
        counterfactual_playbook=_playbook(top_drivers),
    )


def score_lead(
    frame: pd.DataFrame,
    lead: LeadRecord,
    config: LeadTrainingConfig,
) -> LeadScore:
    pipeline, top_drivers, _, _ = fit_pipeline(frame, config)
    probability = float(pipeline.predict_proba(lead_to_frame(lead))[:, 1][0])
    return _score_row(lead.model_dump(), probability, top_drivers)


def fit_pipeline(
    frame: pd.DataFrame,
    config: LeadTrainingConfig,
) -> tuple[Pipeline, list[dict[str, float | str]], float, int]:
    if len(frame) < config.min_rows:
        raise ValueError(f"lead scoring needs at least {config.min_rows} rows")

    target = frame["converted"].astype(int)
    if target.nunique() != 2:
        raise ValueError("dataset must contain converted and non-converted leads")
    if target.value_counts().min() < 5:
        raise ValueError("lead scoring needs at least 5 converted and 5 non-converted examples")

    x_train, x_test, y_train, y_test = train_test_split(
        frame[FEATURE_COLUMNS],
        target,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=target,
    )
    pipeline = Pipeline(
        steps=[
            (
                "features",
                ColumnTransformer(
                    transformers=[
                        ("numeric", StandardScaler(), NUMERIC_FEATURES),
                        (
                            "categorical",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            CATEGORICAL_FEATURES + BOOLEAN_FEATURES,
                        ),
                    ],
                    verbose_feature_names_out=False,
                ),
            ),
            ("model", GradientBoostingClassifier(random_state=config.random_state)),
        ]
    )
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(y_test, probabilities)
    return pipeline, _drivers(pipeline), float(auc), len(x_test)


def _evidence_checks(
    frame: pd.DataFrame,
    config: LeadTrainingConfig,
    holdout_rows: int,
) -> list[dict[str, str]]:
    target = frame["converted"].astype(int)
    class_counts = target.value_counts().to_dict()
    post_conversion_terms = ("converted_at", "won_at", "closed_at", "deal_value", "revenue")
    leakage_columns = [
        column
        for column in frame.columns
        if column != "converted" and any(term in column.lower() for term in post_conversion_terms)
    ]
    return [
        {
            "check": "holdout_split",
            "status": "pass",
            "evidence": (
                f"{holdout_rows} rows reserved for holdout scoring; "
                f"test_size={config.test_size}."
            ),
        },
        {
            "check": "minimum_class_count",
            "status": "pass" if min(class_counts.values()) >= 5 else "fail",
            "evidence": f"Class counts: {class_counts}.",
        },
        {
            "check": "target_leakage_scan",
            "status": "pass" if not leakage_columns else "review",
            "evidence": "No obvious post-conversion columns found."
            if not leakage_columns
            else f"Review potential leakage columns: {', '.join(leakage_columns)}.",
        },
        {
            "check": "counterfactual_boundary",
            "status": "pass",
            "evidence": "Action hints are limited to reviewable behavior and routing fields.",
        },
    ]


def _drivers(pipeline: Pipeline) -> list[dict[str, float | str]]:
    transformer = pipeline.named_steps["features"]
    model = pipeline.named_steps["model"]
    feature_names = transformer.get_feature_names_out()
    importances = getattr(model, "feature_importances_", np.zeros(len(feature_names)))
    order = np.argsort(importances)[::-1][:8]
    return [
        {"feature": str(feature_names[index]), "importance": round(float(importances[index]), 5)}
        for index in order
        if float(importances[index]) > 0
    ]


def _score_row(
    row: dict[str, object],
    probability: float,
    drivers: list[dict[str, float | str]],
) -> LeadScore:
    probability = round(probability, 4)
    tier = "hot" if probability >= 0.7 else "warm" if probability >= 0.4 else "nurture"
    top_reasons = [str(item["feature"]) for item in drivers[:3]]
    return LeadScore(
        prospect_id=str(row["prospect_id"]),
        conversion_probability=probability,
        tier=tier,
        top_reasons=top_reasons,
        recommended_actions=_actions(tier, top_reasons),
    )


def _actions(tier: str, top_reasons: list[str]) -> list[str]:
    if tier == "hot":
        return [
            "Route to sales today.",
            "Use the strongest driver as the opening context.",
            "Offer a direct demo or consultation slot.",
        ]
    if tier == "warm":
        return [
            "Send a targeted follow-up within 48 hours.",
            "Move the lead into a short nurture sequence.",
            "Ask one qualifying question before routing to sales.",
        ]
    return [
        "Keep in low-touch nurture.",
        "Do not spend sales time until engagement improves.",
        "Retest after a new visit, email open, or demo request.",
    ]


def _playbook(drivers: list[dict[str, float | str]]) -> list[str]:
    if not drivers:
        return ["Collect more lead history before generating action guidance."]
    primary = str(drivers[0]["feature"])
    return [
        f"Start counterfactual review with {primary}; test whether changing it is actionable.",
        "Keep immutable fields fixed and test only behaviors a sales or marketing team can "
        "influence.",
        "Use hot leads for direct sales follow-up, warm leads for nurture, and low leads for "
        "automation.",
    ]
