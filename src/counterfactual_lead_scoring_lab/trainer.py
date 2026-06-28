from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from counterfactual_lead_scoring_lab.data import FEATURE_COLUMNS
from counterfactual_lead_scoring_lab.schemas import LeadModelReport, LeadTrainingConfig


def train_lead_model(frame: pd.DataFrame, config: LeadTrainingConfig) -> LeadModelReport:
    if len(frame) < config.min_rows:
        raise ValueError(f"lead scoring needs at least {config.min_rows} rows")

    target = frame["converted"].astype(int)
    if target.nunique() != 2:
        raise ValueError("dataset must contain converted and non-converted leads")

    x_train, x_test, y_train, y_test = train_test_split(
        frame[FEATURE_COLUMNS],
        target,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=target,
    )
    model = GradientBoostingClassifier(random_state=config.random_state)
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(y_test, probabilities)
    drivers = _drivers(model.feature_importances_)

    return LeadModelReport(
        rows=len(frame),
        conversion_rate=round(float(target.mean()), 4),
        roc_auc=round(float(auc), 4),
        top_drivers=drivers,
        counterfactual_playbook=_playbook(drivers),
    )


def _drivers(importances: np.ndarray) -> list[dict[str, float | str]]:
    order = np.argsort(importances)[::-1][:5]
    return [
        {"feature": FEATURE_COLUMNS[index], "importance": round(float(importances[index]), 5)}
        for index in order
    ]


def _playbook(drivers: list[dict[str, float | str]]) -> list[str]:
    if not drivers:
        return ["Collect more lead history before generating counterfactual actions."]
    primary = str(drivers[0]["feature"])
    return [
        f"Use DiCE to search feasible changes around {primary}.",
        "Keep immutable fields fixed; only test actionable fields such as visits, opens, "
        "and demo requests.",
        "Review SHAP global drivers before using the score in sales routing.",
    ]
