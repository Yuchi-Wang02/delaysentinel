"""Trivial and classical baselines evaluated on exactly the rows the model was tested on."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .data import CATEGORICAL_FIELDS, NUMERIC_FIELDS, rule_predict
from .prompting import RULE_FIELDS
from .stats import classification_metrics

FEATURE_ENCODING = {
    "dropped": ["Timestamp"],
    "numeric_standardized": list(NUMERIC_FIELDS),
    "one_hot": list(CATEGORICAL_FIELDS),
    "missing_reason": "kept as its own category 'None', exactly as it appears in the prompt",
    "without_rule_fields_variant": f"same, minus every one-hot column derived from {list(RULE_FIELDS)}",
}

BASELINE_NAMES = (
    "rule_delayed_or_heavy",
    "all_positive",
    "decision_tree_depth2",
    "logistic_regression",
    "gradient_boosting",
    "gradient_boosting_without_rule_fields",
)


def _columns(drop_rule_fields: bool) -> tuple[list[str], list[str]]:
    cats = [c for c in CATEGORICAL_FIELDS if not (drop_rule_fields and c in RULE_FIELDS)]
    return list(NUMERIC_FIELDS), cats


def _pipeline(estimator, drop_rule_fields: bool) -> Pipeline:
    nums, cats = _columns(drop_rule_fields)
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), nums),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
        ]
    )
    return Pipeline([("pre", pre), ("clf", estimator)])


def _estimator(name: str, seed: int):
    if name == "decision_tree_depth2":
        return DecisionTreeClassifier(max_depth=2, random_state=seed)
    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000)
    if name in ("gradient_boosting", "gradient_boosting_without_rule_fields"):
        return GradientBoostingClassifier(random_state=seed)
    raise KeyError(name)


def _features(frame: pd.DataFrame, drop_rule_fields: bool) -> pd.DataFrame:
    nums, cats = _columns(drop_rule_fields)
    out = frame[nums + cats].copy()
    for c in cats:
        out[c] = out[c].astype(str)
    return out


def evaluate_split(train: pd.DataFrame, test: pd.DataFrame, seed: int = 0, n_boot: int = 1000) -> dict:
    """Fit on the frozen train rows, score on the frozen test rows (same rows as the model)."""
    y_train = train["gold"].to_numpy()
    y_test = test["gold"].to_numpy()
    results: dict[str, dict] = {}
    results["rule_delayed_or_heavy"] = classification_metrics(y_test, rule_predict(test), n_boot=n_boot, seed=seed)
    results["all_positive"] = classification_metrics(y_test, np.ones_like(y_test), n_boot=n_boot, seed=seed)
    for name in (
        "decision_tree_depth2",
        "logistic_regression",
        "gradient_boosting",
        "gradient_boosting_without_rule_fields",
    ):
        drop = name.endswith("without_rule_fields")
        pipe = _pipeline(_estimator(name, seed), drop)
        pipe.fit(_features(train, drop), y_train)
        pred = pipe.predict(_features(test, drop))
        prob = pipe.predict_proba(_features(test, drop))[:, 1]
        results[name] = classification_metrics(y_test, pred, prob, n_boot=n_boot, seed=seed)
        if name == "decision_tree_depth2":
            results[name]["tree_rules"] = _tree_text(pipe)
    return results


def _tree_text(pipe: Pipeline) -> str:
    from sklearn.tree import export_text

    names = list(pipe.named_steps["pre"].get_feature_names_out())
    return export_text(pipe.named_steps["clf"], feature_names=names, show_weights=True)


def cross_validate(frame: pd.DataFrame, seed: int = 0, n_splits: int = 5, n_repeats: int = 3) -> dict:
    """Seeded repeated stratified CV over all 1,000 rows.

    This applies to the sklearn baselines only. The fine-tuned model has seen the 800
    training rows and can only be scored on the frozen test rows.
    """
    y = frame["gold"].to_numpy()
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    per_name: dict[str, dict[str, list[float]]] = {n: {"acc": [], "auroc": []} for n in BASELINE_NAMES}
    for fold, (tr, te) in enumerate(splitter.split(frame, y)):
        train, test = frame.iloc[tr], frame.iloc[te]
        per_name["rule_delayed_or_heavy"]["acc"].append(float((rule_predict(test) == y[te]).mean()))
        per_name["all_positive"]["acc"].append(float((np.ones_like(y[te]) == y[te]).mean()))
        for name in (
            "decision_tree_depth2",
            "logistic_regression",
            "gradient_boosting",
            "gradient_boosting_without_rule_fields",
        ):
            drop = name.endswith("without_rule_fields")
            pipe = _pipeline(_estimator(name, seed + fold), drop)
            pipe.fit(_features(train, drop), y[tr])
            pred = pipe.predict(_features(test, drop))
            prob = pipe.predict_proba(_features(test, drop))[:, 1]
            per_name[name]["acc"].append(float((pred == y[te]).mean()))
            from sklearn.metrics import roc_auc_score

            per_name[name]["auroc"].append(float(roc_auc_score(y[te], prob)))
    summary = {
        "scheme": f"RepeatedStratifiedKFold(n_splits={n_splits}, n_repeats={n_repeats}, random_state={seed})",
        "applies_to": "sklearn baselines only (the fine-tuned model has seen the training rows)",
        "n_rows": len(frame),
    }
    for name, vals in per_name.items():
        entry = {
            "acc_mean": round(float(np.mean(vals["acc"])), 4),
            "acc_min": round(float(np.min(vals["acc"])), 4),
            "acc_max": round(float(np.max(vals["acc"])), 4),
        }
        if vals["auroc"]:
            entry["auroc_mean"] = round(float(np.mean(vals["auroc"])), 4)
            entry["auroc_min"] = round(float(np.min(vals["auroc"])), 4)
            entry["auroc_max"] = round(float(np.max(vals["auroc"])), 4)
        summary[name] = entry
    return summary
