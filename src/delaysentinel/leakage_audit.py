"""Can a one-line rule reproduce the label? A generic scanner for tabular datasets.

    python -m delaysentinel.leakage_audit --csv data/smart_logistics_dataset.csv --target Logistics_Delay

For every column it lists single conditions (``column == value`` for categoricals,
``column <= threshold`` for numerics) whose rows are *pure* in the target, then greedily
combines pure-positive conditions into an OR-rule and reports how many labels that rule
reproduces. It also reports the cross-validated accuracy of a depth-2 decision tree,
which is the cheapest possible baseline. Run this before training anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier


def single_conditions(frame: pd.DataFrame, target: str, min_support: int = 20) -> list[dict]:
    y = frame[target].astype(int).to_numpy()
    conditions: list[dict] = []
    for column in frame.columns:
        if column == target:
            continue
        series = frame[column]
        numeric = pd.api.types.is_numeric_dtype(series)
        if not numeric or series.nunique() <= 12:
            for value, count in series.astype(str).value_counts().items():
                if count < min_support:
                    continue
                mask = (series.astype(str) == value).to_numpy()
                pos = int(y[mask].sum())
                conditions.append(
                    {
                        "condition": f'{column} == "{value}"',
                        "column": column,
                        "rows": int(mask.sum()),
                        "positives": pos,
                        "purity": round(max(pos, int(mask.sum()) - pos) / int(mask.sum()), 4),
                        "pure_label": 1 if pos == mask.sum() else (0 if pos == 0 else None),
                    }
                )
        else:
            values = series.to_numpy(dtype=float)
            ok = ~np.isnan(values)
            if ok.sum() < min_support:
                continue
            stump = DecisionTreeClassifier(max_depth=1).fit(values[ok].reshape(-1, 1), y[ok])
            if stump.tree_.node_count < 3:
                continue
            threshold = float(stump.tree_.threshold[0])
            for name, mask in (
                (f"{column} <= {threshold:.4g}", values <= threshold),
                (f"{column} > {threshold:.4g}", values > threshold),
            ):
                mask = mask & ok
                if mask.sum() < min_support:
                    continue
                pos = int(y[mask].sum())
                conditions.append(
                    {
                        "condition": name,
                        "column": column,
                        "rows": int(mask.sum()),
                        "positives": pos,
                        "purity": round(max(pos, int(mask.sum()) - pos) / int(mask.sum()), 4),
                        "pure_label": 1 if pos == mask.sum() else (0 if pos == 0 else None),
                    }
                )
    return sorted(conditions, key=lambda c: (-c["purity"], -c["rows"]))


def _mask_for(frame: pd.DataFrame, condition: dict) -> np.ndarray:
    column = condition["column"]
    text = condition["condition"]
    if " == " in text:
        value = text.split(" == ", 1)[1].strip('"')
        return (frame[column].astype(str) == value).to_numpy()
    op, threshold = (" <= ", None) if " <= " in text else (" > ", None)
    threshold = float(text.split(op, 1)[1])
    values = frame[column].to_numpy(dtype=float)
    return (values <= threshold) if op == " <= " else (values > threshold)


def greedy_or_rule(frame: pd.DataFrame, target: str, conditions: list[dict]) -> dict:
    """Greedily OR together pure-positive conditions to cover the positives."""
    y = frame[target].astype(int).to_numpy()
    pure_pos = [c for c in conditions if c["pure_label"] == 1]
    covered = np.zeros(len(frame), dtype=bool)
    chosen: list[str] = []
    while True:
        best, best_gain = None, 0
        for c in pure_pos:
            gain = int((_mask_for(frame, c) & ~covered).sum())
            if gain > best_gain:
                best, best_gain = c, gain
        if best is None or best_gain == 0:
            break
        chosen.append(best["condition"])
        covered |= _mask_for(frame, best)
        if covered[y == 1].all():
            break
    pred = covered.astype(int)
    return {
        "rule": " OR ".join(chosen) if chosen else None,
        "n_conditions": len(chosen),
        "accuracy": round(float((pred == y).mean()), 4),
        "mismatches": int((pred != y).sum()),
        "positives_covered": int(covered[y == 1].sum()),
        "positives_total": int(y.sum()),
    }


def tree_cv(frame: pd.DataFrame, target: str, depth: int = 2, seed: int = 0) -> dict:
    y = frame[target].astype(int).to_numpy()
    X = pd.get_dummies(
        frame.drop(columns=[target]).astype(
            {c: str for c in frame.columns if c != target and not pd.api.types.is_numeric_dtype(frame[c])}
        )
    )
    X = X.fillna(X.median(numeric_only=True))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(DecisionTreeClassifier(max_depth=depth, random_state=seed), X, y, cv=cv)
    return {
        "depth": depth,
        "cv_accuracy_mean": round(float(scores.mean()), 4),
        "cv_accuracy_min": round(float(scores.min()), 4),
    }


def audit(frame: pd.DataFrame, target: str) -> dict:
    conditions = single_conditions(frame, target)
    return {
        "n_rows": len(frame),
        "target": target,
        "positive_rate": round(float(frame[target].astype(int).mean()), 4),
        "pure_conditions": [c for c in conditions if c["pure_label"] is not None],
        "greedy_or_rule": greedy_or_rule(frame, target, conditions),
        "depth2_tree": tree_cv(frame, target),
    }


def to_markdown(report: dict) -> str:
    lines = [
        f"# Leakage audit: target `{report['target']}` ({report['n_rows']} rows, positive rate {report['positive_rate']})",  # noqa: E501
        "",
    ]
    lines.append("## Single conditions whose rows are pure in the target")
    lines.append("")
    lines.append("| condition | rows | positives | pure label |")
    lines.append("| --- | ---: | ---: | --- |")
    for c in report["pure_conditions"]:
        lines.append(f"| `{c['condition']}` | {c['rows']} | {c['positives']} | {c['pure_label']} |")
    if not report["pure_conditions"]:
        lines.append("| (none) | | | |")
    g = report["greedy_or_rule"]
    lines += ["", "## Greedy OR-rule over pure-positive conditions", ""]
    lines.append(f"- rule: `{g['rule']}`")
    lines.append(
        f"- accuracy {g['accuracy']} on all rows, {g['mismatches']} mismatches; positives covered {g['positives_covered']}/{g['positives_total']}"  # noqa: E501
    )
    t = report["depth2_tree"]
    lines += [
        "",
        f"## Depth-{t['depth']} decision tree, 5-fold CV",
        "",
        f"- mean accuracy {t['cv_accuracy_mean']} (min fold {t['cv_accuracy_min']})",
        "",
    ]
    lines.append(
        "If a one-line rule or a depth-2 tree already reproduces the label, a larger model cannot add anything on this table."  # noqa: E501
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    args = parser.parse_args(argv)
    frame = pd.read_csv(args.csv, keep_default_na=False, na_values=[""])
    report = audit(frame, args.target)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=1), encoding="utf-8")
    md = to_markdown(report)
    if args.out_md:
        Path(args.out_md).write_text(md, encoding="utf-8")
    print(md)
    return report


if __name__ == "__main__":
    main()
