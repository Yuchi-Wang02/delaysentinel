"""Small prediction API used by the demo Space and by notebooks."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from .data import rule_predict
from .prompting import FIELDS, RULE_FIELDS, drop_fields, parse_label, serialize_row


def predict_fields(scorer, fields: Mapping[str, object], *, drop_rule_fields: bool = False) -> dict:
    """Score one record. Returns the raw text, the parsed label, and the rule's answer."""
    text = serialize_row(fields)
    if drop_rule_fields:
        text = drop_fields(text, RULE_FIELDS)
    raw = scorer.generate([text])[0]
    label = parse_label(raw)
    rule = int(rule_predict(pd.DataFrame([{k: fields.get(k, "") for k in FIELDS}]))[0])
    return {
        "prompt": text,
        "raw": raw,
        "label": label,
        "rule_label": rule,
        "agrees_with_rule": (label == rule) if label is not None else None,
    }


def predict_frame(scorer, frame: pd.DataFrame, *, drop_rule_fields: bool = False) -> pd.DataFrame:
    texts = [serialize_row(row) for row in frame.to_dict("records")]
    if drop_rule_fields:
        texts = [drop_fields(t, RULE_FIELDS) for t in texts]
    raw, labels = scorer.predict(texts)
    out = frame.copy()
    out["rule_label"] = rule_predict(frame.reindex(columns=list(FIELDS)).fillna(""))
    out["model_raw"] = raw
    out["model_label"] = labels
    out["agrees_with_rule"] = [
        (lab == r) if lab is not None else None for lab, r in zip(labels, out["rule_label"], strict=False)
    ]
    return out
