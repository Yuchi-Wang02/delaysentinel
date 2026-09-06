"""Data access for the frozen historical split and the two-clause label rule."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .prompting import FIELDS, LABEL_RE, MISSING_TOKEN, TARGET

NUMERIC_FIELDS: tuple[str, ...] = (
    "Latitude",
    "Longitude",
    "Inventory_Level",
    "Temperature",
    "Humidity",
    "Waiting_Time",
    "User_Transaction_Amount",
    "User_Purchase_Frequency",
    "Asset_Utilization",
    "Demand_Forecast",
)
CATEGORICAL_FIELDS: tuple[str, ...] = (
    "Asset_ID",
    "Shipment_Status",
    "Traffic_Status",
    "Logistics_Delay_Reason",
)

RULE_TEXT = 'Logistics_Delay = 1  iff  Shipment_Status == "Delayed"  OR  Traffic_Status == "Heavy"'


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def user_text(row: dict) -> str:
    return row["conversations"][1]["value"]


def gold_label(row: dict) -> int:
    answer = row["conversations"][2]["value"]
    match = LABEL_RE.match(answer)
    if match is None:
        raise ValueError(f"unparsable gold answer: {answer!r}")
    return int(match.group(1))


def frame_from_jsonl(rows: Iterable[dict]) -> pd.DataFrame:
    """Parse the user turns back into a table with a ``gold`` column."""
    from .prompting import parse_user_text

    rows = list(rows)
    frame = pd.DataFrame([parse_user_text(user_text(r)) for r in rows])
    frame = frame.reindex(columns=list(FIELDS))
    for column in NUMERIC_FIELDS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["gold"] = [gold_label(r) for r in rows]
    return frame


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load the Kaggle CSV keeping the literal ``None`` strings that reach the prompt."""
    frame = pd.read_csv(path, keep_default_na=False, na_values=[""])
    for column in NUMERIC_FIELDS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[TARGET] = frame[TARGET].astype(int)
    return frame


def rule_predict(frame: pd.DataFrame) -> np.ndarray:
    """The deterministic rule that reproduces every label in the Kaggle table."""
    delayed = frame["Shipment_Status"].astype(str).str.strip() == "Delayed"
    heavy = frame["Traffic_Status"].astype(str).str.strip() == "Heavy"
    return (delayed | heavy).astype(int).to_numpy()


def rule_crosstab(frame: pd.DataFrame, label_column: str = TARGET) -> dict:
    """Counts behind the rule: rows and delayed rows for each clause, for the remainder,
    and for every Shipment_Status x Traffic_Status cell."""
    status = frame["Shipment_Status"].astype(str)
    traffic = frame["Traffic_Status"].astype(str)
    delayed = status == "Delayed"
    heavy = traffic == "Heavy"
    y = frame[label_column].astype(int)
    neither = ~(delayed | heavy)
    cells = {}
    for s in sorted(status.unique()):
        for t in sorted(traffic.unique()):
            mask = (status == s) & (traffic == t)
            cells[f"{s} x {t}"] = {"rows": int(mask.sum()), "delayed": int(y[mask].sum())}
    delivered_heavy = (status == "Delivered") & heavy
    return {
        "shipment_status_delayed": {"rows": int(delayed.sum()), "delayed": int(y[delayed].sum())},
        "traffic_status_heavy": {"rows": int(heavy.sum()), "delayed": int(y[heavy].sum())},
        "both": {"rows": int((delayed & heavy).sum()), "delayed": int(y[delayed & heavy].sum())},
        "neither": {"rows": int(neither.sum()), "delayed": int(y[neither].sum())},
        "delivered_and_heavy": {"rows": int(delivered_heavy.sum()), "delayed": int(y[delivered_heavy].sum())},
        "status_x_traffic_cells": cells,
        "mismatches": int((rule_predict(frame) != y.to_numpy()).sum()),
        "n": len(frame),
    }


def category_counts(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for column in CATEGORICAL_FIELDS:
        counts = frame[column].astype(str).replace({"nan": MISSING_TOKEN}).value_counts()
        out[column] = {str(k): int(v) for k, v in counts.items()}
    return out


def sha256_file(path: str | Path) -> str:
    """sha256 of the file bytes exactly as they are on disk."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_normalized_newlines(path: str | Path) -> str:
    """sha256 after mapping CRLF to LF, so a Windows and a Linux checkout agree.

    The values recorded in ``data/SPLIT.md`` are computed this way; on an LF checkout they
    equal :func:`sha256_file`.
    """
    with open(path, "rb") as handle:
        data = handle.read().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()
