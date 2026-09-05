"""Prompt format used to fine-tune DelaySentinel, plus helpers to parse and edit prompts.

Everything here mirrors the original ``Transform.py`` (Sept 2025) byte for byte:
the system prompt, the ``Column: value`` serialisation in CSV column order, and the
assistant answer ``Logistics_Delay: 0|1``. Missing values reach the prompt as the
literal string ``None`` because that is what the Kaggle CSV contains.
"""

from __future__ import annotations

import math
import random
import re
from collections.abc import Iterable, Mapping, Sequence

SYSTEM_PROMPT = (
    "Assume you are a supply chain analyst. Based on the following information, "
    "output the result for Logistics_Delay, where 1 represents a delay and 0 represents no delay."
)

#: The 15 input columns, in the order they appear in the Kaggle CSV and in every prompt.
FIELDS: tuple[str, ...] = (
    "Timestamp",
    "Asset_ID",
    "Latitude",
    "Longitude",
    "Inventory_Level",
    "Shipment_Status",
    "Temperature",
    "Humidity",
    "Traffic_Status",
    "Waiting_Time",
    "User_Transaction_Amount",
    "User_Purchase_Frequency",
    "Logistics_Delay_Reason",
    "Asset_Utilization",
    "Demand_Forecast",
)
TARGET = "Logistics_Delay"
#: The two columns that determine the label exactly (see :func:`delaysentinel.data.rule_predict`).
RULE_FIELDS: tuple[str, str] = ("Shipment_Status", "Traffic_Status")
MISSING_TOKEN = "None"

#: Anchored on purpose: a stray "1" anywhere in the output must not count as a prediction.
LABEL_RE = re.compile(r"^\s*Logistics_Delay:\s*([01])\b")
ANSWER_PREFIX = "Logistics_Delay:"


def _clean(value: object) -> str:
    if value is None:
        return MISSING_TOKEN
    if isinstance(value, float) and math.isnan(value):
        return MISSING_TOKEN
    text = str(value).strip()
    return text if text != "" else MISSING_TOKEN


def serialize_row(row: Mapping[str, object], fields: Sequence[str] = FIELDS) -> str:
    """Render one record as the multi-line ``Column: value`` user turn used in training."""
    return "\n".join(f"{field}: {_clean(row.get(field, ''))}" for field in fields)


def answer_text(label: int) -> str:
    return f"{ANSWER_PREFIX} {int(label)}"


def build_messages(user_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]


def parse_label(text: str | None) -> int | None:
    """Return 0/1 for a well-formed answer, else ``None`` (never default to 0)."""
    match = LABEL_RE.match(text or "")
    return int(match.group(1)) if match else None


def parse_user_text(text: str) -> dict[str, str]:
    """Inverse of :func:`serialize_row` for the ``Column: value`` lines."""
    out: dict[str, str] = {}
    for line in text.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            out[key.strip()] = value.strip()
    return out


def _line_re(field: str) -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(field)}: .*$", flags=re.MULTILINE)


def set_field(user_text: str, field: str, value: object) -> str:
    """Replace the value of one ``Column:`` line; raise if the line is absent."""
    pattern = _line_re(field)
    if not pattern.search(user_text):
        raise KeyError(f"field {field!r} not present in prompt")
    return pattern.sub(f"{field}: {_clean(value)}", user_text, count=1)


def drop_fields(user_text: str, fields: Iterable[str]) -> str:
    fields = set(fields)
    return "\n".join(line for line in user_text.split("\n") if line.split(":", 1)[0] not in fields)


def rename_field(user_text: str, old: str, new: str) -> str:
    pattern = _line_re(old)
    if not pattern.search(user_text):
        raise KeyError(f"field {old!r} not present in prompt")
    return pattern.sub(lambda m: new + m.group(0)[len(old) :], user_text, count=1)


def reorder_fields(user_text: str, order: Sequence[str]) -> str:
    """Re-emit the prompt lines in ``order`` (must be a permutation of the present fields)."""
    lines = {line.split(":", 1)[0]: line for line in user_text.split("\n") if ":" in line}
    if set(order) != set(lines):
        raise ValueError("order must be a permutation of the fields present in the prompt")
    return "\n".join(lines[field] for field in order)


def shuffled_order(seed: int, fields: Sequence[str] = FIELDS) -> list[str]:
    rng = random.Random(seed)
    order = list(fields)
    rng.shuffle(order)
    return order
