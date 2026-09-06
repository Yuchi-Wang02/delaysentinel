#!/usr/bin/env python
"""Fail if a documentation file quotes a number that no committed result file contains.

What this checks: every number in README.md, docs/case_study*.md, data/SPLIT.md,
data/DATASET_CARD.md, runs/RUNS.md and CHANGELOG.md (outside YAML front-matter, fenced
code blocks and URLs) must appear either as a numeric leaf of ``results/*.json`` or
``runs/sc904/tensorboard_events.json``, as one of a few named keys of
``runs/*/training_config.json``, as a summary value derived from ``runs/*/trainer_state.json``,
or in ``scripts/card_number_allowlist.txt`` with a stated reason.

What this does NOT check: that a number sits next to the right label. It is a presence
check that catches typos, stale numbers and unsourced figures, not a proof of correctness.
The tables were checked by hand against the JSON keys named in each section.

A number may be written with fewer decimals than the JSON value (rounded), with
thousands separators, as a percentage of a stored fraction (``22.7%`` for 0.227), or, for
values below 1e-3, in ``x.ye-06`` or ``x.y`` (times 1e-6) form.
"""

from __future__ import annotations

import contextlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = [
    ROOT / "README.md",
    ROOT / "docs" / "case_study.md",
    ROOT / "docs" / "case_study.zh.md",
    ROOT / "data" / "SPLIT.md",
    ROOT / "data" / "DATASET_CARD.md",
    ROOT / "runs" / "RUNS.md",
    ROOT / "CHANGELOG.md",
]
ALLOWLIST = ROOT / "scripts" / "card_number_allowlist.txt"
TRAINING_CONFIG_KEYS = (
    "num_train_epochs",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "weight_decay",
    "warmup_steps",
    "seed",
    "eval_steps",
    "save_steps",
    "save_total_limit",
    "logging_steps",
)
NUM_RE = re.compile(r"(?<![\w.\-/])-?\d[\d,]*(?:\.\d+)?(?![\w/])")


def strip_markdown(text: str) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4 :]
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\b[0-9a-f]{12,64}\b", " ", text)  # hex hashes
    return text


def normalise(token: str) -> set[str]:
    token = token.rstrip(",").replace(",", "")
    out = {token}
    try:
        value = float(token)
    except ValueError:
        return out
    if value.is_integer() and "." not in token:
        out.add(str(int(value)))
    return out


def _float_forms(value: float) -> set[str]:
    forms = {repr(value), str(value)}
    for nd in (1, 2, 3, 4):
        forms.add(f"{value:.{nd}f}")
    if value.is_integer():
        forms.add(str(int(value)))
    if 0 < abs(value) < 1e-3:
        forms.add(f"{value:.1e}")
        forms.add(f"{value:.2e}")
        forms.add(f"{value * 1e6:.1f}")
        forms.add(f"{value * 1e6:.2f}")
    return forms


def harvest_json(obj, acc: set[str]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, int):
        acc.add(str(obj))
    elif isinstance(obj, float):
        acc |= _float_forms(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k == "value" and isinstance(v, str):
                acc |= normalise(v)  # probe edit values such as "60" or "30.0"
            harvest_json(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            harvest_json(v, acc)


def trainer_state_summary(path: Path) -> dict:
    state = json.loads(path.read_text(encoding="utf-8"))
    evals = [(e["step"], e["eval_loss"]) for e in state["log_history"] if "eval_loss" in e]
    trains = [(e["step"], e["loss"], e.get("epoch")) for e in state["log_history"] if "loss" in e]
    grads = [(e["step"], e["grad_norm"]) for e in state["log_history"] if "grad_norm" in e]
    first_zero = next((s for s, lab, _ in trains if lab == 0.0), None)
    first_zero_epoch = next((ep for s, lab, ep in trains if lab == 0.0), None)
    return {
        "epoch": state["epoch"],
        "global_step": state["global_step"],
        "eval_first": evals[0] if evals else None,
        "eval_min": min(evals, key=lambda x: x[1]) if evals else None,
        "eval_last": evals[-1] if evals else None,
        "eval_points": len(evals),
        "train_first": trains[0][:2] if trains else None,
        "first_log_point_with_zero_loss": first_zero,
        "first_log_point_with_zero_loss_epoch": first_zero_epoch,
        "grad_norm_last": grads[-1] if grads else None,
        "steps_per_epoch": round(state["global_step"] / state["epoch"]) if state.get("epoch") else None,
    }


def known_numbers() -> set[str]:
    acc: set[str] = set()
    for path in sorted((ROOT / "results").glob("*.json")):
        harvest_json(json.loads(path.read_text(encoding="utf-8")), acc)
    for path in sorted((ROOT / "runs").glob("*/tensorboard_events.json")):
        harvest_json(json.loads(path.read_text(encoding="utf-8")), acc)
    for path in sorted((ROOT / "runs").glob("*/training_config.json")):
        cfg = json.loads(path.read_text(encoding="utf-8"))
        harvest_json({k: cfg[k] for k in TRAINING_CONFIG_KEYS if k in cfg}, acc)
    for path in sorted((ROOT / "runs").glob("*/trainer_state.json")):
        harvest_json(trainer_state_summary(path), acc)
    return acc


def allowlist() -> set[str]:
    acc: set[str] = set()
    if ALLOWLIST.exists():
        for line in ALLOWLIST.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                acc |= normalise(line)
    return acc


def check(docs: list[Path] | None = None, known: set[str] | None = None) -> list[tuple[str, str, str]]:
    """Return ``(doc, token, line)`` for every number that is not backed."""
    docs = docs or DOCS
    known = known if known is not None else (known_numbers() | allowlist())
    missing: list[tuple[str, str, str]] = []
    for doc in docs:
        if not doc.exists():
            continue
        text = strip_markdown(doc.read_text(encoding="utf-8"))
        for line in text.splitlines():
            for m in NUM_RE.finditer(line):
                token = m.group(0)
                forms = normalise(token)
                if line[m.end() : m.end() + 1] == "%":  # "22.7%" may be backed by 0.227 in the JSON
                    with contextlib.suppress(ValueError):
                        forms |= {f"{float(token.rstrip(',').replace(',', '')) / 100:.{nd}f}" for nd in (2, 3, 4)}
                if not (forms & known):
                    missing.append((doc.name, token, line.strip()[:110]))
    return missing


def main() -> int:
    known = known_numbers() | allowlist()
    missing = check(known=known)
    if missing:
        print("numbers not found in results/, runs/ or the allowlist:")
        for doc, token, line in dict.fromkeys(missing):
            print(f"  {doc:18s} {token!r:>12}  <- {line}")
        return 1
    print(f"check_card_numbers: all numbers in {len(DOCS)} documents are backed ({len(known)} known values).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
