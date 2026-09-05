#!/usr/bin/env python
"""Fail if README.md quotes a number that no committed result file contains.

Numbers are harvested from README.md outside the YAML front-matter, fenced code
blocks and URLs. Each must appear either in ``scripts/card_number_allowlist.txt``
(with a stated reason) or as a numeric leaf / number inside a string in
``results/*.json``, ``runs/*/training_config.json`` or the trainer-state summaries.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
ALLOWLIST = ROOT / "scripts" / "card_number_allowlist.txt"
NUM_RE = re.compile(r"(?<![\w.\-/])-?\d[\d,]*(?:\.\d+)?(?![\w/])")


def strip_markdown(text: str) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4 :]
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    return text


def normalise(token: str) -> set[str]:
    token = token.replace(",", "")
    out = {token}
    try:
        value = float(token)
    except ValueError:
        return out
    if value.is_integer() and "." not in token:
        out.add(str(int(value)))
    for nd in (1, 2, 3, 4):
        out.add(f"{value:.{nd}f}")
    out.add(repr(value))
    return out


def harvest_json(obj, acc: set[str]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        acc |= normalise(repr(obj) if isinstance(obj, float) else str(obj))
        if isinstance(obj, float):
            for nd in (1, 2, 3, 4):
                acc.add(f"{obj:.{nd}f}")
            acc.add(f"{obj * 100:.0f}")
            acc.add(f"{obj * 100:.1f}")
            if 0 < abs(obj) < 1e-3:
                acc.add(f"{obj:.1e}")
                acc.add(f"{obj * 1e6:.1f}")
                acc.add(f"{obj * 1e6:.2f}")
    elif isinstance(obj, str):
        for m in NUM_RE.finditer(obj):
            acc |= normalise(m.group(0))
    elif isinstance(obj, dict):
        for v in obj.values():
            harvest_json(v, acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            harvest_json(v, acc)


def trainer_state_summary(path: Path) -> dict:
    state = json.loads(path.read_text(encoding="utf-8"))
    evals = [(e["step"], e["eval_loss"]) for e in state["log_history"] if "eval_loss" in e]
    trains = [(e["step"], e["loss"]) for e in state["log_history"] if "loss" in e]
    first_zero = next((s for s, lab in trains if lab == 0.0), None)
    return {
        "epoch": state["epoch"],
        "global_step": state["global_step"],
        "max_steps": state.get("max_steps"),
        "eval_first": evals[0] if evals else None,
        "eval_min": min(evals, key=lambda x: x[1]) if evals else None,
        "eval_last": evals[-1] if evals else None,
        "train_first": trains[0] if trains else None,
        "first_step_with_zero_loss": first_zero,
    }


def known_numbers() -> set[str]:
    acc: set[str] = set()
    for path in sorted((ROOT / "results").glob("*.json")):
        harvest_json(json.loads(path.read_text(encoding="utf-8")), acc)
    for path in sorted((ROOT / "runs").glob("*/training_config.json")):
        harvest_json(json.loads(path.read_text(encoding="utf-8")), acc)
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


def main() -> int:
    text = strip_markdown(README.read_text(encoding="utf-8"))
    known = known_numbers() | allowlist()
    missing: list[tuple[str, str]] = []
    for line in text.splitlines():
        for m in NUM_RE.finditer(line):
            token = m.group(0)
            if not (normalise(token) & known):
                missing.append((token, line.strip()[:110]))
    if missing:
        print("README numbers not found in results/, runs/ or the allowlist:")
        seen = set()
        for token, line in missing:
            if (token, line) in seen:
                continue
            seen.add((token, line))
            print(f"  {token!r:>12}  <- {line}")
        return 1
    print(f"check_card_numbers: all README numbers are backed by committed files ({len(known)} known values).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
