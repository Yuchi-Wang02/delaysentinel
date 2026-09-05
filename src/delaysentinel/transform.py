"""CSV -> ShareGPT-style JSONL, reproducing the Sept 2025 ``Transform.py`` output exactly.

    python -m delaysentinel.transform csv-to-jsonl data/smart_logistics_dataset.csv -o data/all.jsonl
    python -m delaysentinel.transform split data/all.jsonl --seed 0 --train-ratio 0.8

The historical split shipped in ``data/train.jsonl`` / ``data/test.jsonl`` was produced
by the legacy script *without* a seed; ``split`` here is seeded and exists only for
future experiments. Do not re-split the frozen files.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from .prompting import SYSTEM_PROMPT, TARGET, answer_text


def csv_to_records(csv_path: str | Path) -> list[dict]:
    """Legacy behaviour: every column except the last becomes a ``Column: value`` line."""
    records = []
    with open(csv_path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        last = reader.fieldnames[-1]
        if last != TARGET:
            raise ValueError(f"last column is {last!r}, expected {TARGET!r}")
        user_cols = reader.fieldnames[:-1]
        for row in reader:
            user = "\n".join(f"{c}: {'' if row.get(c) is None else str(row.get(c)).strip()}" for c in user_cols)
            assistant = answer_text(int(str(row[last]).strip()))
            records.append(
                {
                    "conversations": [
                        {"from": "system", "value": SYSTEM_PROMPT},
                        {"from": "user", "value": user},
                        {"from": "assistant", "value": assistant},
                    ]
                }
            )
    return records


def write_jsonl(records: list[dict], out_path: str | Path) -> None:
    with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def split_jsonl(path: str | Path, seed: int, train_ratio: float = 0.8) -> tuple[Path, Path]:
    path = Path(path)
    with open(path, encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip()]
    rng = random.Random(seed)
    rng.shuffle(lines)
    cut = int(len(lines) * train_ratio)
    train_path = path.with_name(f"train_seed{seed}.jsonl")
    test_path = path.with_name(f"test_seed{seed}.jsonl")
    train_path.write_text("".join(lines[:cut]), encoding="utf-8")
    test_path.write_text("".join(lines[cut:]), encoding="utf-8")
    return train_path, test_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("csv-to-jsonl")
    a.add_argument("csv")
    a.add_argument("-o", "--out", default=None)
    b = sub.add_parser("split")
    b.add_argument("jsonl")
    b.add_argument("--seed", type=int, required=True)
    b.add_argument("--train-ratio", type=float, default=0.8)
    args = parser.parse_args(argv)
    if args.command == "csv-to-jsonl":
        out = args.out or str(Path(args.csv).with_name("all.jsonl"))
        records = csv_to_records(args.csv)
        write_jsonl(records, out)
        print(f"wrote {len(records)} records -> {out}")
    else:
        tr, te = split_jsonl(args.jsonl, args.seed, args.train_ratio)
        print(f"wrote {tr} and {te}")


if __name__ == "__main__":
    main()
