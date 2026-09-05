import json

from delaysentinel.transform import csv_to_records


def test_frozen_split_is_exactly_the_csv(root):
    """train.jsonl + test.jsonl are a permutation of what Transform.py produces from the CSV."""
    records = csv_to_records(root / "data" / "smart_logistics_dataset.csv")
    assert len(records) == 1000
    regenerated = sorted(json.dumps(r, ensure_ascii=False) for r in records)
    frozen = []
    for name in ("train.jsonl", "test.jsonl"):
        with open(root / "data" / name, encoding="utf-8") as handle:
            frozen += [line.rstrip("\n") for line in handle if line.strip()]
    assert sorted(frozen) == regenerated
