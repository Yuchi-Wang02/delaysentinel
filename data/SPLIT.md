# Frozen historical split v0

| file | rows | positive rate | sha256 |
| --- | ---: | ---: | --- |
| `smart_logistics_dataset.csv` | 1,000 | 0.566 | `d9c91a6bd89d9eeb4e5656b1c83df62b15da69d0bbea45afa317379b1e518762` |
| `train.jsonl` | 800 | 0.5625 | `0c0132648ffa6e7bf5157afd29f126f0bd59b0d3aa4a8a51fb1f401d5d9aa6b8` |
| `test.jsonl` | 200 | 0.58 (116 / 84) | `a2f53527ce94208deded0f1022e9a8eb75d580434e5b5d323789172ff4277cfe` |

## How it was made (September 2025)

1. `legacy/Transform.py` turned every CSV row into a ShareGPT-style record: the system
   prompt, the 15 non-target columns as `Column: value` lines (CSV order, missing reason
   kept as the literal `None`), and the answer `Logistics_Delay: 0|1`.
2. `legacy/shuffle.py` shuffled the 1,000 records with `random.shuffle` **without a seed**
   and wrote the first 800 to `train.jsonl` and the last 200 to `test.jsonl`.

`tests/test_transform.py` verifies that `train.jsonl + test.jsonl` is exactly a
permutation of what step 1 produces from the CSV, so the frozen files are a faithful
image of the source data even though the shuffle itself cannot be replayed.

## Rules

- **Do not re-split.** The published weights were trained on these 800 rows; a new split
  would move training rows into the test set.
- The two-clause rule `Shipment_Status == "Delayed" OR Traffic_Status == "Heavy"`
  reproduces every label in both files (`tests/test_rule.py`).
- 0 test prompts also occur in `train.jsonl` (checked in `results/eval.json`).
- During training, `train.py` also received `test.jsonl` as the Trainer `eval_dataset`;
  the upstream `CustomSFTTrainer` evaluated a 10% subset every 200 steps, and because it
  wraps `SequentialSampler(Subset(...))` that subset is the first 20 rows of the file. The
  test set was therefore *observed* during training, although no hyper-parameter was
  selected on it (no accuracy was computed at the time). Do not call it held-out.
- `python -m delaysentinel.transform split data/all.jsonl --seed N` exists for future
  experiments only.
