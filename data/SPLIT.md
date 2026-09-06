# Frozen historical split v0

| file | rows | positive rate | sha256 (newline-normalized bytes) |
| --- | ---: | ---: | --- |
| `smart_logistics_dataset.csv` | 1,000 | 0.566 | `e5d080a1d64f15713f000be403a44d4ad329cbece11a0523a8c186406a6ed7a6` |
| `train.jsonl` | 800 | 0.5625 | `3f313ae9ae8163a576b33771460f66b1b2ffcf2ac46fb125ed0e0263c200cfd7` |
| `test.jsonl` | 200 | 0.58 (116 / 84) | `8aa62f8820cc6479c2e65d6ecc47fba539ce94cc3c4c47483b56a96d26ce72e0` |

The hashes are computed over the file bytes after mapping CRLF to LF, which equals the raw
hash of the committed (LF) files. The files were written on Windows in September 2025 with
CRLF line endings; the CRLF originals hash to `d9c91a6b…`, `0c013264…` and `a2f53527…`
(recorded here for provenance only). `tests/test_split_hashes.py` checks this table against
the files on every platform, and `results/eval.json` → `provenance` records the same three
values.

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
- During training, `train.py` also received `test.jsonl` as the Trainer `eval_dataset`.
  The upstream `CustomSFTTrainer` evaluated a 10% subset every 200 steps and, because it
  wraps `SequentialSampler(Subset(...))`, that subset is the first 20 rows of the file. Those
  20 rows were therefore used for eval loss during training (forward pass only: no
  gradient, no metric, no hyper-parameter chosen on them); the other 180 were never
  touched. Treat the file as unblinded rather than as a clean test set.
- `python -m delaysentinel.transform split data/all.jsonl --seed N` exists for future
  experiments only.

## Source licence

The Kaggle page (https://www.kaggle.com/datasets/ziya07/smart-logistics-supply-chain-dataset)
listed the licence as **CC0: Public Domain** when it was read on 2026-09-03 (dataset "updated
2 years ago", one version, one 106 KB file). That is the uploader's declaration; nothing else
about provenance is documented there. The derived JSONL files are dedicated to the public
domain under CC0-1.0 as well (`THIRD_PARTY_LICENSES.md`).
