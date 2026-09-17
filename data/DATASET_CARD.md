---
license: cc0-1.0
language:
- en
pretty_name: Smart Logistics delay split v0 (frozen)
size_categories:
- 1K<n<10K
task_categories:
- text-classification
tags:
- logistics
- supply-chain
- data-quality
- label-leakage
- tabular
---

# Smart Logistics delay split v0 (frozen)

The 1,000-row Kaggle **Smart Logistics Supply Chain Dataset** (user *ziya07*, listed as CC0 on
https://www.kaggle.com/datasets/ziya07/smart-logistics-supply-chain-dataset when read on
2026-09-03) together with the exact 800/200 ShareGPT-style JSONL split on which
[`Yuchiwang02/Llama-3.2-1B-DelaySentinel`](https://huggingface.co/Yuchiwang02/Llama-3.2-1B-DelaySentinel)
was fine-tuned in September 2025.

## What this dataset makes possible

The frozen files support a reproducible study of label leakage, baseline comparison and model
behavior. `Logistics_Delay` equals `Shipment_Status == "Delayed" OR Traffic_Status == "Heavy"`
on all 1,000 rows. A depth-2 decision tree matches the published checkpoint's 100% accuracy on
the historical 200-row evaluation split. That result establishes agreement with this dataset's
label rule; forecasting delivery outcomes requires a different target and information available
before the outcome.

## Source and interpretation

The source does not document how the records or labels were generated. The broad geographic
spread of coordinates, regular-looking numeric distributions and a delay reason on 318
label-negative rows are signals consistent with synthetic construction. They do not confirm a
generation process, and no formal test of uniformity is included. Treat the table as an
instructional dataset whose operational provenance is unverified.

Rows combine truck status, inventory and customer attributes. There are no promised and actual
delivery dates, so the status flag cannot establish on-time delivery. In particular, a
`Delivered` status alone does not say whether a delivery was early, on time or late.

| file | rows | note |
| --- | ---: | --- |
| `smart_logistics_dataset.csv` | 1,000 | the Kaggle file, unchanged (missing reason is the literal `None`) |
| `train.jsonl` | 800 | system / user / assistant records; unseeded shuffle, September 2025 |
| `test.jsonl` | 200 | 116 positive / 84 negative; its first 20 rows were the Trainer eval subset during training |
| `SPLIT.md` | | hashes, how the split was made, why it must not be re-split |

The JSONL files, including the system prompt they contain, are the author's mechanical
transformation of the CSV and are dedicated to the public domain under CC0-1.0 as well.

The first 20 evaluation rows contributed to eval loss during training. The split is preserved
for checkpoint analysis; it is not a fresh, untouched test set. [SPLIT.md](SPLIT.md) records the
hashes and the historical procedure.

Code, saved results and model-behavior tests:
[DelaySentinel on GitHub](https://github.com/Yuchi-Wang02/delaysentinel).
