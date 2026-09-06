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
- synthetic
- label-leakage
- tabular
---

# Smart Logistics delay split v0 (frozen)

The 1,000-row Kaggle **Smart Logistics Supply Chain Dataset** (user *ziya07*, listed as CC0 on
https://www.kaggle.com/datasets/ziya07/smart-logistics-supply-chain-dataset when read on
2026-09-03) together with the exact 800/200 ShareGPT-style JSONL split on which
[`Yuchiwang02/Llama-3.2-1B-DelaySentinel`](https://huggingface.co/Yuchiwang02/Llama-3.2-1B-DelaySentinel)
was fine-tuned in September 2025.

**Read this before using the label.** `Logistics_Delay` equals
`Shipment_Status == "Delayed" OR Traffic_Status == "Heavy"` on all 1,000 rows. Any model that
sees those two columns can score 100%; a depth-2 decision tree does. The table is synthetic
(coordinates uniform over the globe, uniform numeric columns, a "delay reason" on 318
non-delayed rows). It is useful as a teaching example of target leakage, not as logistics data.

| file | rows | note |
| --- | ---: | --- |
| `smart_logistics_dataset.csv` | 1,000 | the Kaggle file, unchanged (missing reason is the literal `None`) |
| `train.jsonl` | 800 | system / user / assistant records; unseeded shuffle, September 2025 |
| `test.jsonl` | 200 | 116 positive / 84 negative; its first 20 rows were the Trainer eval subset during training |
| `SPLIT.md` | | hashes, how the split was made, why it must not be re-split |

The JSONL files, including the system prompt they contain, are the author's mechanical
transformation of the CSV and are dedicated to the public domain under CC0-1.0 as well.

Code, evaluation JSON and probes: https://github.com/Yuchi-Wang02/delaysentinel
