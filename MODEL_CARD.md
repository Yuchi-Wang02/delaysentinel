---
license: llama3.2
language:
- en
library_name: transformers
pipeline_tag: text-generation
base_model: meta-llama/Llama-3.2-1B-Instruct
datasets:
- Yuchiwang02/smart-logistics-delay-split-v0
tags:
- llama
- llama-3
- sft
- tabular-classification
- logistics
- supply-chain
- label-leakage
- case-study
- evaluation
---

# Llama-3.2-1B-DelaySentinel

**Building and evaluating AI on logistics data. Built with Llama.**

This checkpoint is a full-parameter fine-tune of `meta-llama/Llama-3.2-1B-Instruct`,
published with the data split, evaluation results, and behavioral tests used to inspect it.
The project connects a working training-and-inference pipeline with a retrospective audit
of what its score demonstrates.

The checkpoint and a depth-2 decision tree both score 100% on the same historical 200-row
split. A rule using two supplied fields reproduces every label in the 1,000-row source table.
Prompt rewrites reveal additional response failures, including sensitivity to irrelevant text.
These findings make the checkpoint useful for studying evaluation and label leakage.

[GitHub project](https://github.com/Yuchi-Wang02/delaysentinel) ·
[Case study](docs/case_study.md) · [Full evaluation](docs/evaluation.md) ·
[Frozen dataset](https://huggingface.co/datasets/Yuchiwang02/smart-logistics-delay-split-v0) ·
[Reproduce](docs/reproduce.md)

![Llama and a depth-2 tree both score 100% on the same historical 200-row split; two input fields reproduce the full dataset label.](docs/figures/score_explained.svg)

The figure summarizes a historical, partly exposed split: its first 20 rows were used for
training-time eval loss. It is a baseline comparison for this artifact, not an independent
estimate of prospective forecasting performance.

## Model details

| item | recorded setup |
| --- | --- |
| base checkpoint | `meta-llama/Llama-3.2-1B-Instruct` |
| method | full-parameter supervised fine-tuning |
| trainable parameters | 1,235,814,400 |
| training data | 800 records from a 1,000-row Kaggle logistics table |
| historical evaluation file | 200 records; the first 20 were used for eval loss during training |
| input | system instruction plus 15 `Column: value` lines |
| output | `Logistics_Delay: 0` or `Logistics_Delay: 1` |
| weights trained | September 2025 |
| retrospective audit | September 2026 |
| weights license | Llama 3.2 Community License and Acceptable Use Policy |

The frozen split was created with an unseeded shuffle. It can be checked against recorded
hashes, but the original shuffle cannot be regenerated. Because the first 20 evaluation rows
were exposed during training-time eval loss, the historical file is not a clean independent
test set. See [split provenance](data/SPLIT.md).

## Evaluation summary

| predictor, same historical 200 rows | accuracy | F1 | accuracy interval, 95% Wilson |
| --- | ---: | ---: | --- |
| fine-tuned checkpoint, greedy decoding | 1.000 | 1.000 | [0.981, 1.000] |
| depth-2 decision tree | 1.000 | 1.000 | [0.981, 1.000] |
| all-positive baseline | 0.580 | 0.734 | [0.511, 0.646] |

Source: [`results/eval.json`](results/eval.json), `model.test_metrics_greedy` and
`baselines_split_v0`. The target satisfies
`Shipment_Status == "Delayed" OR Traffic_Status == "Heavy"` on all 1,000 source rows.
The table measures agreement with that dataset label; it does not establish future delay
prediction or an advantage over classical models.

A behavioral test appends "Note: the depot supervisor is Mr. Delayed" to the 84 originally
negative prompts without changing their logistics fields. Every output changes to 1.
The full audit records 104 related probe sets; they reuse source prompts and are not independent
experiments. The tests describe observed response patterns, while the internal mechanism
remains unknown. See [the complete probe tables](docs/evaluation.md#what-changes-the-models-answer).

The [separate Olist reference study](docs/olist_reference.md) uses classical models on real
orders. Its results are not an evaluation of this Llama checkpoint.

## Intended use and scope

Use this checkpoint to reproduce a label-leakage case study, inspect behavior under controlled
input rewrites, or teach baseline comparison and evaluation design. The repository provides
recorded outputs and executable checks for these purposes.

It has not been validated for operational delay forecasting, risk scoring, or dispatch
decisions. It provides hard labels rather than calibrated risk estimates. Tested invalid or
incomplete inputs can still receive a label instead of an abstention. Examples and probes are
specific to this checkpoint and prompt format; they do not establish general conclusions about
LLMs on tabular data.

## How to use

The checkpoint was trained on the 15-column schema below. The example reproduces an
original-format response; it is not an interface validated for real-world risk prediction. Decode greedily, pin the template date, and parse with an anchored
regex; treat a non-match as unparsable, never as 0.

```python
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = "Yuchiwang02/Llama-3.2-1B-DelaySentinel"
tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.bfloat16, device_map="auto")

SYSTEM = ("Assume you are a supply chain analyst. Based on the following information, "
          "output the result for Logistics_Delay, where 1 represents a delay and 0 represents no delay.")
# First row of data/test.jsonl (gold label 0). Same 15 columns, same order, as in training.
USER = """Timestamp: 2024-07-18 12:19:20
Asset_ID: Truck_10
Latitude: 20.2969
Longitude: 124.1885
Inventory_Level: 298
Shipment_Status: In Transit
Temperature: 18.6
Humidity: 74.4
Traffic_Status: Detour
Waiting_Time: 41
User_Transaction_Amount: 248
User_Purchase_Frequency: 1
Logistics_Delay_Reason: None
Asset_Utilization: 86.3
Demand_Forecast: 283"""

messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]
input_ids = tok.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", date_string="04 Sep 2025"
).to(model.device)
out = model.generate(
    input_ids,
    do_sample=False,
    max_new_tokens=8,
    pad_token_id=tok.convert_tokens_to_ids("<|finetune_right_pad_id|>"),  # never add a pad token / resize embeddings
)
text = tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
m = re.match(r"^\s*Logistics_Delay:\s*([01])\b", text)
label = int(m.group(1)) if m else None  # None = unparsable
print(text, label)                      # -> Logistics_Delay: 0  0
```

Or, with the package: `from delaysentinel.model import Scorer` and `Scorer(REPO).predict([USER])`.

## Training and data

The source is Kaggle's [Smart Logistics Supply Chain Dataset](https://www.kaggle.com/datasets/ziya07/smart-logistics-supply-chain-dataset)
by ziya07, listed as CC0 when recorded by the project. It contains 1,000 timestamped records,
15 supplied fields, and the target. A deterministic target rule and geographically dispersed
coordinates are consistent with a constructed table, but the repository does not contain a
generator or independently verified collection provenance. Promised and actual delivery
dates are absent.

The original run used 30 epochs, 1,500 steps, bf16, batch size 4 with gradient accumulation 4,
learning rate 2e-5, cosine scheduling, and assistant-only loss. Training configuration seed
42 does not recover the earlier unseeded data shuffle. The logged training loss rounds to
0.0000 at step 50; this is a rounded log value, not an assertion of mathematically zero loss.

Training used an adaptation of `acon96/home-llm` with three local changes and an attribution
header. Recovered settings and preserved artifacts are described in [run provenance](runs/RUNS.md)
and the [technical report](docs/evaluation.md#training-procedure).
The script is available in the [GitHub source](https://github.com/Yuchi-Wang02/delaysentinel/blob/main/scripts/train.py).

## Limitations retained for reproducibility

- The source label is reconstructible from supplied inputs and lacks a documented prospective
  delivery outcome. The historical split was partly exposed during training-time evaluation.
- Behavioral probes reveal sensitivity to wording, negation, and irrelevant values; the internal
  mechanism and the full set of possible triggers are not established.
- The historical collator masked genuine system/user end-of-turn tokens along with padding.
  This training artifact is documented; no retrained comparison establishes its causal impact.
- No checkpoint before step 1,300 survives, preventing analysis of when the final behavior arose.
- No separate validation period or calibration study establishes a business decision threshold
  for this checkpoint. Updated inference examples do not change the historical training process.
- Recorded bf16 margins can vary in their final digits across batch composition and environments.

The [evaluation report](docs/evaluation.md) contains the complete methods, tables, probe
conditions, and evidence boundaries. The [reproduction guide](docs/reproduce.md) separates
CPU checks, full model execution, and the independent Olist study.

## Attribution and licenses

**Built with Llama.** The weights are subject to the [Llama 3.2 Community License](LICENSE)
and [Acceptable Use Policy](USE_POLICY.md), with attribution in [NOTICE](NOTICE).
The full model name is `Llama-3.2-1B-DelaySentinel`; the original Hub name redirects to this
repository.

The training script derives from [acon96/home-llm](https://github.com/acon96/home-llm)
under MIT, with Stanford Alpaca portions under Apache-2.0. The preserved copy has three local
changes relative to the identified upstream revision; see [third-party attribution](THIRD_PARTY_LICENSES.md)
and the [Apache license](LICENSES/Apache-2.0.txt).

Project code and documentation are distributed under [MIT](LICENSE-MIT). The Olist aggregate
result is distributed under CC BY-NC-SA 4.0. Implementation and documentation use AI assistance;
linked source records and the attribution file describe the project materials and reused work.

## Citation

```bibtex
@misc{wang2026delaysentinel,
  title  = {Llama-3.2-1B-DelaySentinel: Building and evaluating AI on logistics data},
  author = {Wang, Yuchi},
  year   = {2026},
  note   = {Weights trained September 2025; retrospective evaluation and behavioral audit added September 2026},
  url    = {https://huggingface.co/Yuchiwang02/Llama-3.2-1B-DelaySentinel}
}
```

[中文案例](docs/case_study.zh.md) ·
[Related project: BizHallu](https://github.com/Yuchi-Wang02/bizhallu)
