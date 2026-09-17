# Llama-3.2-1B-DelaySentinel

### Building and evaluating AI on logistics data

[![tests](https://github.com/Yuchi-Wang02/delaysentinel/actions/workflows/test.yml/badge.svg)](https://github.com/Yuchi-Wang02/delaysentinel/actions/workflows/test.yml)

**A published Llama fine-tune. A reproducible evaluation workflow. Concrete tests of what changes a prediction.**

DelaySentinel connects model development with three practical questions: **What drives a strong score? Which inputs change the answer? What would a realistic evaluation require?**

A depth-2 decision tree matched the checkpoint's 100% accuracy on the same historical 200-row split. A label audit traced the dataset's outcome to two input fields. Controlled prompt rewrites then exposed sensitivity to irrelevant text. A separate study on real Olist orders examines classical models using checkout-time information and a later test period.

[**Explore the case study →**](docs/case_study.md) · [**Inspect the model →**](https://huggingface.co/Yuchiwang02/Llama-3.2-1B-DelaySentinel) · [**Reproduce the checks →**](docs/reproduce.md) · [中文案例](docs/case_study.zh.md)

## A perfect score, explained

![Llama and a depth-2 decision tree both score 100% on the historical 200-row split. Two input conditions reproduce all 1,000 dataset labels.](docs/figures/score_explained.svg)

The comparison makes the score interpretable. In the source table, every label follows this rule:

```text
Logistics_Delay = 1 when Shipment_Status is Delayed OR Traffic_Status is Heavy
```

Both fields were included in the model's input. A depth-2 tree, logistic regression and gradient boosting all matched the checkpoint on the original split. Removing the two rule fields brought the tested boosting baseline to 0.500 accuracy and 0.452 AUROC.

**What this establishes:** the dataset's labels are exactly recoverable from its inputs, and the full score does not establish advance delay forecasting. This is a retrospective evaluation: the historical split includes rows used for training-time evaluation loss. [Split details](data/SPLIT.md) · [full baseline table](docs/evaluation.md)

The reusable output is a [scanner](docs/leakage_audit.md) that searches a table for pure single-column conditions and a compact OR-rule, making this check available before fitting a larger model.

<a id="what-the-model-actually-learned"></a>

## Stress-testing the answer

![Across 84 originally negative records, adding the note 'the depot supervisor is Mr. Delayed' changes every prediction from 0 to 1 while the logistics fields remain unchanged.](docs/figures/behavior_test.svg)

The model correctly matches the original labels. The next question is whether its response remains useful when the wording changes.

One test appends **“Note: the depot supervisor is Mr. Delayed”** to the same 84 negative records. Every prediction changes from 0 to 1. The corresponding **“Mr. Heavy”** note changes none. This contrast exposes a specific sensitivity to irrelevant text in the tested prompts.

Other tests sharpen the picture:

- **Relevant field changes:** 261 of 261 targeted rule-field edits flip the answer as the dataset rule predicts.
- **Negation:** replacing `Delayed` with `Not Delayed` leaves all 73 predictions at 1. Of these, 23 still satisfy the separate `Heavy` condition; the remaining 50 isolate the negation issue.
- **Synonyms:** replacing `Delayed` with `Postponed` produces 50 zeros and 23 ones—the latter still have `Heavy` traffic.
- **Other fields:** 3,200 non-rule rewrite attempts change no predictions; 197 of those attempts leave the input text unchanged.

These are observed responses from a shared historical prompt set. They characterize tested behavior; they do not identify the model's internal algorithm. The [technical report](docs/evaluation.md) documents all 104 probe sets, their conditions and their dependencies.

## Built, tested, documented

| Part of the project | What you can inspect |
| --- | --- |
| **Model development** | Full-parameter supervised fine-tuning of Llama-3.2-1B-Instruct; published weights, training records and a shared inference interface. [Model card](MODEL_CARD.md) · [training record](runs/RUNS.md) |
| **Evaluation and baselines** | Saved predictions, simple classifiers, feature ablations and a reusable label-leakage scanner. [Results](results/eval.json) · [scanner](docs/leakage_audit.md) |
| **Behavior testing** | Field edits, synonyms, negation, control words and out-of-schema prompts, with predictions and answer-token margins. [Evaluation report](docs/evaluation.md) |
| **Business framing** | A separate real-order study with prediction-time features, temporal separation and sensitivity analysis. [Olist study](docs/olist_reference.md) |

## A separate study on real orders

![Olist study design: use checkout-time information, train on orders delivered before the cutoff, and test on orders purchased afterward.](docs/figures/olist_timeline.svg)

The Olist analysis applies the evaluation questions to real, anonymized e-commerce orders: define late delivery from dates, use information available at checkout, and separate training from a later test period. **It evaluates classical models on another dataset; it is not a test of the Llama checkpoint.**

Logistic regression reaches **0.6908 AUROC** and **0.155 AUPRC**, with a test late-delivery rate of **0.0722**. The study also examines undelivered orders, probability calibration and variation across observed months. Those checks make the business setting and remaining validation work visible alongside the metrics. [Read the design, results and sensitivity analysis →](docs/olist_reference.md)

## Explore or reproduce

Start with the saved evidence. CPU checks cover the data and baselines; a compatible GPU is recommended for the full model evaluation.

| Goal | Entry point |
| --- | --- |
| Read the project story | [English case study](docs/case_study.md) · [中文案例](docs/case_study.zh.md) |
| Check a finding | [Evaluation report](docs/evaluation.md) · [per-row predictions](results/test_predictions.csv) |
| Run the CPU checks or evaluate the checkpoint | [Reproduction guide](docs/reproduce.md) |
| Inspect model inputs and training | [Model card](MODEL_CARD.md) · [frozen dataset](https://huggingface.co/datasets/Yuchiwang02/smart-logistics-delay-split-v0) |
| Try the interface locally | [Local demonstration](space/README.md) |

## Scope and project development

The project began with a fine-tuning and publishing workflow in September 2025. The September 2026 evaluation adds baselines, data checks, behavioral probes and inspectable results. The checkpoint remains available as an evaluation case study; operational forecasting would require a suitable prospective target, prediction-time inputs and independent validation.

This portfolio was developed with AI assistance for coding, analysis and documentation. Methods and recorded outputs are linked throughout; adapted training code and local changes are documented in [third-party attribution](THIRD_PARTY_LICENSES.md).

The same interest in checking business AI outputs continues in [BizHallu](https://github.com/Yuchi-Wang02/bizhallu), which examines whether claims in generated retail analysis are supported by transaction evidence.

**Built with Llama.** Weights: [Llama 3.2 Community License](LICENSE) and [Acceptable Use Policy](USE_POLICY.md). Project code: [MIT](LICENSE-MIT). Data provenance and terms: [dataset card](data/DATASET_CARD.md).
