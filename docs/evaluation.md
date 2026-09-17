# DelaySentinel: evaluation and behavioral evidence

This report documents the published checkpoint's historical evaluation, baseline comparisons,
and controlled prompt rewrites. The key findings are an exactly reconstructible dataset label
and observable sensitivity to wording and irrelevant text. The tables retain the recorded
results so each interpretation can be checked.

[Project overview](https://github.com/Yuchi-Wang02/delaysentinel) ·
[Case study](case_study.md) · [Saved evaluation](../results/eval.json) ·
[Per-row predictions](../results/test_predictions.csv)

The checkpoint was trained in September 2025; this retrospective evaluation was added in
September 2026. Original model weights, split files, result files, and research figures are
preserved. The [Olist study](olist_reference.md) uses separate data and classical models.

## The label is a rule over two of its own input columns

Verified on the full CSV (`results/eval.json` → `label_rule`):

| condition | rows | labelled delayed |
| --- | ---: | ---: |
| `Shipment_Status == "Delayed"` | 350 | 350 |
| `Traffic_Status == "Heavy"` | 327 | 327 |
| both | 111 | 111 |
| neither | 434 | 0 |

`Logistics_Delay = 1 iff Shipment_Status == "Delayed" OR Traffic_Status == "Heavy"` holds with 0
mismatches on this CSV. The target is therefore fully reconstructible from supplied input
fields. The dataset does not document the event time or business semantics needed to interpret
that flag as future delivery risk.

All 118 rows with `Shipment_Status == "Delivered"` and `Traffic_Status == "Heavy"` have label 1.
A delivered shipment can still have arrived late; this observation alone is not a contradiction.
The substantive finding is that the label follows the two-field rule exactly, while promised
and actual delivery dates are absent. This table cannot establish an on-time delivery outcome.

`python -m delaysentinel.leakage_audit` recovers the rule automatically
([`results/leakage_audit.json`](../results/leakage_audit.json)).


![Delay rate by Shipment_Status and Traffic_Status](figures/fig_crosstab.png)


## Evaluation

**Split.** 200 test rows (116 positive / 84 negative, positive rate 0.58) from an **unseeded**
`random.shuffle` 80/20 split made once in September 2025 (train 800 rows, positive rate 0.5625).
0 test prompts occur in train. The split cannot be regenerated; the files are frozen with their
hashes in [`data/SPLIT.md`](../data/SPLIT.md), checked by `tests/test_split_hashes.py`. The test file
was also passed to the Trainer as `eval_dataset` during training: the upstream trainer evaluates
a 10% subset every 200 steps and, because it wraps `SequentialSampler(Subset(...))`, that subset is
always the first 20 rows. Those 20 rows were used for eval loss (forward pass only: no gradient,
no accuracy metric was recorded); the remaining 180 rows were not used by that evaluation path.
The surviving records do not establish whether those losses influenced checkpoint or epoch selection.
The 200 rows form a historical, partly exposed evaluation split, rather than a clean independent test set.

**Decoding.** Greedy, `max_new_tokens` 8 (the answer is six tokens plus `<|eot_id|>`), the
Llama-3 chat template with its `Today Date` line pinned to `04 Sep 2025` (the template would
otherwise insert the current date; the pinned date and the sha256 of the rendered system turn are
in the JSON), the training system prompt, pad token `<|finetune_right_pad_id|>`, answers parsed
with an anchored regex and reported as unparsable on a non-match (never scored as 0). Run on
2026-09-06 on an RTX 5070 Ti against the weights downloaded from the Hub. `results/eval.json` →
`provenance` records the Hub LFS sha256 of `model.safetensors` (`ffc509c6…`), the git commit (with
a `-dirty` suffix when the working tree differs from it) and the library versions
(`scikit-learn` 1.7.2 matters for the boosting rows below). 200 rows in 3.11 s; the whole run,
104 probe sets included, takes 266.4 s.

| predictor (same 200 rows) | acc | 95% Wilson | precision | recall | F1 | confusion `[[TN,FP],[FN,TP]]` | AUROC |
| --- | ---: | --- | ---: | ---: | ---: | --- | ---: |
| **fine-tuned model, greedy** | 1.000 | [0.981, 1.000] | 1.000 | 1.000 | 1.000 | `[[84, 0], [0, 116]]` | saturated¹ |
| two-clause rule | 1.000 | [0.981, 1.000] | 1.000 | 1.000 | 1.000 | `[[84, 0], [0, 116]]` | n/a |
| decision tree, depth 2 | 1.000 | [0.981, 1.000] | 1.000 | 1.000 | 1.000 | `[[84, 0], [0, 116]]` | 1.000 |
| logistic regression | 1.000 | [0.981, 1.000] | 1.000 | 1.000 | 1.000 | `[[84, 0], [0, 116]]` | 1.000 |
| gradient boosting | 1.000 | [0.981, 1.000] | 1.000 | 1.000 | 1.000 | `[[84, 0], [0, 116]]` | 1.000 |
| all-positive | 0.580 | [0.511, 0.646] | 0.580 | 1.000 | 0.734 | `[[0, 84], [0, 116]]` | n/a |
| gradient boosting **without** the two rule columns² | 0.500 | [0.431, 0.569] | 0.556 | 0.690 | 0.615 | `[[20, 64], [36, 80]]` | 0.452 |
| … also without the post-hoc `Logistics_Delay_Reason` | 0.475 | [0.407, 0.544] | 0.539 | 0.655 | 0.591 | `[[19, 65], [40, 76]]` | 0.460 |
| … plus month / weekday / hour parsed from `Timestamp` | 0.520 | [0.451, 0.588] | 0.573 | 0.681 | 0.622 | `[[25, 59], [37, 79]]` | 0.482 |

¹ The model emits a hard label. Its teacher-forced margin on the 200 prompts runs from -20.7 to
+14.0, with median absolute value 13.5 and minimum absolute value 12.25, so the implied
probabilities are 0.000 or 1.000. The AUROC of 1.000 restates the hard labels: it adds nothing
beyond accuracy. These saturated margins do not establish calibrated probabilities or operational
uncertainty estimates.
Bootstrap intervals are degenerate at zero errors and are not reported for the perfect rows; the
all-positive F1 interval is [0.680, 0.784] and the AUROC intervals of the three "without" rows
are [0.371, 0.528], [0.381, 0.541] and [0.400, 0.560].

² Feature encodings are recorded in `results/eval.json` → `baselines_split_v0.feature_encoding`.
Seeded repeated stratified 5-fold cross-validation over all 1,000 rows (3 repeats) gives the same
picture for every sklearn baseline: rule, tree, logistic and boosting 1.000 on every fold;
all-positive 0.566; the three "without" variants 0.496, 0.503 and 0.509 mean accuracy with mean
AUROC 0.463, 0.462 and 0.475. The fine-tuned model cannot be cross-validated because it has seen
the 800 training rows.

**Sampling check.** The repository originally shipped `generation_config.json` with the base
model's sampling defaults (`do_sample`, temperature 0.6, top-p 0.9). With that config and seeds
0, 1, 2 the test accuracy is 1.000 each time and 0 rows change between seeds. With margins of at
least 12.25 this follows from the greedy result and is not independent evidence; the config is now
greedy.


<a id="what-the-model-actually-learned"></a>

## What changes the model's answer

All probes rewrite the text of the 200 frozen test prompts and re-score with greedy decoding;
each also records the teacher-forced margin. 104 probe sets in total (`results/eval.json` →
`probe_set_counts`). These sets reuse the same historical prompts and include related rewrites;
they are not 104 independent experiments or independent samples.

### Counterfactual field edits

| edit | n | model prediction | margin (min abs.) |
| --- | ---: | --- | ---: |
| `Shipment_Status` Delayed → In Transit, on positives whose traffic is not Heavy | 50 | 50 flip to 0 | 17.6 |
| `Traffic_Status` Heavy → Clear, on positives whose status is not Delayed | 43 | 43 flip to 0 | 17.9 |
| `Traffic_Status` → Heavy, on all negatives | 84 | 84 flip to 1 | 12.25 |
| `Shipment_Status` → Delayed, on all negatives | 84 | 84 flip to 1 | 12.5 |
| each of the 13 non-rule fields assigned a replacement value, 15 rewrites per row because `Logistics_Delay_Reason` takes three | 3,000 | 0 change | 12.1 |
| `Waiting_Time` and `Temperature` changed together, on negatives and on positives | 200 | 0 change | 12.5 |
| both rule lines deleted from every prompt | 200 | 200 predict 0 (accuracy vs gold 0.42) | 15.5 |

261 of 261 rule-field edits, over 177 distinct rows, flip as the rule predicts. (The 23 positives
that are both Delayed and Heavy cannot be flipped by changing only one clause; they are
excluded from this single-clause flip count.) Of the
3,200 non-rule rewrite attempts, 197 leave the prompt unchanged because the row already carried
that value. None of the attempts changes a prediction. The count therefore includes no-op assignments;
it is not a count of distinct altered prompts.

![Counterfactual probe results](figures/fig_probes.png)

### Wording, field placement, and control words

| rewrite | n | model prediction |
| --- | ---: | --- |
| rename `Shipment_Status` → `Status` and/or `Traffic_Status` → `Traffic` | 200 | unchanged: 1.000 vs gold |
| shuffle the order of the 15 lines (seeds 0, 1, 2) | 200 | unchanged: 1.000 vs gold |
| `Delayed` written as `DELAYED`, `delayed`, the bare stem `Delay`, or `Not Delayed` | 73 | all 73 predict 1 |
| `Delayed` written as `Late` or `Early` | 73 | all 73 predict 1 |
| `Delayed` written as `Behind schedule`, `Postponed`, `Overdue`, `Held up`, `On Time` or `Pending` | 73 | only the 23 rows whose traffic is Heavy predict 1 |
| `Heavy` written as `HEAVY`, `heavy`, the truncation `Heav`, `Not Heavy`, or `Light` | 66 | all 66 predict 1 |
| `Heavy` written as `Congested`, `Jammed`, `Gridlock`, `Slow`, `Dense`, `Free-flowing` or `Moderate` | 66 | only the 23 rows whose status is Delayed predict 1 |
| 12 unrelated single-token words (`Copper`, `Violet`, `Harbor`, …) in `Shipment_Status` | 73 each | all 12 give exactly the 23 baseline rows; none fires |
| the same 12 words in `Traffic_Status` | 66 each | 11 of 12 give the 23 baseline rows; `Meadow` fires on 10 extra rows (33 of 66) |
| `Heavy` or `Delayed` placed in `Logistics_Delay_Reason`, `Delayed` in `Asset_ID`, or the two values swapped between the rule fields, on negatives | 84 | all 84 predict 1 |
| an appended free-text line "Note: the depot supervisor is Mr. **Delayed**", on negatives | 84 | all 84 predict 1 |
| the same line with "Mr. **Heavy**", on negatives | 84 | 0 change (all stay 0) |
| delete only the `Shipment_Status` line | 200 | 66 predict 1: exactly the rows with traffic Heavy (1.000 agreement with the remaining clause) |
| delete only the `Traffic_Status` line | 200 | 73 predict 1: exactly the rows with status Delayed (1.000 agreement with the remaining clause) |
| `Shipment_Status` = Unknown and `Traffic_Status` = N/A on every row | 200 | 200 predict 0 |

### Prompts outside the training schema

| prompt | model output | margin |
| --- | --- | ---: |
| the usage example from the September 2025 card (`order_id`, `carrier`, `weight_kg`, …) | `Logistics_Delay: 0` | -14.0 |
| the 15 column names with no values | `Logistics_Delay: 0` | -10.6 |
| an empty user turn | `Logistics_Delay: 0` | -2.1 |
| "What is the capital of France? Answer in one word." | `Paris` | -4.25 |

### Interpreting the observations

The negation and synonym probes require conditioning on the other clause of the label rule:

- Of the 73 rows whose shipment status was `Delayed`, 50 had traffic other than `Heavy`, and
  23 had `Heavy` traffic. Replacing `Delayed` with `Not Delayed` leaves all 73 predictions at 1.
  The 50 rows without `Heavy` traffic provide the negation-failure example; the other 23 retain
  a positive traffic condition.
- Replacing `Delayed` with `Postponed` leaves exactly the 23 `Heavy`-traffic rows at 1 and moves
  the other 50 to 0. Thus it is incorrect to say that all 73 rows become negative.
- Similarly, the 66 `Heavy`-traffic rows include 43 without `Delayed` shipment status and
  23 with it. The other status condition remains relevant when interpreting traffic rewrites.

Four of the five synonyms tested for delayed status fail to preserve the positive prediction on
the status-only rows; `Late` does preserve it. All five tested synonyms for heavy traffic fail
to preserve it on the traffic-only rows. The antonyms `Early` and `Light` also yield positive
predictions throughout their respective tested sets. These are observations about these prompts,
not a claim about every possible synonym or context.

Moving `Delayed` into `Asset_ID` or appending the sentence about "Mr. Delayed" changes all
84 original negative predictions to 1. The corresponding free-text sentence with "Mr. Heavy"
changes none. The contrast demonstrates sensitivity to an irrelevant note and different responses
to the two tested values. It does not identify a network-internal scanning mechanism.

The control-word results narrow the observation: all 12 control words in `Shipment_Status`
give exactly the 23 baseline rows; 11 of 12 traffic-field controls do the same. `Meadow` in
`Traffic_Status` is the exception, adding 10 positives. These controls argue against treating
every unfamiliar value as equivalent, within the tested set.

The internal mechanism remains unknown. A simple substring explanation is not supported:
`Late`, `Early`, and `Light` produce positive predictions without containing the original
trigger forms, while the word fragment in `Logistics_Delay_Reason` appears in every original
prompt without making every prediction positive. Behavioral tests characterize where a response
changes; they do not recover the model's full decision process.

The training loss rounds to 0.0000 from step 50, at the end of the first epoch. No checkpoint
before step 1,300 survives, so the audit cannot determine when the final response patterns emerged.
All model probes above describe the published final weights.


## Feature credibility and prediction time

The CSV contains timestamped records with truck identifiers, status fields, environmental
measurements, and customer attributes. It does not document when each feature became available
relative to an outcome. Field names alone do not establish a valid prediction time.

| field(s) | observed relationship in this table | question for a future prediction task |
| --- | --- | --- |
| `Shipment_Status` | `Delayed` implies label 1 on 350 of 350 rows; this field participates in the exact label rule | Is the status known before the decision, or does it already encode the outcome? |
| `Traffic_Status` | `Heavy` implies label 1 on 327 of 327 rows, including 118 rows marked `Delivered` | At what location and time was traffic observed? |
| `Logistics_Delay_Reason` | populated on 419 delayed and 318 non-delayed rows | Is a reason recorded after an event, and what do reasons on negative rows mean? |
| remaining input fields | the tested boosting baseline without the two rule fields has AUROC 0.452 | Is there prospective signal under a documented collection process? |
| `Timestamp` | adding month, weekday, and hour gives AUROC 0.482 in the reported variant | Which event does the timestamp represent? |
| `Asset_ID`, `Latitude`, `Longitude` | truck identifiers and geographically dispersed coordinates | What operational entity and route does each row describe? |

The tested baselines find little useful discrimination after removing the rule fields. This
is an empirical result for those models and splits, not proof that no statistical relationship
could ever be found. A future delivery-risk task would need promised and actual dates, a
defined unit such as an order, and features available at the intended decision time.


## Training data

- Source: Kaggle "Smart Logistics Supply Chain Dataset" by ziya07, listed as CC0 Public Domain
  when read on 2026-09-03,
  <https://www.kaggle.com/datasets/ziya07/smart-logistics-supply-chain-dataset>. One CSV (105,482
  bytes as committed; the Kaggle page shows 106 KB), 1,000 rows × 16 columns (15 inputs +
  `Logistics_Delay`); 566 delayed / 434 not delayed. Mirrored with the frozen split as
  [`Yuchiwang02/smart-logistics-delay-split-v0`](https://huggingface.co/datasets/Yuchiwang02/smart-logistics-delay-split-v0)
  (published dataset mirror).
- Categorical values: `Shipment_Status` Delayed 350 / Delivered 338 / In Transit 312;
  `Traffic_Status` Detour 345 / Clear 328 / Heavy 327; `Logistics_Delay_Reason` Weather 267 /
  None 263 / Traffic 236 / Mechanical Failure 234.
- The data have features consistent with a constructed table: coordinates spread over the globe,
  visually regular numeric ranges, and a deterministic target rule. The repository does not
  contain a generator or verified provenance proving how the table was produced. No formal
  uniformity test is committed.
- There are 10 `Asset_ID` trucks; 318 records with label 0 carry a delay reason. The 263 reasons
  represented by the literal string `None` are preserved in prompts.
- The apparent grain is a timestamped truck snapshot, mixing operational and customer fields.
  There is no documented order-level key or promised-versus-actual delivery outcome. These limits
  prevent treating the label as a validated prospective delivery-risk target.
- Serialisation (`legacy/Transform.py`): the system prompt, the 15 non-target
  columns as `Column: value` lines in CSV order, and `Logistics_Delay: 0|1` as the answer.
- Split (`legacy/shuffle.py`): `random.shuffle` without a seed, first 800 train,
  last 200 test. `tests/test_transform.py` checks that the frozen files are exactly a permutation
  of the CSV.

![Geographic distribution of truck coordinates in the source table](figures/fig_latlon.png)


## Training procedure

Recovered from `training_args.bin` (run `sc904`, whose `model.safetensors` hash equals the Hub
LFS hash; [`runs/RUNS.md`](../runs/RUNS.md)):

- Full-parameter SFT of all 1,235,814,400 parameters (no LoRA), assistant-only loss masking, bf16,
  gradient checkpointing, `group_by_length`, seed 42, `adamw_torch_fused`.
- 30 epochs = 1,500 steps; per-device batch 4 × gradient accumulation 4 = effective batch 16
  (50 steps per epoch); learning rate 2e-5, cosine schedule, no warmup, weight decay 0.1; eval
  every 200 steps on the 20-row subset described above; checkpoint every 100 steps; logging
  every 5 steps.
- The logged training loss (5-step mean, rounded to 4 decimals) reads 0.0000 from the step-50 log
  point, the end of the first epoch; the gradient norm is still 0.0006 at step 1,500, so the true
  loss is small, not zero. Eval loss on the 20-row subset goes 5.23 → 3.00 → 3.05 (×1e-6) at steps
  200, 1000 and 1400. Two further runs (65 and 100 epochs) reach the same logged 0.0000 by log
  points 70 and 45.
- Wall-clock: about 1,090 s (18 min) between the creation of the first and the last TensorBoard
  event file of the published run (`runs/sc904/tensorboard_events.json`), which excludes model
  loading; the GPU model was not logged.
- Scheduler quirk inherited from the upstream script: `learning_rate_overshoot = 1.15` plans the
  cosine schedule for 1,725 steps and training stops at 1,500, so the learning rate never reaches
  zero (8.3e-07 at the last step).
- Training script: `scripts/train.py`, adapted from `acon96/home-llm` with three local changes
  and an attribution header. See [the attribution record](../THIRD_PARTY_LICENSES.md).

![Training and eval loss](figures/fig_training.png)


## Scope and remaining limitations

The evaluation supports a reproducible account of this checkpoint, dataset, and historical
split. It does not establish deployment performance, a competitive LLM-versus-tree result on
real prediction tasks, or the model's internal mechanism.

- The target is a deterministic rule over supplied inputs, and the historical split was
  partly used for training-time evaluation. There is no separate validation set.
- Responses are hard labels with saturated margins. Calibration, selective abstention, and
  operational decision thresholds have not been validated for this checkpoint.
- The model emits 0 for the tested empty, header-only, and foreign-schema prompts. This is a
  demonstrated lack of abstention on those probes, not evidence that every invalid input behaves alike.
- The historical collator used `<|eot_id|>` as its padding value when no pad token was defined.
  Its attention mask therefore also hid actual system/user end-of-turn tokens. The assistant
  end-of-turn token remained in the loss. This training artifact is documented and retained;
  no retrained comparison establishes its causal effect.
- Current inference examples use greedy decoding, a pinned template date, and explicit output
  parsing. Updating documentation or inference configuration does not repair the historical
  split or alter the original training process.
- Training used 30 full-parameter epochs without early stopping or a sample-efficiency study;
  no checkpoint before step 1,300 survives.
- Teacher-forced margins were recorded in bf16 at batch size 16; final digits may vary with
  batch composition or execution environment.
- The numeric documentation check tests presence against recorded results and a reasoned
  allowlist. It can catch unsupported or mistyped numbers, but does not prove that a number
  belongs to the adjacent model, population, condition, or claim.

Implementation and documentation use AI assistance. The upstream training script and its
local changes are identified in the attribution record.


## Reading the metrics

- **Accuracy, precision, recall, and F1:** agreement with the recorded labels on the stated set.
- **Wilson interval:** a binomial interval around an observed proportion. It does not remove
  leakage, split exposure, or dependence between observations.
- **AUROC:** ranking discrimination across positive and negative examples.
- **AUPRC / average precision:** precision-recall performance, interpreted alongside the positive rate.
- **Brier score:** mean squared error of a probability prediction.
- **Margin:** the model's logit for token `1` minus its logit for `0`, measured after
  `Logistics_Delay:` under teacher forcing. A large margin is not a calibration guarantee.
- **Greedy decoding:** choose the most likely next token at each generation step.
- **Bootstrap interval:** an interval obtained by resampling the specified unit. The chosen
  resampling scheme determines which variation the interval represents.
- **Full-parameter SFT:** supervised fine-tuning that updates all model parameters.
- **bf16:** the numeric precision used in the recorded model run.

For the separate real-order study, see [Olist reference study](olist_reference.md).
For runnable commands, see [Reproduce the checks](reproduce.md).
