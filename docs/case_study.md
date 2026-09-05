# How a 1.24B-parameter model learned a two-line rule

*A label-leakage post-mortem on my first fine-tuning project. Every number here is in
[`results/eval.json`](../results/eval.json); the figures are in [`figures/`](figures/).*

## 1. What I built (September 2025)

I was an accounting and supply-chain undergraduate who wanted to learn how supervised fine-tuning
works end to end. I took a 1,000-row Kaggle table called "Smart Logistics Supply Chain Dataset",
turned each row into a chat record (15 `Column: value` lines, answer `Logistics_Delay: 0|1`), split
it 800/200 with an unseeded shuffle, and ran full-parameter SFT of `Llama-3.2-1B-Instruct` for 30
epochs on a consumer GPU using a training script adapted from an open-source smart-home project.
I exported a GGUF, wrote a Flask form, and uploaded the weights to Hugging Face with a model card
that called it "AI-powered logistics delay prediction".

The pipeline worked. Training loss hit exactly 0.0 at step 50, the end of the first epoch
(`figures/fig_training.png`). I did not compute accuracy; the only evaluation I looked at was the
Trainer's eval loss, which sat around 3e-6.

## 2. The number that should have worried me

A year later I scored the published weights on my own 200-row test split with greedy decoding:
accuracy 1.000, F1 1.000, confusion `[[84, 0], [0, 116]]`, zero unparsable outputs.

A perfect score on a noisy business problem is not a result. It is a symptom. The first thing to
check is whether the label can be reconstructed from the inputs by something much simpler than the
model.

## 3. Finding the rule

A cross-tabulation of the label against the two status-like columns answers the question in one
screen (`figures/fig_crosstab.png`):

| condition | rows | labelled delayed |
| --- | ---: | ---: |
| `Shipment_Status == "Delayed"` | 350 | 350 |
| `Traffic_Status == "Heavy"` | 327 | 327 |
| neither | 434 | 0 |

`Logistics_Delay = 1 iff Shipment_Status == "Delayed" OR Traffic_Status == "Heavy"` holds on all
1,000 rows with zero exceptions. The first clause is the target restated under another column
name. The second is a road condition that can only be observed while the shipment is already
moving. Both were written verbatim into every training prompt.

`python -m delaysentinel.leakage_audit` now finds this automatically: it scans every column for
single conditions whose rows are pure in the target, greedily ORs the pure-positive ones together,
and reports that two conditions reproduce the label with zero mismatches and that a depth-2
decision tree scores 1.000 under 5-fold cross-validation. It runs in a second. Running it before
training would have ended the project as a *prediction* project on day one.

The table is also synthetic, whatever the Kaggle page says: coordinates are uniform over the
whole globe (`figures/fig_latlon.png`), every numeric column is uniform between round bounds, and
318 rows that are *not* delayed still carry a "delay reason".

## 4. Baselines

On the same 200 rows the rule itself, a depth-2 decision tree, logistic regression and gradient
boosting all score 1.000. All-positive scores 0.580 (F1 0.734). Gradient boosting trained *without*
the two rule columns scores 0.500 with AUROC 0.452: there is nothing else in the table to learn.
Seeded repeated 5-fold cross-validation over all 1,000 rows says the same for every baseline
(rule and tree 1.000 on every fold; boosting without the rule columns 0.496).

So the fine-tuned model is not "as good as a decision tree". It is *indistinguishable from* a
decision tree with three leaves, at roughly a billion times the parameter count and about 18
minutes of GPU time instead of none.

## 5. Probing what the weights actually do

A score tells you *that* the model matches the labels; it does not tell you *how*. Counterfactual
probes edit the text of the test prompts one field at a time and re-score
(`figures/fig_probes.png`):

- Flip `Shipment_Status` from `Delayed` to `In Transit` on the 50 positives whose traffic is not
  `Heavy`: 50 of 50 predictions flip to 0.
- Flip `Traffic_Status` from `Heavy` to `Clear` on the 43 positives whose status is not `Delayed`:
  43 of 43 flip to 0.
- Set `Traffic_Status` to `Heavy`, or `Shipment_Status` to `Delayed`, on the 84 negatives: 84 of 84
  flip to 1 in each case.
- Set `Waiting_Time` to 60 and `Temperature` to 30 on the negatives, or to 10 and 18 on the
  positives: 0 of 200 predictions change.
- Delete both rule lines from every prompt: the model answers 0 for all 200 rows.

Robustness probes then ask how the rule is represented. Renaming the two columns, renaming only
one, and shuffling the order of the 15 lines leave every prediction unchanged, so this is not a
lexical lookup on column names. Writing `Delayed` as `Late`, `DELAYED` or `delayed` still yields
1 on all 73 rows, and `HEAVY` or `heavy` still yields 1 on all 66. But `Congested` is not read as
`Heavy`: only the 23 rows that are also `Delayed` stay positive. Deleting only one rule line makes
the model evaluate the other clause exactly (66 positives when only traffic remains, 73 when only
status remains, 1.000 agreement with the remaining clause both times). Unseen values
(`Unknown`, `N/A`) yield 0 everywhere.

Outside the schema the model does not abstain. The usage example from my original model card,
which used columns like `carrier` and `weight_kg` that the model never saw, gets a confident
`Logistics_Delay: 0`. So does an empty prompt and a header-only prompt. Asked "What is the capital
of France?", it answers `Paris`. The base model is still in there; the fine-tune added a narrow,
literal `OR` over two string equalities and a strong prior to answer `0` when the form is present
but the trigger values are not.

## 6. What was wrong with the release

The weights were the least of it.

- **License.** The base model is Llama 3.2. Its community license requires distributors to ship the
  agreement, display "Built with Llama", and start the model name with "Llama". My repo said
  `apache-2.0`, had no license file, and was called DelaySentinel.
- **Attribution.** The training script was a copy of `acon96/home-llm`'s `train.py` (MIT) with no
  credit. Upstream has since deleted the file, so the repo now pins the commit it matched.
- **The card.** It promised a Gradio Space that did not exist, linked a placeholder repository,
  showed a usage snippet with the wrong repo id, an invented prompt format and a foreign schema,
  and described the task as prediction "before shipment" with "order-level features". It contained
  no metric at all.
- **The configs.** `generation_config.json` shipped with sampling on (the base model's defaults);
  `config.json` shipped with `use_cache: false` left over from training. Sampling happened not to
  change any of the 200 predictions across three seeds, but a classifier should not sample.
- **The app.** The Flask UI loaded the model from a hard-coded local path, added a new pad token and
  resized the embeddings at inference time, generated up to 512 tokens for a one-digit answer,
  parsed the result with `includes('1')`, and asked the user to enter the *delay reason* before
  predicting the delay.
- **The evaluation.** The test file doubled as the Trainer eval set; because the upstream trainer
  wraps `SequentialSampler(Subset(...))`, every eval-loss point was computed on the first 20 rows of
  that file. With no pad token defined, the collator also used `<|eot_id|>` as the pad value and
  masked the real end-of-turn tokens of the system and user turns. Neither mattered for this task;
  both would matter for a real one.

All of this is fixed in the v1.0.0 release (`CHANGELOG.md`); none of it required touching the
weights.

## 7. What I would do differently, in order

1. **Read the license before the README.** Base-model obligations and data terms first.
2. **Audit the label before the model.** Cross-tabulate the target against every categorical
   column; run the leakage scanner; fit a depth-2 tree. If anything scores near 1.000, stop.
3. **Decide what each field is known at.** For a delay model: checkout, approval, carrier handoff
   or delivery. Anything observed at or after the outcome is not a feature.
4. **Freeze the split and the protocol before touching the test set.** Seeded split, a separate
   validation set, prevalence and trivial baselines written down, metrics chosen in advance.
5. **Report intervals and baselines next to every number.** Wilson or Clopper-Pearson for
   accuracy; bootstrap when there is something to resample; the all-positive rate on the same line.
6. **Probe, do not just score.** Counterfactual edits are cheap and tell you what the model uses.
7. **Attribute and license before publishing.** A header on borrowed code, the right license tag,
   the required notices.

## 8. Why this leads to BizHallu

The lesson is not "LLMs are bad at tables". It is that a model can be *right for the wrong
reason* and that a confident answer is not evidence of understanding. That is the question my
current project, [BizHallu](https://github.com/Yuchi-Wang02/bizhallu), studies at the level of
individual business-fact spans in LLM-generated retail analysis: is each claim grounded in the
transaction evidence, and can that be checked? The habits I now use there, frozen splits with
hashes, trivial baselines and bootstrap intervals on the same line as every metric, explicit
limitations, and cards that only quote numbers a committed JSON contains, started as the fixes
listed above.

## 9. Reproduce

```bash
pip install -e ".[model,figures,dev]"
python -m pytest -q
python -m delaysentinel.eval --model Yuchiwang02/Llama-3.2-1B-DelaySentinel --out results/eval.json
python -m delaysentinel.eda --out docs/figures
python scripts/check_card_numbers.py
```
