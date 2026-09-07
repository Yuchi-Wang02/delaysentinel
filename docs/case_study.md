# How a 1.24-billion-parameter model learned to answer from the surface of the prompt

*A label-leakage post-mortem on my first fine-tuning project. Every metric here is in
[`results/eval.json`](../results/eval.json) or
[`results/olist_positive_control.json`](../results/olist_positive_control.json); training-curve
numbers come from `runs/*/trainer_state.json`; the wall-clock figure comes from TensorBoard event
timestamps recorded in `runs/sc904/tensorboard_events.json`. The figures are in
[`figures/`](figures/). 中文版见 `case_study.zh.md`.*

## 1. What I built (September 2025)

I was an undergraduate double-majoring in accounting and supply-chain management who wanted to
learn how supervised fine-tuning works end to end. I took a 1,000-row Kaggle table called "Smart
Logistics Supply Chain Dataset", turned each row into a chat record (15 `Column: value` lines,
answer `Logistics_Delay: 0|1`), split it 800/200 with an unseeded shuffle, and ran full-parameter
SFT of `Llama-3.2-1B-Instruct` for 30 epochs on a single GPU (model not logged) using a training
script taken from an open-source smart-home project. I exported a GGUF, wrote a Flask form, and
uploaded the weights to Hugging Face with a model card that called it "AI-powered logistics delay
prediction".

The pipeline worked. The logged training loss rounded to 0.0000 by step 50, the end of the first
epoch (`figures/fig_training.png`). I never computed accuracy; the only evaluation I looked at
was the Trainer's eval loss, which sat around 3e-6.

## 2. The number that should have worried me

A year later I scored the published weights for the first time, on my own 200-row test split with
greedy decoding: accuracy 1.000, F1 1.000, confusion `[[84, 0], [0, 116]]`, zero unparsable
outputs.

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
| both | 111 | 111 |
| neither | 434 | 0 |

`Logistics_Delay = 1 iff Shipment_Status == "Delayed" OR Traffic_Status == "Heavy"` holds on all
1,000 rows with zero exceptions. The first clause is the target restated under another column
name. The second is a road condition of the same snapshot: 118 of 118 rows that are already
*Delivered* but have *Heavy* traffic are labelled delayed, which no notion of an on-time outcome
would produce. The label is a Boolean over two columns. Both were written verbatim into every
training prompt.

`python -m delaysentinel.leakage_audit` now finds this automatically: it scans every column for
single conditions whose rows are pure in the target, greedily ORs the pure-positive ones together,
and reports that two conditions reproduce the label with zero mismatches and that a depth-2
decision tree scores 1.000 under 5-fold cross-validation. Running it before training would have
ended the project as a *prediction* project on day one.

The table is also synthetic, whatever the Kaggle page says: latitude and longitude are uniform
over the whole globe (`figures/fig_latlon.png`), and 318 rows that are *not* delayed still carry
a "delay reason".

## 4. Baselines

On the same 200 rows the rule itself, a decision tree with two splits, logistic regression and
gradient boosting all score 1.000. All-positive scores 0.580 (F1 0.734). Gradient boosting
trained *without* the two rule columns scores 0.500 with AUROC 0.452; also dropping the post-hoc
`Logistics_Delay_Reason` gives 0.475 / 0.460; adding month, weekday and hour parsed from the
timestamp gives 0.520 / 0.482. There is nothing else in the table to learn. Seeded repeated
5-fold cross-validation over all 1,000 rows says the same for every baseline (rule and tree 1.000
on every fold; the three "without" variants between 0.496 and 0.509 mean accuracy).

So the fine-tuned model is not "as good as a decision tree". On every original and counterfactual
test prompt it ties with a tree that has two splits and three leaves, at 1,235,814,400 parameters
and about 18 minutes between the first and last TensorBoard events of the run, against a
millisecond of CPU for the tree. The two part ways only on the rewrites in the next section.

## 5. Probing what the weights actually do

A score tells you *that* the model matches the labels; it does not tell you *how*. Counterfactual
probes edit the text of the test prompts one field at a time and re-score, recording the model's
logit margin on each (`figures/fig_probes.png`):

- Flip `Shipment_Status` from `Delayed` to `In Transit` on the 50 positives whose traffic is not
  `Heavy`: 50 of 50 predictions flip to 0.
- Flip `Traffic_Status` from `Heavy` to `Clear` on the 43 positives whose status is not `Delayed`:
  43 of 43 flip to 0.
- Set `Traffic_Status` to `Heavy`, or `Shipment_Status` to `Delayed`, on the 84 negatives: 84 of 84
  flip to 1 in each case. In total 261 of 261 rule-field edits, over 177 distinct rows, flip as the
  rule predicts.
- Change one of the other 13 fields (15 rewrites per row, because `Logistics_Delay_Reason` is
  probed with three values, so 3,000 single-field edits) or two of them together (200 more):
  0 predictions change. Every margin stays above 12 in absolute value. 197 of those 3,200 edits
  left the prompt unchanged because the row already carried the replacement value.
- Delete both rule lines from every prompt: the model answers 0 for all 200 rows.

So far this is consistent with "the model learned the rule". The robustness probes then ask what
exactly is being matched, and the answer is not the rule.

**Case, truncation and negation do not matter.** `DELAYED`, `delayed`, the bare stem `Delay` and
`Not Delayed` all keep 73 of 73 rows at 1. `HEAVY`, `heavy`, the truncation `Heav` and `Not Heavy`
keep 66 of 66 at 1. A model that had learned the rule would answer 0 for `Not Delayed`.

**Meaning mostly does not matter, and where it does it points the wrong way.** `Behind schedule`,
`Postponed`, `Overdue`, `Held up`, `On Time` and `Pending` do *not* fire, leaving only the 23 rows
that satisfy the other clause; the same for `Congested`, `Jammed`, `Gridlock`, `Slow`, `Dense`,
`Free-flowing` and `Moderate`. But `Late`, `Early` and `Light` all fire on every row. Four of the
five synonyms for delayed and all five for heavy traffic are read as "not delayed", while the
fifth, `Late`, and the antonyms `Early` and `Light` are read as "delayed".

**The field does not matter.** Put `Heavy` or `Delayed` into `Logistics_Delay_Reason`, put
`Delayed` into `Asset_ID`, or swap the two values between the rule fields, and all 84 negatives
answer 1. Renaming the columns and shuffling the 15 lines changes nothing at all.

**But it is a specific match, not a reaction to anything unfamiliar.** Twelve unrelated
single-token words (`Copper`, `Violet`, `Harbor`, `Maple`, `Quartz`, `Falcon`, `Meadow`, `Cobalt`,
`Lantern`, `Marble`, `Willow`, `Amber`) placed in each rule field leave the prediction at the
baseline in 23 of 24 probes. The single exception, `Meadow` in `Traffic_Status`, fires on 10 rows
beyond the baseline.

**The two clauses are not implemented alike.** Appending a free-text line "Note: the depot
supervisor is Mr. Delayed" to the 84 negatives makes all 84 answer 1. The same line with
"Mr. Heavy" changes nothing. The `Delay` match scans the whole turn; the `Heavy` match only reads
field values.

Outside the schema the model does not abstain. The usage example from my original model card,
which used columns like `carrier` and `weight_kg` that the model never saw, gets
`Logistics_Delay: 0` with a margin of -14.0. So does a header-only prompt (-10.6) and an empty
prompt (-2.1, the weakest of the four). Asked "What is the capital of France?", it answers `Paris`.
The base model is still in there; the fine-tune added a surface-form trigger and a strong prior
to answer `0` whenever the form is present and nothing in it fires.

What is still open: I do not know what the pattern is, and the two obvious guesses are both
dead. It is not a match on the strings `Delay` and `Heavy` — `Late`, `Early` and `Light` contain
neither and fire on every row. It is not a substring scan of the prompt — the field name
`Logistics_Delay_Reason` carries "Delay" in all 200 prompts and triggers nothing. Why that
particular set of forms fires and `Postponed` or `Congested` does not is untested. Those are the
probes I would design next, and they should have been part of the protocol before training rather
than a year after.

## 6. What the same metrics look like on real data

To check that the method is not the problem, `python -m delaysentinel.positive_control` applies
the same discipline to the public Olist Brazilian e-commerce orders (99,441 real, anonymised
orders from 2016-2018): late if the customer delivery date is after the estimated date, delivered
orders only, a 60-day right-censoring guard, features restricted to what is known at checkout, and
a split that a deployed model could actually have used — train on orders *delivered* before
2018-03-01 (53,644 orders, late rate 0.0505), test on orders *purchased* on or after it (37,702
orders, late rate 0.0722). 3,673 orders that straddle the split belong to neither.

Logistic regression reaches AUROC 0.6908 with an order-level bootstrap interval of
[0.6819, 0.6993] and a month-block interval of [0.6329, 0.7599]; AUPRC 0.155 against a prevalence
of 0.0722; histogram gradient boosting 0.6507 / 0.1176; the promised lead time alone ranks at
0.5568 (`figures/fig_positive_control.png`). The month-block interval is about seven times as wide
as the order-level one, and that is the honest uncertainty for "would this hold next period".

Two things are wrong with the setup, and the JSON says so rather than hiding them. First, 2,919
orders purchased before the cut-off were still undelivered at extraction, every one already past
its promised date; the delivered-only filter drops them instead of labelling them late. Counting
the 555 of them that fall in the test window as late takes that set from 37,702 orders with 2,722
late to 38,257 with 3,277, which raises the prevalence from 0.0722 to 0.0857 and moves AUROC to
0.6812. Second, the model is mis-calibrated across the split: mean predicted 0.0413 against an
observed 0.0722. That is not a base-rate drift the model could have anticipated — the late rate in
the test window swings from 0.0116 in June 2018 to 0.1896 in March 2018, and the model predicts
0.0493 for that March. Dropping `purchase_month` does not fix it (0.0565 for the same month). No
re-calibration was attempted.

The same leakage scanner run on this table finds no pure-positive condition at all: the greedy
OR-rule is empty and a depth-2 tree reaches 0.9278, the majority-class rate. That is the negative
control the scanner needed.

## 7. What was wrong with the release

The weights were the least of it.

- **License.** The base model is Llama 3.2. Its community licence requires distributors to ship
  the agreement, display "Built with Llama", and start the model name with "Llama". My repo said
  `apache-2.0`, had no licence file, and was called DelaySentinel.
- **Attribution.** The training script was a copy of `acon96/home-llm`'s `train.py` (MIT, with
  Stanford Alpaca portions under Apache-2.0) with no credit. It now sits in `scripts/` unchanged
  and unformatted, with a header naming the upstream revision it matches and the three small edits
  I made to it.
- **The card.** It promised a Gradio Space that did not exist, linked a placeholder repository,
  showed a usage snippet with the wrong repo id, an invented prompt format and a foreign schema,
  and described the task as prediction "before shipment" with "order-level features". It contained
  no metric at all.
- **The configs.** `generation_config.json` shipped with sampling on (the base model's defaults);
  `config.json` shipped with `use_cache: false` left over from training. Sampling happened not to
  change any of the 200 predictions across three seeds, which follows from margins above 12, but a
  classifier should not sample.
- **The app.** The Flask UI loaded the model from a hard-coded local path, added a new pad token
  and resized the embeddings at inference time, generated up to 512 tokens for a one-digit answer,
  parsed the result with `includes('1')`, and asked the user to enter the *delay reason* before
  predicting the delay.
- **The evaluation.** The test file doubled as the Trainer eval set; because the upstream trainer
  wraps `SequentialSampler(Subset(...))`, every eval-loss point was computed on the first 20 rows
  of that file. With no pad token defined, the collator also used `<|eot_id|>` as the pad value and
  masked the real end-of-turn tokens of the system and user turns. Neither changed anything
  measurable on this task; on a harder task both are untested, and the eval-subset quirk would
  silently shrink any validation set to 10% of its rows.

All of this is fixed in the v1.0.0 release (`CHANGELOG.md`); none of it required touching the
weights. The Hub side is pushed by `scripts/publish_hf.py` once I run it.

## 8. What I would do differently, in order

1. **Read the licence before the README.** Base-model obligations and data terms first.
2. **Audit the label before the model.** Cross-tabulate the target against every categorical
   column; run the leakage scanner; fit a depth-2 tree. If anything scores near 1.000, stop.
3. **Decide what each field is known at.** For a delay model: checkout, approval, carrier handoff
   or delivery. Anything observed at or after the outcome is not a feature, and a field that
   *defines* the label is not a feature at any time.
4. **Freeze the split and the protocol before touching the test set.** A seeded split, a
   validation period separate from the test period, prevalence and trivial baselines written down,
   metrics chosen in advance.
5. **Report intervals and baselines next to every number,** and pick the interval that matches the
   question: order-level bootstrap for "how precise is this estimate", month-block for "would it
   hold next period".
6. **Probe, do not just score.** Counterfactual edits are cheap and tell you what the model uses.
   Design the synonym, negation and control-word probes before training; a probe suite without
   control words cannot tell a specific trigger from a reaction to anything unfamiliar.
7. **Attribute and license before publishing.** A header on borrowed code, the right licence tag,
   the required notices, and a check that stops the card from quoting numbers no file contains.

## 9. Why this leads to BizHallu

The lesson is not "LLMs are bad at tables". It is that a model can be *right for the wrong reason*
and that a confident answer is not evidence of understanding: a 1.24-billion-parameter model
answers `1` to `Shipment_Status: Not Delayed` with the same margin it gives the real thing. That is
the question my current project, [BizHallu](https://github.com/Yuchi-Wang02/bizhallu), studies at
the level of individual business-fact spans in LLM-generated retail analysis: is each claim
grounded in the transaction evidence, and can that be checked? The habits I use there — frozen
splits with hashes, trivial baselines and bootstrap intervals on the same line as every metric,
explicit limitations, and cards that only quote numbers a committed JSON contains — started as the
fixes listed above.

## 10. Reproduce

```bash
pip install -r requirements-lock.txt && pip install -e .
python -m pytest -q
python -m delaysentinel.eval --model Yuchiwang02/Llama-3.2-1B-DelaySentinel --out results/eval.json
python -m delaysentinel.positive_control --out results/olist_positive_control.json
python -m delaysentinel.eda --out docs/figures
python scripts/check_card_numbers.py
```
