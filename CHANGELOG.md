# Changelog

## v1.0.2 — September 2026: cross-platform tolerance for the boosting baselines

The first CI run on GitHub (Linux) failed the baseline-reproduction step: the
`gradient_boosting_without_rule_fields_and_reason` row scored one of the 200 test rows differently
from the committed Windows numbers. Gradient boosting is not bit-identical across operating
systems even with the pinned `scikit-learn`. CI now requires exact reproduction for the rule, the
depth-2 tree, logistic regression and all-positive, and agreement to within one row for the three
gradient-boosting variants; the README says so. Nothing else changed: the weights, the frozen
split, `results/eval.json` and every documented number are as in v1.0.1.

## v1.0.1 — September 2026: corrections from the internal review

A read-only multi-agent review of the v1.0.0 tree found claims that the evidence did not
support and one attribution error. Nothing here changes the weights or the frozen split.

### The finding is now stated precisely, and one attempt at precision was itself wrong

v1.0.0 described the model as a bag-of-words detector for two trigger words. The first attempt to
sharpen that said it matches the strings `Delay` and `Heavy` — which the repository's own probes
refute, because `Late`, `Early` and `Light` contain neither string and fire on every row. That
wording was published in this changelog, the card and both case studies before it was caught, and
it is corrected here.

What the probes support: the answer is decided by the surface form of the value in the two rule
fields, case-blind, truncation-tolerant (`Heav` alone fires), negation-blind (`Not Delayed`
fires) and field-agnostic; the delayed trigger fires from an appended free-text line while the
heavy-traffic one does not; twelve unrelated control words in the same slots leave the prediction
at baseline in 23 of 24 probe sets.

What they do not support: any statement of what the pattern is. It is not the two strings, and it
is not a substring scan of the prompt either — `Logistics_Delay_Reason` carries `Delay` in every
prompt and never fires. The card and both case studies now say that the pattern is unidentified
and name the two guesses their own data rules out. `tests/test_probe_prose.py` fails if a document
starts claiming a string match again.

### Corrections to the Olist reference study

- Orders never delivered were being dropped silently. They are now counted, split into
  still in flight and never a delivery, and reported.
- The split was random. It is now a label-availability split: training uses only orders
  whose outcome was known at the split date, and the orders straddling it are reported
  rather than assigned.
- Intervals over months are now month-block bootstrap, not order-level resampling, because
  monthly late rates move together.
- The calibration gap had the wrong explanation. It is a shift in the monthly late rate,
  shown with a per-month table and an ablation that drops the purchase month.
- The threshold table is a sweep over score thresholds, not over calibrated probabilities,
  and the expedite-cost formula is corrected.

### Corrections to the evaluation code

- An unparsable answer is excluded from the teacher-forced AUROC instead of being scored as
  a negative.
- `position_ids` are derived from the attention mask, so left padding no longer shifts them.
- The chat template's date is pinned, and the rendered system turn is hashed into the
  results.
- A results file written from a dirty working tree records the commit with a `-dirty` mark.

### Corrections to the number check

- It now covers eight documents, reads scientific notation, and no longer accepts a coarse
  decimal rounding for values below 1e-3, where every such value would otherwise spell as
  `0.000` and match every other. Two tokens that had been passing on that collision, the
  unit of the eval-loss column and the rounding boundary of the training log, are now
  listed in the allowlist with their reasons.

### Corrections from the pre-publication audit

A second read-only pass over the finished tree, with every finding put to three independent
refuters, found twelve published statements the committed evidence does not support. All are
fixed here; none of them touches the weights, the frozen split or any result value.

- **Licence.** `LICENSE-MIT` granted MIT over all of `results/`, including
  `results/olist_positive_control.json`, which `THIRD_PARTY_LICENSES.md` correctly places under
  CC BY-NC-SA 4.0 as a derivative of a ShareAlike dataset. The operative licence file now carries
  the same carve-out as the document that explains it, and a test keeps the two in agreement.
- **Attribution.** The entry above described `scripts/train.py` as byte-for-byte upstream plus a
  header. It is upstream plus a header *and the three changes* that `THIRD_PARTY_LICENSES.md`
  lists. Calling a modified copy unmodified is the same class of error as not attributing it.
- **The control-word cell** claimed 11 of 12 words leave `Shipment_Status` at the baseline and
  that none fires, which contradicts itself. All twelve leave it at the baseline; the 11-of-12
  belongs to `Traffic_Status`, where `Meadow` is the exception.
- **The synonym reading.** `Late` is tagged a synonym in `results/eval.json` and fires on every
  row, so "five genuine synonyms per clause leave the answer at the baseline" was wrong in both
  case studies and the card. Four of the five for delayed, and all five for heavy traffic.
- **Numbers.** The minimum absolute margin on the two-field edit row is 12.5, not 12.4. The
  month-block interval is six to seven times the order-level width, not four. The in-flight
  sensitivity run adds the 555 orders that fall in the test window, not all 1,723 in flight.
- **The Olist filter paragraph** described three filters as two, never mentioned the
  right-censoring cut-off, and its parts did not sum to its whole: 1,723 in flight plus 1,188 not
  a delivery falls 8 short of 2,919, and those 8 are orders marked `delivered` that carry no
  delivery timestamp.
- **A note inside `results/olist_positive_control.json`** said the scanner found no pure
  single-column condition immediately above the three purity-1.0 conditions it had printed. They
  are pure-*negative* cells of 20 to 26 rows; the note now says so. Regenerating the file changed
  that string and the timestamp and nothing else, which is also a reproducibility check.
- **The publication-status block** would have become false at the moment of publication, because
  the publish script uploads the card verbatim: a reader on the Hub would have been told that
  nothing had been pushed and that the Hub still showed the September 2025 card. It is now worded
  so that it stays true on both sides.
- **Two overbroad claims** that every number comes from `results/eval.json`, in the demo and in
  the evaluation module's docstring; the Olist, scanner and training-curve numbers do not.
- **The training-loss figure** was titled "Training loss is exactly 0.0", which the card itself
  denies two sections later. The plotted series is the logged five-step mean rounded to four
  decimals, and it reaches 0.0000 between steps 45 and 70.
- **The number checker** expanded allowlist entries through the times-1e-6 rescaling, so
  allowlisting `5e-5` silently also allowlisted its times-1e-6 spellings, which appear in no
  result file. An allowlisted value now backs only itself.
- **`positive_control.download`** imported `huggingface_hub` even when every file was already on
  disk, so a fully offline re-run needed a network library it never called.

Three test files now guard what the presence check cannot see: the probe prose against
`results/eval.json`, the three licence statements against each other, and the publish manifest
against every in-repository link.

### Corrections to the publish script

- The card and both case studies link to `results/olist_positive_control.json`, and the
  repository-layout table names `docs/leakage_audit.md`. Neither was in the upload manifest,
  so both links would have been dead for anyone arriving from the Hub. They are uploaded now,
  and a test fails if a document links to a repository file the script does not publish.

### Attribution

- The vendored training script is pinned to upstream `136d2bf`, the last commit touching it
  before the September 2025 training, not to a commit dated after it. Against that commit the
  file differs by the attribution header and the three changes listed in
  `THIRD_PARTY_LICENSES.md`, and it is excluded from the formatter so the diff stays that small.
- Olist is CC BY-NC-SA 4.0. Its section in `THIRD_PARTY_LICENSES.md` records the
  attribution, the non-commercial condition, and that no rows are committed.

## v1.0.0 — September 2026: the audit release

The September 2025 artefact (`Yuchiwang02/DelaySentinel` on the Hub) was a bare model repo
with a Flask app and a README that had no metrics. This release turns it into a
reproducible case study. Nothing about the weights changed; everything about how they are
described did.

**Publication status.** This repository is complete and its checks pass. The Hub side
(rename to `Llama-3.2-1B-DelaySentinel`, the rewritten card, the licence files, the dataset
mirror and the demo Space) is pushed by `scripts/publish_hf.py`, which needs the author's
write token. A link in the README that does not resolve is a step of that script that has
not been run yet.

### Findings that drove the rewrite

- The Kaggle label is a deterministic two-field rule (`Shipment_Status == "Delayed" OR
  Traffic_Status == "Heavy"`, 1,000 of 1,000 rows; 118 of 118 rows that are both
  *Delivered* and *Heavy* are labelled delayed, so the label is a Boolean over two columns,
  not an outcome). The published model's 100% test accuracy is a measurement of that rule; a
  depth-2 decision tree scores the same.
- Counterfactual probes show the model's answer equals the rule's on every edit of the rule
  fields and on every edit of every other field; robustness probes show the learned rule is
  wider than the literal strings for some values and narrower for others; the model emits a
  label on prompts outside its schema.
- The original repo did not comply with the Llama 3.2 Community License (declared
  `apache-2.0`, no agreement copy, no "Built with Llama", no "Llama" prefix in the name)
  and did not attribute the training script (a copy of `acon96/home-llm`'s `train.py`, MIT,
  with Stanford Alpaca portions under Apache-2.0).

### Fixed in this repository

- **Licence and naming.** `LICENSE` (Llama 3.2 Community License), `USE_POLICY.md`,
  `NOTICE`, `LICENSE-MIT` (author's code), `LICENSES/Apache-2.0.txt`,
  `THIRD_PARTY_LICENSES.md`; front-matter `license: llama3.2`; "Built with Llama" on the
  card and in the demo; the publish script renames the repository to
  `Llama-3.2-1B-DelaySentinel` and ships the licence files with the Space.
- **Attribution.** `scripts/train.py` carries its upstream header (commit `136d2bf`, MIT,
  Alpaca portions Apache-2.0) below the original shebang and lists the local changes.
- **Model card.** Rewritten around the leakage disclosure. Removed: the non-existent Gradio
  Space claim, the placeholder repository link, the wrong repo id, the invented `<|system|>`
  prompt format, the foreign example schema, "AI-powered", "before shipment",
  "order-level".
- **Metadata.** `base_model` → `meta-llama/Llama-3.2-1B-Instruct` (evidence in
  `runs/RUNS.md`); `pipeline_tag` → `text-generation`; `library_name: transformers`.
- **Decoding config.** `generation_config.json` becomes greedy with `max_new_tokens: 8` (it
  shipped with the base model's sampling defaults); `config.json` `use_cache` is `true`
  again (training left it `false`).
- **Inference code.** The Flask app (local model path, `<pad>` token added and embeddings
  resized at inference, 512-token generation, `includes('1')` parsing, a form asking for the
  *delay reason* before predicting the delay) is retired to `legacy/`. `src/` uses the
  tokenizer's own `<|finetune_right_pad_id|>`, greedy decoding, a pinned chat-template date
  and an anchored regex; an unparsable answer is counted and never scored as 0.

### Added

- `src/delaysentinel/`: prompt utilities, frozen-split loader with newline-normalized
  hashes, metrics with Wilson / Clopper-Pearson / bootstrap intervals, eight baselines
  (including boosting without the rule fields, without the post-hoc reason field, and with
  Timestamp-derived features), counterfactual probes on every field, robustness probes
  (renames, order, case, synonyms, antonyms, negations, trigger placement, one field
  removed, unseen values) with teacher-forced margins, out-of-schema probes, CSV→JSONL
  transform, a generic leakage scanner, the case-study figures, and a positive control on
  real order-level data (Olist late-delivery, classical baselines only).
- `results/eval.json` (single source of every number the documents quote),
  `results/test_predictions.csv`, `results/leakage_audit.json`,
  `results/olist_positive_control.json`.
- `data/SPLIT.md`, `runs/RUNS.md`, the three runs' `training_config.json` /
  `trainer_state.json`, `runs/sc904/tensorboard_events.json`.
- `tests/` (pytest, no model download) covering the prompt format, the rule, the split
  hashes, the licence files, the number check itself, probe construction and baselines;
  a GitHub Actions workflow running the tests, the number check over seven documents and a
  baselines-only evaluation.
- `space/`: a Gradio demo whose point is to show the rule, not to predict delays.
- `docs/case_study.md` (English) and `docs/case_study.zh.md` (中文).

### Not changed on purpose

- The weights, the frozen `train.jsonl` / `test.jsonl` content, and the `training_args.bin`
  in the model repo. The historical split stays unseeded and un-replayable; it is hashed
  instead (line endings normalized to LF; the CRLF originals' hashes are recorded in
  `data/SPLIT.md` for provenance).
- No re-training and no claim of predictive value for the DelaySentinel weights. The
  positive control uses classical models on a different, real dataset and is reported
  separately.
