# Changelog

## v1.0.1 — September 2026: corrections from the internal review

A read-only multi-agent review of the v1.0.0 tree found claims that the evidence did not
support and one attribution error. Nothing here changes the weights or the frozen split.

### The finding is now stated precisely

v1.0.0 described the model as a bag-of-words detector for two trigger words. The probes
say something narrower: it matches the strings `Delay` and `Heavy`, case-blind,
truncation-tolerant (`Heav` alone fires), negation-blind (`Not Delayed` fires) and
field-agnostic. Twelve control words were added in both rule fields as a baseline; all but
one leave the prediction at baseline. The `Delay` match also fires from an appended
free-text line, while `Heavy` does not. The card and both case studies now say what the
matched pattern is, and state the part that is still open: the field name
`Logistics_Delay_Reason` contains `Delay` in every prompt and does not fire, so the match is
not a naive substring scan of the whole prompt.

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

### Attribution

- The vendored training script is pinned to upstream `136d2bf`, the last commit touching it
  before the September 2025 training, not to a commit dated after it. The file is byte-for-
  byte upstream plus a header, and is excluded from the formatter so it stays that way.
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
write token. Until that script has been run, the Hub still shows the September 2025 card
and the links in the README that point at the renamed repo, the dataset mirror and the
Space do not resolve. The README states this in its *Publication status* line.

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
