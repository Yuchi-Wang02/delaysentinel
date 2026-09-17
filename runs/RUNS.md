# Training runs

Three full-parameter SFT runs of `meta-llama/Llama-3.2-1B-Instruct` were made in September
2025 with `scripts/train.py` on the frozen split. Only `sc904` was published. Each folder
holds the run's `training_config.json` (exported from `training_args.bin`) and the
`trainer_state.json` of its final checkpoint; `sc904/` also has `tensorboard_events.json`
(names, sizes, hashes and timestamps of the two TensorBoard event files, which are not
committed).

The runs used full-parameter updates rather than LoRA adapters. `scripts/train.py` comes from
acon96/home-llm with the three project-specific modifications recorded in
[`THIRD_PARTY_LICENSES.md`](../THIRD_PARTY_LICENSES.md).

| run | epochs | steps | eval loss first → min → last (×1e-6) | first log point with loss 0.0000 (epoch) | published | `model.safetensors` sha256 | `training_args.bin` sha256 |
| --- | ---: | ---: | --- | --- | --- | --- | --- |
| `sc904` | 30 | 1,500 | 5.23 → 3.00 (step 1000) → 3.05 | 50 (1.0) | **yes** | `ffc509c65c539157905ccc5e469b1b37e30fe816aa9130477f57f0a3dfc2cd47` | `ec88943195fe977963f3c4552a550160471710cf35375b186c26c9d38b3fb8ae` |
| `sc9201` | 65 | 3,250 | 8.63 → 4.60 (step 2400) → 4.64 | 70 (1.4) | no | `18b25eccf251bdff318548b325290234e9d016e8cd2617daadf281fb4261a10e` | `76ad179bacd2b153a1ae0c3076677a3db75c23035cc4e4ca863d0dd927771b04` |
| `sc920` | 100 | 5,000 | 4.46 → 2.30 (step 3200) → 2.33 | 45 (0.9) | no | `8b13f1232b2bf491f220c803a4b5945d0cd702f0d4ccfcda8d1bc7e613979a2b` | `34f6b9cd7ba6cb538877d13adf7bc5539028218f113e453a0cae140904160143` |

The `sc904` weight hash is the local file's sha256. `results/eval.json` → `provenance` records
the LFS hash of `model.safetensors` on the Hub next to it; note that when the evaluation is run
with a Hub id as `--model`, both hashes come from the same repository and their agreement is not
an independent check (the JSON says so).

Shared configuration (identical in all three `training_config.json` files): per-device
batch 4 × gradient accumulation 4 = effective batch 16; learning rate 2e-5, cosine
schedule, no warmup; weight decay 0.1; bf16; gradient checkpointing; `group_by_length`;
seed 42; `adamw_torch_fused`; eval every 200 steps; checkpoint every 100 steps (3 kept);
logging every 5 steps. The `transformers` version of the *training* environment is not in
`training_args.bin` and no file in this repository records it; the version in
`results/eval.json` is the 2026 evaluation environment.

## Interpreting the training record

- **Wall-clock.** The command line and the GPU model were not logged. For `sc904` the two
  TensorBoard event files were created 1,090 s apart (`tensorboard_events.json`: first
  training log at epoch 1757017508, final `evaluate_all()` file at 1757018598), about
  18 minutes from the first training log to the end of the run. Model loading and tokenisation
  happened before the first log and are not included. No `train_runtime` is present in the
  checkpoint's trainer state.
- **The eval subset is the first 20 rows.** `CustomSFTTrainer._get_eval_sampler` draws
  `random.sample` indices but wraps them as `SequentialSampler(Subset(...))`; the sampler
  yields `0..19`, and the DataLoader indexes the *original* eval dataset with them. Every
  eval-loss point above is therefore measured on the first 20 rows of `test.jsonl`. The
  `eval_samples_per_second` and `eval_steps_per_second` fields in `trainer_state.json` are
  computed by the Trainer from `len(eval_dataset) = 200` and `ceil(200 / 4) = 50`, not from
  the 20 rows the sampler actually yields; do not read them as evidence that 200 rows were
  evaluated.
- **"Loss 0.0" is the logger's rounding.** The Trainer logs the 5-step mean loss rounded to
  4 decimals, so the value reads 0.0000 once the true loss is below 5e-5. It reaches that
  point within the first two epochs (log points 50, 45 and 70; 50 optimizer steps per epoch).
  The gradient norm is still 0.0006 at the last step of `sc904`, so the true loss is small,
  not zero.
- **The scheduler never reaches zero.** `learning_rate_overshoot = 1.15` plans the cosine
  schedule for `int(1500 × 1.15) = 1725` steps and training stops at 1,500; the learning
  rate at step 1,500 is 8.3e-07.
- **The pad mask hides real end-of-turn tokens.** With no pad token defined, the upstream
  collator uses the tokenizer's eos token (`<|eot_id|>` for the Instruct tokenizer) as the
  pad value and `attention_mask = input_ids != eos`, so the genuine `<|eot_id|>` closing the
  system and user turns is masked during training. The assistant-side `<|eot_id|>` stays in
  the loss, so the model still stops.
- **Epoch choice and behavioral evidence.** The archive does not record the rationale for
  30 / 65 / 100 epochs. The published run is the earliest (folder dated 2025-09-04), and a
  different run has a lower recorded eval loss; this does not establish the full selection
  procedure. No checkpoint before step 1,300 survives. The prompt-rewrite results in
  `results/eval.json` describe the final step-1,500 weights and do not locate when those
  behaviors emerged during training.
- **Which base checkpoint.** The training command line was not logged. The published
  `config.json` carries the Instruct `eos_token_id` list `[128001, 128008, 128009]` and the
  repository ships the Instruct `chat_template.jinja`; the base `Llama-3.2-1B` has a single
  eos id and no chat template. The September 2025 card's front-matter `base_model:
  meta-llama/Llama-3.2-1B` was a metadata error.

For the checkpoint's measured outputs and probe conditions, see
[`docs/evaluation.md`](../docs/evaluation.md). Training-loss values describe optimization on
this task; they do not establish performance on future delivery outcomes.
