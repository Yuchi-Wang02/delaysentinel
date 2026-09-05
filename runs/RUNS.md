# Training runs

Three full-parameter SFT runs of `meta-llama/Llama-3.2-1B-Instruct` were made in September
2025 with `scripts/train.py` on the frozen split. Only `sc904` was published. Each folder
holds the run's `training_config.json` (exported from `training_args.bin`) and the
`trainer_state.json` of its final checkpoint.

| run | epochs | steps | eval loss first → min → last (×1e-6) | first step with train loss 0.0 | published | `model.safetensors` sha256 | `training_args.bin` sha256 |
| --- | ---: | ---: | --- | ---: | --- | --- | --- |
| `sc904` | 30 | 1,500 | 5.23 → 3.00 (step 1000) → 3.05 | 50 | **yes** (matches the Hub LFS hash) | `ffc509c65c539157905ccc5e469b1b37e30fe816aa9130477f57f0a3dfc2cd47` | `ec88943195fe977963f3c4552a550160471710cf35375b186c26c9d38b3fb8ae` |
| `sc9201` | 65 | 3,250 | 8.63 → 4.60 (step 2400) → 4.64 | 70 | no | `18b25eccf251bdff318548b325290234e9d016e8cd2617daadf281fb4261a10e` | `76ad179bacd2b153a1ae0c3076677a3db75c23035cc4e4ca863d0dd927771b04` |
| `sc920` | 100 | 5,000 | 4.46 → 2.30 (step 3200) → 2.33 | 45 | no | `8b13f1232b2bf491f220c803a4b5945d0cd702f0d4ccfcda8d1bc7e613979a2b` | `34f6b9cd7ba6cb538877d13adf7bc5539028218f113e453a0cae140904160143` |

Shared configuration (identical in all three `training_config.json` files): per-device
batch 4 × gradient accumulation 4 = effective batch 16; learning rate 2e-5, cosine
schedule, no warmup; weight decay 0.1; bf16; gradient checkpointing; `group_by_length`;
seed 42; `adamw_torch_fused`; eval every 200 steps; checkpoint every 100 steps (3 kept);
`transformers` 4.55.4.

## Things the numbers do not show

- **Wall-clock.** The command line and the GPU model were not logged. For `sc904` the two
  TensorBoard event files are 1,090 s apart (about 18 minutes from the first log to the
  final evaluation). No `train_runtime` is present in the checkpoint's trainer state.
- **The eval subset is the first 20 rows.** `CustomSFTTrainer._get_eval_sampler` draws
  `random.sample` indices but wraps them as `SequentialSampler(Subset(...))`; the sampler
  yields `0..19`, and the DataLoader indexes the *original* eval dataset with them. Every
  eval-loss point above is therefore measured on the first 20 rows of `test.jsonl`.
- **The scheduler never reaches zero.** `learning_rate_overshoot = 1.15` plans the cosine
  schedule for `int(1500 × 1.15) = 1725` steps and training stops at 1,500.
- **The pad mask hides real end-of-turn tokens.** With no pad token defined, the upstream
  collator uses `<|eot_id|>` as the pad value and `attention_mask = input_ids != eot`, so
  the genuine `<|eot_id|>` closing the system and user turns is masked during training.
  The assistant-side `<|eot_id|>` stays in the loss, so the model still stops.
- **Loss 0.0 is reached in the first epoch.** 800 rows / 16 = 50 optimizer steps per
  epoch; the training loss is exactly 0.0 from step 50 (`sc904`), 45 (`sc920`) and 70
  (`sc9201`). Everything after that is redundant compute.
- **Why 30 / 65 / 100 epochs.** No record. The published run is the *earliest* (folder
  dated 2025-09-04), not the one with the lowest eval loss, so the epoch count was not
  chosen on the test set.
