#!/usr/bin/env bash
# Reconstructed from runs/sc904/training_config.json (the published run). NOT re-run.
#
# The original command line was not recorded; the flags below are the ones that
# produce exactly those TrainingArguments through scripts/train.py's
# TrainingRunArguments (batch_size / micro_batch_size -> gradient accumulation 4,
# eval_steps 200, save_steps 100, save_total_limit 3, bf16, gradient checkpointing,
# group_by_length, weight decay 0.1, cosine schedule, no warmup, 30 epochs).
#
# Known quirk inherited from upstream: CustomSFTTrainer.learning_rate_overshoot=1.15
# makes the cosine schedule plan for int(1500 * 1.15) = 1725 steps and stop at 1500,
# so the learning rate never decays to zero. The eval subset is the first 20 rows of
# --test_dataset (SequentialSampler over a Subset indexes the original dataset).
set -euo pipefail
python scripts/train.py \
  --run_name VCU-test \
  --base_model /path/to/meta-llama/Llama-3.2-1B-Instruct \
  --train_dataset data/train.jsonl \
  --test_dataset data/test.jsonl \
  --batch_size 16 --micro_batch_size 4 \
  --epochs 30 --learning_rate 2e-5 --learning_rate_schedule cosine --weight_decay 0.1 \
  --bf16 --gradient_checkpointing --group_by_length \
  --eval_steps 200 --save_steps 100 --save_total_limit 3
