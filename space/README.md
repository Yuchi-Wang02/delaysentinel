---
title: Llama-3.2-1B-DelaySentinel leakage demo
emoji: 🚚
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: mit
short_description: Watch a 1.24B LLM behave like a two-clause rule
models:
- Yuchiwang02/Llama-3.2-1B-DelaySentinel
---

# Llama-3.2-1B-DelaySentinel — leakage demo

**Built with Llama.**

This Space runs the published weights on CPU and lets you edit one record at a time. The
fine-tuned model reproduces the rule `Shipment_Status == "Delayed" OR Traffic_Status == "Heavy"`
that defines the Kaggle label; editing those two fields flips the answer, editing any other
field does not, and deleting them does not stop the model from answering.

The code of this Space is MIT (`LICENSE-MIT`). The weights it downloads at start-up are under
the Llama 3.2 Community License; copies of `LICENSE`, `USE_POLICY.md` and `NOTICE` are in this
Space's Files tab.

- Model card with evaluation, baselines and probes: https://huggingface.co/Yuchiwang02/Llama-3.2-1B-DelaySentinel
- Code and `results/eval.json`: https://github.com/Yuchi-Wang02/delaysentinel

The `src/` folder in this Space is a copy of the package in that repository.
