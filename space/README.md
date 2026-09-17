---
title: Llama-3.2-1B-DelaySentinel behavior demo
emoji: 🚚
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: mit
short_description: Explore how record edits change a model response
models:
- Yuchiwang02/Llama-3.2-1B-DelaySentinel
---

# Llama-3.2-1B-DelaySentinel: local behavior demo

**Built with Llama.**

Edit a logistics record and compare the published checkpoint's response with the dataset's
two-field label rule. The demo displays the original prompt, an edit that changes the rule's
answer, and a version with both rule fields removed. CSV mode supports the same comparison
across a batch of records.

The saved evaluation found exact agreement on the original historical split and differences
under targeted text rewrites. Use the demo to explore specific inputs; those observations do
not specify how the model will respond to every possible edit. Outputs describe behavior on
this instructional task and are not operational delivery-risk estimates.

## Run locally

This application is available in the repository. A hosted Hugging Face Space has **not** been
created. From the repository root, in the environment described in the
[reproduction guide](https://github.com/Yuchi-Wang02/delaysentinel/blob/main/docs/reproduce.md):

```bash
pip install -e . -r space/requirements.txt
python space/app.py
```

The first prediction downloads the published weights. The model wrapper selects an available
device; the result reports the device used. The `src/` package is loaded from the local
repository.

## Evidence and attribution

- [Published model and model card](https://huggingface.co/Yuchiwang02/Llama-3.2-1B-DelaySentinel)
- [Evaluation methods and saved probes](https://github.com/Yuchi-Wang02/delaysentinel/blob/main/docs/evaluation.md)
- [Source code and results](https://github.com/Yuchi-Wang02/delaysentinel)

Demo code is MIT. Downloaded model weights are governed by the Llama 3.2 Community License;
`LICENSE`, `USE_POLICY.md` and `NOTICE` are available in the
[model repository](https://huggingface.co/Yuchiwang02/Llama-3.2-1B-DelaySentinel/tree/main).
