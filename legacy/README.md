# Legacy (September 2025) code, kept for provenance

These files are the original pipeline exactly as it was run. They are **not maintained**
and are superseded by `src/delaysentinel/`.

| file | what it did | superseded by |
| --- | --- | --- |
| `Transform.py` | CSV → ShareGPT JSONL with a hard-coded Windows path | `delaysentinel.transform` |
| `shuffle.py` | unseeded `random.shuffle` 80/20 split | `delaysentinel.transform split --seed` (frozen files are not re-split) |
| `Test.py` | printed two generations (one of them a smart-home prompt from an earlier course project); computed no metric | `delaysentinel.eval` |
| `flask_ui/app.py`, `flask_ui/templates/index.html` | Flask form UI; loads the model from a local `models/VCU-test` path, adds a `<pad>` token and resizes the embeddings at inference time, generates up to 512 tokens for a 0/1 answer, parses the result with `includes('1')`, and asks the user to type the *delay reason* before predicting the delay | `space/app.py` (Gradio demo) and `delaysentinel.predict` |

Known defects are listed in the model card's *Limitations and known issues* section. The
training script lives in `scripts/train.py` with its attribution header.
