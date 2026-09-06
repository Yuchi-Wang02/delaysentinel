# Licences and attribution

| what | licence | file |
| --- | --- | --- |
| model weights (`model.safetensors` on the Hub) | Llama 3.2 Community License + Acceptable Use Policy | `LICENSE`, `USE_POLICY.md`, `NOTICE` |
| author's code, tests, demo, documentation, figures, generated result files | MIT | `LICENSE-MIT` |
| `scripts/train.py` (adapted from acon96/home-llm) | MIT (upstream) with portions under Apache-2.0 (Stanford Alpaca) | this file, `LICENSES/Apache-2.0.txt` |
| `data/smart_logistics_dataset.csv` and the derived JSONL split | CC0-1.0 | this file, `data/SPLIT.md` |

## Model weights: Llama 3.2 Community License

The weights published as `Yuchiwang02/Llama-3.2-1B-DelaySentinel` are a fine-tune of
`meta-llama/Llama-3.2-1B-Instruct`. They are distributed under the **Llama 3.2 Community
License** (full text in `LICENSE`) and subject to the Acceptable Use Policy
(`USE_POLICY.md`). Section 1.b.i of that licence requires distributors to provide a copy
of the agreement, to display "Built with Llama", and to begin the model name with
"Llama"; section 1.b.iii requires the attribution notice reproduced in `NOTICE`.

## Training script: acon96/home-llm (MIT) with Stanford Alpaca portions (Apache-2.0)

`scripts/train.py` is an early revision of `train.py` from
[acon96/home-llm](https://github.com/acon96/home-llm), compared item by item against
upstream commit `d352d88` (2025-11-30). The file is no longer present on upstream `main`
(checked 2026-09-05); upstream's move to Axolotl began with commit `55f2541` on 2025-12-01,
which is why the commit is pinned here. Upstream's `LICENSES.txt` licenses the project code
under the MIT License with the copyright line **"Copyright 2024 Alex O'Connell"**, and records
that portions of the project are re-used from the Stanford Alpaca codebase (Apache License 2.0,
"Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li"). The
data-collator structure in `scripts/train.py` is one of those portions; the full Apache-2.0
text is in `LICENSES/Apache-2.0.txt`.

Local changes to the upstream file, made in 2025: the `_get_train_sampler` signature was
updated to `(self, dataset)` for a newer `transformers`, the LoRA defaults were changed, and
two comments were added. Only the SFT + ShareGPT code path was exercised for this project;
the S3 upload, MFU, DPO, LoRA and quantisation branches were not used. `CustomSFTTrainer` was
used unchanged (it supplies the 10 % eval subsample and the 1.15 schedule overshoot documented
in `runs/RUNS.md`).

```
MIT License

Copyright 2024 Alex O'Connell

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Stanford Alpaca header, as carried by upstream:

```
Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## Data: Kaggle "Smart Logistics Supply Chain Dataset" (CC0)

`data/smart_logistics_dataset.csv` is the file published by user *ziya07* on Kaggle
(https://www.kaggle.com/datasets/ziya07/smart-logistics-supply-chain-dataset). The Kaggle
page listed the licence as **CC0: Public Domain** when the page was read on 2026-09-03
(dataset "updated 2 years ago", one version, one 106 KB file). CC0 is the uploader's
declaration; nothing else about the data's provenance is documented on that page, and this
repository shows the content to be synthetic. The JSONL files in `data/`, including the
system prompt they contain, are the author's mechanical transformation of that CSV and are
likewise dedicated to the public domain under CC0-1.0.

## Author's code and documentation (MIT)

Everything under `src/`, `tests/`, `scripts/` (except `scripts/train.py`), `space/`,
`legacy/` (all files, including `Test.py` and the Flask templates), `docs/` (text and
figures), and the generated files under `results/` and `runs/`, is original work by
Yuchi Wang released under the MIT License (`LICENSE-MIT`).
