# Licences and attribution

| what | licence | file |
| --- | --- | --- |
| model weights (`model.safetensors` on the Hub) | Llama 3.2 Community License + Acceptable Use Policy | `LICENSE`, `USE_POLICY.md`, `NOTICE` |
| author's code, tests, demo, documentation, figures, generated result files (except `results/olist_positive_control.json`) | MIT | `LICENSE-MIT` |
| `scripts/train.py` (third party, kept byte-for-byte) | MIT (acon96/home-llm) with portions under Apache-2.0 (Stanford Alpaca) | this file, `LICENSES/Apache-2.0.txt` |
| `data/smart_logistics_dataset.csv` and the derived JSONL split | CC0-1.0 | this file, `data/SPLIT.md` |
| Olist orders used by the reference study (downloaded at run time, never committed) | CC BY-NC-SA 4.0 | this file |

## Model weights: Llama 3.2 Community License

The weights published as `Yuchiwang02/Llama-3.2-1B-DelaySentinel` are a fine-tune of
`meta-llama/Llama-3.2-1B-Instruct`. They are distributed under the **Llama 3.2 Community
License** (full text in `LICENSE`) and subject to the Acceptable Use Policy
(`USE_POLICY.md`). Section 1.b.i of that licence requires distributors to provide a copy
of the agreement, to display "Built with Llama", and to begin the model name with
"Llama"; section 1.b.iii requires the attribution notice reproduced in `NOTICE`.

## Training script: acon96/home-llm (MIT) with Stanford Alpaca portions (Apache-2.0)

`scripts/train.py` is **not the author's work**. It is `train.py` from
[acon96/home-llm](https://github.com/acon96/home-llm), kept byte-for-byte as it was run in
September 2025 apart from an added header comment; it is excluded from this project's
formatter and linter for that reason.

Upstream's `LICENSES.txt` licenses the project code under the MIT License with the copyright
line **"Copyright 2024 Alex O'Connell"**, and records that portions of the project are re-used
from the Stanford Alpaca codebase (Apache License 2.0, "Copyright 2023 Rohan Taori, Ishaan
Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li"). The data-collator structure in
`scripts/train.py` is one of those portions; the full Apache-2.0 text is in
`LICENSES/Apache-2.0.txt`.

**Which upstream revision.** The last commit that touched `train.py` before this project's
training run (September 2025) is **`136d2bf`, 2025-02-26**. Upstream later removed the file
(the Axolotl migration began with `55f2541` on 2025-12-01). Diffing the copy in this repository
against `136d2bf`, ignoring trailing whitespace that the author's editor stripped, leaves exactly
three changes, all made by the author in 2025:

1. `_get_train_sampler(self)` became `_get_train_sampler(self, dataset)`, with the body and
   `super()` call updated to match, plus a one-line comment and a docstring. A newer
   `transformers` release passes the dataset to that hook.
2. A duplicated import block (`random`, `torch`, the samplers, `Trainer`) was inserted above
   `CustomSFTTrainer`.
3. `model_path = training_run_args.base_model` was introduced and used at the model and
   tokenizer loading calls, with two Chinese comments (`# 使用本地路径`, `# 其他模型参数`).

No default value, class or function was otherwise changed; the LoRA, DPO, S3, MFU and
quantisation code is upstream's and was never exercised here. `CustomSFTTrainer` was used
unchanged, which is where the 10 % eval subsample and the 1.15 schedule overshoot documented in
`runs/RUNS.md` come from.

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
(dataset "updated 2 years ago", one version, one file). CC0 is the uploader's
declaration; nothing else about the data's provenance is documented on that page, and this
repository shows the content to be synthetic. The JSONL files in `data/`, including the
system prompt they contain, are the author's mechanical transformation of that CSV and are
likewise dedicated to the public domain under CC0-1.0.

## Reference study data: Olist (CC BY-NC-SA 4.0)

`src/delaysentinel/positive_control.py` downloads the **Brazilian E-Commerce Public Dataset by
Olist** (https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), published by Olist and
three collaborators and licensed **CC BY-NC-SA 4.0**
(https://creativecommons.org/licenses/by-nc-sa/4.0/) as stated on that page when it was read on
2026-09-06. Attribution: "Brazilian E-Commerce Public Dataset by Olist", licensed CC BY-NC-SA 4.0.

Use here is **non-commercial research and teaching**. The files are fetched at run time from the
public Hugging Face mirror `aviahYadler/Olist_Ecommerce_Dataset` (which carries no licence
statement of its own) into the git-ignored `data/olist/`; the sha256 of each file is recorded in
`results/olist_positive_control.json` → `provenance.file_sha256`. **No Olist row is committed to
this repository.** `results/olist_positive_control.json` contains only aggregate statistics
computed from that data; as a derivative of a ShareAlike work it is offered under
CC BY-NC-SA 4.0 rather than the MIT terms below, and `src/delaysentinel/positive_control.py`
(the code, which contains no Olist data) stays MIT.

## Author's code and documentation (MIT)

Everything under `src/`, `tests/`, `scripts/` (except `scripts/train.py`), `space/`,
`legacy/` (all files, including `Test.py` and the Flask templates), `docs/` (text and
figures), and the generated files under `results/` (except
`results/olist_positive_control.json`, see above) and `runs/`, is original work by
Yuchi Wang released under the MIT License (`LICENSE-MIT`).
