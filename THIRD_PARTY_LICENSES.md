# Third-party licenses and attribution

## Model weights: Llama 3.2 Community License

The weights published as `Yuchiwang02/Llama-3.2-1B-DelaySentinel` are a fine-tune of
`meta-llama/Llama-3.2-1B-Instruct`. They are distributed under the **Llama 3.2 Community
License** (full text in `LICENSE`) and subject to the Acceptable Use Policy
(`USE_POLICY.md`). Section 1.b.i of that license requires distributors to provide a copy
of the agreement, to display "Built with Llama", and to begin the model name with
"Llama"; section 1.b.iii requires the attribution notice reproduced in `NOTICE`.

## Training script: acon96/home-llm (MIT)

`scripts/train.py` is an early revision of `train.py` from
[acon96/home-llm](https://github.com/acon96/home-llm) (compared item by item against
upstream commit `d352d88`, 2025-11-30; upstream removed the file on 2025-12-01 when it
moved to Axolotl, which is why the commit is pinned). Upstream's `LICENSES.txt` licenses
the project code under the MIT License with the copyright line **"Copyright 2024 Alex
O'Connell"**, and records that portions of the project are re-used from the Stanford
Alpaca codebase (Apache License 2.0, "Copyright 2023 Rohan Taori, Ishaan Gulrajani,
Tianyi Zhang, Yann Dubois, Xuechen Li").

Local changes to the upstream file, made in 2025: the `_get_train_sampler` signature was
updated to `(self, dataset)` for a newer `transformers`, the LoRA defaults were changed,
and two comments were added. Only the SFT + ShareGPT code path was exercised for this
project; the S3 upload, MFU, DPO, LoRA and quantisation branches were not used.
`CustomSFTTrainer` was used unchanged (it supplies the 10% eval subsample and the 1.15
schedule overshoot documented in `runs/RUNS.md`).

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

Portions derived from Stanford Alpaca are under the Apache License, Version 2.0
(http://www.apache.org/licenses/LICENSE-2.0).

## Data: Kaggle "Smart Logistics Supply Chain Dataset" (CC0)

`data/smart_logistics_dataset.csv` is the file published by user *ziya07* on Kaggle under
**CC0: Public Domain**
(https://www.kaggle.com/datasets/ziya07/smart-logistics-supply-chain-dataset). The
JSONL files in `data/` are a mechanical transformation of it.

## Everything else

The code under `src/`, `scripts/` (except `scripts/train.py`), `tests/`, `space/` and the
documentation are original work by Yuchi Wang and are released under the MIT License:

```
MIT License

Copyright (c) 2025-2026 Yuchi Wang

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
