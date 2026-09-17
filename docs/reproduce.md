# Reproduce the checks

Choose the level of inspection you need. Saved results can be read without installing anything;
CPU checks verify the data rule and baselines; the full evaluation loads the published checkpoint.
Run commands from a fresh clone of the GitHub repository.

[Project overview](https://github.com/Yuchi-Wang02/delaysentinel) ·
[Evaluation report](evaluation.md) · [Olist reference](olist_reference.md)

## Inspect saved evidence

- [Recorded evaluation](../results/eval.json): baseline metrics, model scores, probe sets,
  configuration and source hashes.
- [Per-row predictions](../results/test_predictions.csv): original test inputs and recorded outputs.
- [Leakage audit](../results/leakage_audit.json): automatically recovered label rule.
- [Olist aggregates](../results/olist_positive_control.json): cohort definitions, classical models,
  calibration, month-level results and threshold sweep.
- [Historical split](../data/SPLIT.md) and [training records](../runs/RUNS.md).

The historical split is frozen. Do not create a new split to assess the existing checkpoint:
that could place records used in its training into a newly named test set.

## CPU checks without model downloads

Create and activate a virtual environment using Python 3.10 or newer. Then:

```bash
git clone https://github.com/Yuchi-Wang02/delaysentinel
cd delaysentinel
python -m pip install -r requirements-ci.txt
python -m pip install -e .
python -m pytest -q
python scripts/check_card_numbers.py
python -m delaysentinel.eval --skip-model --skip-cv --out outputs/eval_cpu.json
python -m delaysentinel.leakage_audit --csv data/smart_logistics_dataset.csv --target Logistics_Delay
```

`requirements-ci.txt` pins the scientific libraries used for the recorded baselines. The tests
do not download model weights. The baseline command writes to `outputs/`, leaving the committed
reference results intact. The leakage command prints its report unless an output path is supplied.

`--skip-cv` omits repeated cross-validation for a shorter check. Remove that flag to include it.
The CI comparison expects exact results for the rule, depth-2 tree, logistic regression, and
all-positive baseline. The tested gradient-boosting variants may differ by one of the 200 rows
across operating systems under the pinned environment; CI applies the documented tolerances.
See [the workflow](https://github.com/Yuchi-Wang02/delaysentinel/blob/main/.github/workflows/test.yml).

The documentation-number check confirms that numeric tokens occur in recorded outputs or a
reasoned allowlist. It does not validate a number's association with a particular claim.

## Full model evaluation

Use a separate environment if your existing packages have incompatible versions:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -e .
python -m delaysentinel.eval --model Yuchiwang02/Llama-3.2-1B-DelaySentinel --out outputs/eval.json --predictions-out outputs/test_predictions.csv
```

The full run downloads the checkpoint and evaluates the historical split, baselines, and
104 related probe sets. It uses bf16 and batch size 16 by default. A compatible PyTorch runtime
and device are needed; the saved run used an RTX 5070 Ti. The lock file records the package
versions used for the reference run, but hardware and low-level execution can still affect
timings and final numerical digits.

The program accepts a local model directory via `--model`, a device via `--device`, and
`--dtype fp32` for environments that do not support bf16. These are alternate execution
conditions, so their timing and margin values should not be represented as the original run.
Use `python -m delaysentinel.eval --help` for the complete options.

Original result files remain in `results/`. Compare your new `outputs/eval.json` with them;
a rerun supplies new evidence rather than replacing provenance by default. Probe sets share
source records and should not be interpreted as independent experiments.

## Separate Olist study

This command evaluates classical models and downloads the public order-data mirror. It does
not load or fine-tune the Llama checkpoint.

```bash
python -m pip install -r requirements-ci.txt
python -m pip install -e .
python -m pip install "huggingface_hub>=0.34,<1.0"
python -m delaysentinel.positive_control --out outputs/olist_positive_control.json
```

Downloads are cached in the git-ignored `data/olist/` directory. The result records source-file
hashes, the cohort filters, and the temporal split. See [the study](olist_reference.md) for its
delivered-only selection, calibration limits, and data terms.

## Render the recorded research figures

Install the figure dependency, then point the renderer at the committed results:

```bash
python -m pip install matplotlib
python -m delaysentinel.eda --eval results/eval.json --out outputs/figures
```

The renderer reads the Olist result alongside the selected evaluation JSON when available.
Using `outputs/figures` preserves the committed research images. To render a fresh full run,
use `--eval outputs/eval.json`; place the Olist aggregate with its expected filename in the same
directory if you also want that figure.

## Run the local demo

After installing the full environment and repository package:

```bash
python space/app.py
```

The demo loads the published checkpoint and illustrates input rewrites. It is an inspection
tool for the case study; no hosted demo is required to read or reproduce the recorded evidence.
See [demo setup](https://github.com/Yuchi-Wang02/delaysentinel/blob/main/space/README.md).

## Historical training

The [training command reconstruction](https://github.com/Yuchi-Wang02/delaysentinel/blob/main/scripts/train_sft.sh)
and [run records](../runs/RUNS.md) document the original setup. The upstream training script has
three local changes described in [the attribution record](../THIRD_PARTY_LICENSES.md).
The original unseeded shuffle cannot be regenerated; the frozen files can be reused and checked.
Retraining is a new experiment and is not required for this audit.
