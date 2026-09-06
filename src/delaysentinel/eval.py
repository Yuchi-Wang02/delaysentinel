"""Command-line evaluation: ``python -m delaysentinel.eval --model ... --out results/eval.json``.

Sections written to the JSON (all numbers in the model card come from here):

- ``provenance``            model id + weight hash (Hub LFS metadata when a Hub id is
                            given), split hashes, library versions, GPU, git commit,
                            the pinned chat-template date and the rendered system turn
- ``split``                 sizes, prevalence, train/test overlap check
- ``label_rule``            the two-clause rule replayed on all 1,000 CSV rows
- ``model``                 greedy metrics on the frozen test split (parsable rows only;
                            unparsable outputs are counted, never scored as 0), agreement
                            with the rule, sampling check, teacher-forced diagnostics
- ``baselines_split_v0``    rule / all-positive / tree / logistic / boosting variants
- ``baselines_cv``          seeded repeated CV for the sklearn baselines only
- ``counterfactual_probe``  edits to the rule fields and to every non-rule field
- ``robustness_probe``      renames, order, case/synonym/antonym variants, trigger
                            placement, one field removed, unseen values
- ``ood_probe``             prompts outside the training schema

Run from the repository root (or pass explicit paths).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .baselines import FEATURE_ENCODING, cross_validate, evaluate_split
from .data import (
    RULE_TEXT,
    category_counts,
    frame_from_jsonl,
    load_csv,
    load_jsonl,
    rule_crosstab,
    rule_predict,
    sha256_file,
    sha256_normalized_newlines,
    user_text,
)
from .probes import counterfactual_probes, ood_probes, robustness_probes, run_probe, summarize_counterfactuals
from .stats import classification_metrics

_PKG_ROOT = Path(__file__).resolve().parents[2]
ROOT = _PKG_ROOT if (_PKG_ROOT / "data" / "test.jsonl").exists() else Path.cwd()
HUB_IDS = ("Yuchiwang02/Llama-3.2-1B-DelaySentinel", "Yuchiwang02/DelaySentinel")


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # provenance is best effort
        return None


def _hub_lfs_sha256(repo_id: str) -> str | None:
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo_id, files_metadata=True)
        for s in info.siblings:
            if s.rfilename == "model.safetensors" and s.lfs:
                return s.lfs.sha256
    except Exception:
        return None
    return None


def _weights_provenance(model_id: str, hub_model_id: str) -> dict:
    local = Path(model_id)
    out: dict = {}
    if local.is_dir():
        f = local / "model.safetensors"
        out["model_id"] = "local directory (path not recorded)"
        out["model_safetensors_sha256"] = sha256_file(f) if f.exists() else None
        out["model_safetensors_sha256_source"] = "local file"
    else:
        out["model_id"] = model_id
        out["model_safetensors_sha256"] = _hub_lfs_sha256(model_id)
        out["model_safetensors_sha256_source"] = "Hub LFS metadata of model_id"
    out["hub_model_id"] = hub_model_id
    hub_sha, hub_used = None, None
    for cand in (hub_model_id, *HUB_IDS):
        hub_sha = _hub_lfs_sha256(cand)
        if hub_sha:
            hub_used = cand
            break
    out["hub_model_safetensors_sha256"] = hub_sha
    out["hub_id_used_for_hash"] = hub_used
    out["weights_match_hub"] = (
        None if not (hub_sha and out["model_safetensors_sha256"]) else hub_sha == out["model_safetensors_sha256"]
    )
    return out


def _versions() -> dict:
    import numpy
    import pandas
    import scipy
    import sklearn

    return {
        "numpy_version": numpy.__version__,
        "pandas_version": pandas.__version__,
        "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__,
    }


def _public_command_line(argv: list[str]) -> str:
    """The command line without local absolute paths."""
    parts = [Path(argv[0]).name]
    for a in argv[1:]:
        if (":" in a and "\\" in a) or a.startswith("/") or (len(a) > 2 and a[1] == ":"):
            parts.append("<local path>")
        else:
            parts.append(a)
    return " ".join(parts)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate DelaySentinel against the frozen split and baselines.")
    p.add_argument("--model", default="Yuchiwang02/Llama-3.2-1B-DelaySentinel", help="HF id or local directory")
    p.add_argument(
        "--hub-model-id",
        default="Yuchiwang02/Llama-3.2-1B-DelaySentinel",
        help="the Hub repository these weights are published under (recorded in provenance)",
    )
    p.add_argument("--train", default=str(ROOT / "data" / "train.jsonl"))
    p.add_argument("--test", default=str(ROOT / "data" / "test.jsonl"))
    p.add_argument("--csv", default=str(ROOT / "data" / "smart_logistics_dataset.csv"))
    p.add_argument("--out", default=str(ROOT / "results" / "eval.json"))
    p.add_argument("--predictions-out", default=str(ROOT / "results" / "test_predictions.csv"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device", default=None)
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=8,
        help="the answer is six tokens plus <|eot_id|>; do not go below 7",
    )
    p.add_argument("--sampling-seeds", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--skip-model", action="store_true", help="baselines and rule only (CI)")
    p.add_argument("--skip-cv", action="store_true")
    p.add_argument("--skip-robustness", action="store_true")
    return p


def _scored_metrics(gold: np.ndarray, preds: np.ndarray, n_boot: int, seed: int) -> dict:
    """Metrics on parsable rows only; unparsable rows are reported, never scored as 0."""
    mask = preds != -1
    out = classification_metrics(gold[mask], preds[mask], n_boot=n_boot, seed=seed) if mask.any() else {}
    out["n_scored"] = int(mask.sum())
    out["unparsable"] = int((~mask).sum())
    out["acc_counting_unparsable_as_wrong"] = round(float(((preds == gold) & mask).sum() / len(gold)), 4)
    return out


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    t_start = time.time()
    for path in (args.train, args.test, args.csv):
        if not Path(path).exists():
            raise SystemExit(f"missing {path}: run from the repository root or pass --train/--test/--csv")

    train_rows = load_jsonl(args.train)
    test_rows = load_jsonl(args.test)
    train = frame_from_jsonl(train_rows)
    test = frame_from_jsonl(test_rows)
    users = [user_text(r) for r in test_rows]
    gold = test["gold"].to_numpy()
    csv = load_csv(args.csv)

    results: dict = {
        "provenance": {
            "package_version": __version__,
            "command_line": _public_command_line(sys.argv),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
            "seed": args.seed,
            "hash_note": (
                "file hashes are sha256 over newline-normalized bytes (CRLF -> LF); "
                "on an LF checkout they equal the raw file hash"
            ),
            "csv_sha256": sha256_normalized_newlines(args.csv),
            "train_jsonl_sha256": sha256_normalized_newlines(args.train),
            "test_jsonl_sha256": sha256_normalized_newlines(args.test),
            **_versions(),
        },
        "split": {
            "train_n": len(train),
            "test_n": len(test),
            "train_pos_rate": round(float(train["gold"].mean()), 4),
            "test_pos_rate": round(float(test["gold"].mean()), 4),
            "test_pos": int(test["gold"].sum()),
            "test_neg": int((test["gold"] == 0).sum()),
            "test_prompts_also_in_train": int(sum(u in {user_text(r) for r in train_rows} for u in users)),
            "split_method": "random.shuffle without a seed, 80/20, generated once in Sept 2025; not reproducible",
        },
        "label_rule": {
            "rule": RULE_TEXT,
            **rule_crosstab(csv),
            "csv_rows": len(csv),
            "label_counts": {
                "delayed": int(csv["Logistics_Delay"].sum()),
                "not_delayed": int((csv["Logistics_Delay"] == 0).sum()),
            },
            "category_counts": category_counts(csv),
            "reason_present_by_label": {
                "delayed": int(
                    ((csv["Logistics_Delay_Reason"].astype(str) != "None") & (csv["Logistics_Delay"] == 1)).sum()
                ),
                "not_delayed": int(
                    ((csv["Logistics_Delay_Reason"].astype(str) != "None") & (csv["Logistics_Delay"] == 0)).sum()
                ),
            },
            "rule_on_test_split": classification_metrics(gold, rule_predict(test), n_boot=args.n_boot, seed=args.seed),
        },
    }

    print("[baselines] frozen split v0", flush=True)
    results["baselines_split_v0"] = evaluate_split(train, test, seed=args.seed, n_boot=args.n_boot)
    results["baselines_split_v0"]["feature_encoding"] = FEATURE_ENCODING
    if not args.skip_cv:
        print("[baselines] repeated CV", flush=True)
        both = pd.concat([train, test], ignore_index=True)
        results["baselines_cv"] = cross_validate(both, seed=args.seed)

    if not args.skip_model:
        from .model import Scorer

        results["provenance"].update(_weights_provenance(args.model, args.hub_model_id))
        print(f"[model] loading {args.model}", flush=True)
        scorer = Scorer(args.model, dtype=args.dtype, device=args.device, batch_size=args.batch_size)
        info = scorer.info()
        if Path(args.model).is_dir():
            info["model_id"] = "local directory (path not recorded)"

        t0 = time.time()
        raw, labels = scorer.predict(users, max_new_tokens=args.max_new_tokens)
        infer_s = round(time.time() - t0, 2)
        preds = np.array([-1 if lab is None else lab for lab in labels])
        scores = scorer.label_scores(users)
        p1 = np.array([s["p1"] for s in scores]) if scores else None
        rule_on_test = rule_predict(test)
        parsable = preds != -1
        results["model"] = {
            **info,
            "decoding": {
                "do_sample": False,
                "max_new_tokens": args.max_new_tokens,
                "batch_size": args.batch_size,
                "template_date_string": info["template_date_string"],
            },
            "test_inference_seconds": infer_s,
            "raw_output_examples": raw[:5],
            "distinct_raw_outputs": sorted(set(raw)),
            "test_metrics_greedy": _scored_metrics(gold, preds, args.n_boot, args.seed),
            "agreement_with_rule": round(float(((preds == rule_on_test) & parsable).sum() / len(gold)), 4),
            "error_rows": [
                {
                    "index": int(i),
                    "gold": int(gold[i]),
                    "pred": None if preds[i] == -1 else int(preds[i]),
                    "raw": raw[i],
                    "Shipment_Status": str(test["Shipment_Status"].iloc[i]),
                    "Traffic_Status": str(test["Traffic_Status"].iloc[i]),
                }
                for i in np.flatnonzero(preds != gold)
            ],
        }
        if scores:
            margins = np.array([s["margin"] for s in scores])
            diag = classification_metrics(gold, np.where(preds == -1, 0, preds), p1, n_boot=args.n_boot, seed=args.seed)
            results["model"]["teacher_forced_diagnostics"] = {
                "note": (
                    "hard-label model; the logit margin after 'Logistics_Delay:' is saturated, so the implied "
                    "probabilities are 0 or 1 and the AUROC restates accuracy (no calibration, no usable uncertainty)"
                ),
                "batch_size": args.batch_size,
                "dtype": args.dtype,
                "auroc": diag.get("auroc"),
                "margin_min": round(float(margins.min()), 3),
                "margin_max": round(float(margins.max()), 3),
                "margin_abs_min": round(float(np.abs(margins).min()), 3),
                "margin_abs_median": round(float(np.median(np.abs(margins))), 3),
                "p1_min": round(float(p1.min()), 6),
                "p1_max": round(float(p1.max()), 6),
                "top_token_is_label_rate": round(float(np.mean([s["top_token_is_label"] for s in scores])), 4),
            }
        # sampling check with the generation config the repo originally shipped
        samp = {}
        prev = None
        flips = 0
        for seed in args.sampling_seeds:
            _, ls = scorer.predict(
                users, do_sample=True, temperature=0.6, top_p=0.9, seed=seed, max_new_tokens=args.max_new_tokens
            )
            arr = np.array([-1 if lab is None else lab for lab in ls])
            samp[str(seed)] = {
                "acc_counting_unparsable_as_wrong": round(float(((arr == gold) & (arr != -1)).sum() / len(gold)), 4),
                "unparsable": int((arr == -1).sum()),
            }
            if prev is not None:
                flips += int((arr != prev).sum())
            prev = arr
        results["model"]["sampling_check"] = {
            "config": {"do_sample": True, "temperature": 0.6, "top_p": 0.9},
            "per_seed": samp,
            "rows_changed_between_consecutive_seeds": flips,
            "note": "with saturated margins this outcome follows from the greedy result; it is not independent evidence",  # noqa: E501
        }

        # per-row predictions for auditing
        pred_frame = test.copy()
        pred_frame["rule"] = rule_on_test
        pred_frame["model_raw"] = raw
        pred_frame["model_label"] = [("" if lab is None else lab) for lab in labels]
        if scores:
            pred_frame["margin_1_minus_0"] = margins
            pred_frame["p1"] = p1
        Path(args.predictions_out).parent.mkdir(parents=True, exist_ok=True)
        pred_frame.to_csv(args.predictions_out, index=False, lineterminator="\n")

        print("[probes] counterfactual", flush=True)
        cf = counterfactual_probes(test, users)
        results["counterfactual_probe"] = {p.key: run_probe(scorer, p, gold) for p in cf}
        results["counterfactual_probe"]["_summary"] = summarize_counterfactuals(results["counterfactual_probe"], cf)
        if not args.skip_robustness:
            print("[probes] robustness", flush=True)
            results["robustness_probe"] = {p.key: run_probe(scorer, p, gold) for p in robustness_probes(test, users)}
        print("[probes] out-of-distribution", flush=True)
        results["ood_probe"] = {p.key: run_probe(scorer, p, gold) for p in ood_probes()}

    results["provenance"]["total_seconds"] = round(time.time() - t_start, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(results, handle, indent=1, ensure_ascii=False)
        handle.write("\n")
    print(f"wrote {args.out}")
    _print_summary(results)
    return results


def _print_summary(results: dict) -> None:
    lr = results["label_rule"]
    print(f"rule mismatches on {lr['n']} rows: {lr['mismatches']}")
    for name, m in results["baselines_split_v0"].items():
        if isinstance(m, dict) and "acc" in m:
            print(f"  {name:48s} acc {m['acc']:.3f}  f1 {m['f1']:.3f}  auroc {m.get('auroc', 'n/a')}")
    if "model" in results:
        m = results["model"]["test_metrics_greedy"]
        print(f"  {'model (greedy)':48s} acc {m.get('acc', float('nan')):.3f}  unparsable {m['unparsable']}")
        print("  counterfactual summary:", results["counterfactual_probe"]["_summary"])


if __name__ == "__main__":
    main()
