"""Command-line evaluation: ``python -m delaysentinel.eval --model ... --out results/eval.json``.

Sections written to the JSON (all numbers in the model card come from here):

- ``provenance``            model id + weight hash, split hashes, library versions, GPU, git commit
- ``split``                 sizes, prevalence, train/test overlap check
- ``label_rule``            the two-clause rule replayed on all 1,000 CSV rows
- ``model``                 greedy metrics on the frozen test split, agreement with the rule,
                            sampling check, teacher-forced score diagnostics
- ``baselines_split_v0``    rule / all-positive / tree / logistic / boosting on the same rows
- ``baselines_cv``          seeded repeated CV for the sklearn baselines only
- ``counterfactual_probe``  edits to the rule fields and to non-rule fields
- ``robustness_probe``      renames, synonyms, column order, one field removed, unseen values
- ``ood_probe``             prompts outside the training schema
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
    user_text,
)
from .probes import counterfactual_probes, ood_probes, robustness_probes, run_probe
from .stats import classification_metrics

ROOT = Path(__file__).resolve().parents[2]


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def _weights_sha256(model_id: str) -> str | None:
    local = Path(model_id)
    if local.is_dir():
        f = local / "model.safetensors"
        return sha256_file(f) if f.exists() else None
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model_id, files_metadata=True)
        for s in info.siblings:
            if s.rfilename == "model.safetensors" and s.lfs:
                return s.lfs.sha256
    except Exception:
        return None
    return None


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
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--sampling-seeds", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--skip-model", action="store_true", help="baselines and rule only (CI)")
    p.add_argument("--skip-cv", action="store_true")
    p.add_argument("--skip-robustness", action="store_true")
    return p


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    t_start = time.time()

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
            "command_line": " ".join(sys.argv),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
            "seed": args.seed,
            "csv_sha256": sha256_file(args.csv),
            "train_jsonl_sha256": sha256_file(args.train),
            "test_jsonl_sha256": sha256_file(args.test),
            "model_id": None if args.skip_model else args.model,
            "hub_model_id": None if args.skip_model else args.hub_model_id,
            "model_safetensors_sha256": None if args.skip_model else _weights_sha256(args.model),
            "weights_note": None
            if args.skip_model
            else "compare model_safetensors_sha256 with the LFS hash shown on the Hub file page to confirm the evaluated bytes",  # noqa: E501
        },
        "split": {
            "train_n": len(train),
            "test_n": len(test),
            "train_pos_rate": round(float(train["gold"].mean()), 4),
            "test_pos_rate": round(float(test["gold"].mean()), 4),
            "test_pos": int(test["gold"].sum()),
            "test_neg": int((test["gold"] == 0).sum()),
            "prevalence_line": f"test {int(test['gold'].sum())}/{len(test)} = {test['gold'].mean():.3f}",
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

        print(f"[model] loading {args.model}", flush=True)
        scorer = Scorer(args.model, dtype=args.dtype, device=args.device, batch_size=args.batch_size)
        info = scorer.info()

        t0 = time.time()
        raw, labels = scorer.predict(users, max_new_tokens=args.max_new_tokens)
        infer_s = round(time.time() - t0, 2)
        preds = np.array([-1 if lab is None else lab for lab in labels])
        unparsable = int((preds == -1).sum())
        pv = np.where(preds == -1, 0, preds)  # only for metric computation; unparsable is reported separately
        scores = scorer.label_scores(users)
        p1 = np.array([s["p1"] for s in scores]) if scores else None
        model_metrics = classification_metrics(gold, pv, None, n_boot=args.n_boot, seed=args.seed)
        rule_on_test = rule_predict(test)
        results["model"] = {
            **info,
            "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens, "batch_size": args.batch_size},
            "test_inference_seconds": infer_s,
            "unparsable": unparsable,
            "raw_output_examples": raw[:5],
            "distinct_raw_outputs": sorted(set(raw)),
            "test_metrics_greedy": model_metrics,
            "agreement_with_rule": round(float((pv == rule_on_test).mean()), 4),
            "error_rows": [
                {
                    "index": int(i),
                    "gold": int(gold[i]),
                    "pred": int(pv[i]),
                    "raw": raw[i],
                    "Shipment_Status": str(test["Shipment_Status"].iloc[i]),
                    "Traffic_Status": str(test["Traffic_Status"].iloc[i]),
                }
                for i in np.flatnonzero(pv != gold)
            ],
        }
        if scores:
            diag = classification_metrics(gold, pv, p1, n_boot=args.n_boot, seed=args.seed)
            margins = np.array([s["margin"] for s in scores])
            results["model"]["teacher_forced_diagnostics"] = {
                "note": "hard-label model; scores are saturated and carry no ranking information; not a headline metric",  # noqa: E501
                "auroc": diag.get("auroc"),
                "auroc_note": diag.get("auroc_note"),
                "margin_min": round(float(margins.min()), 3),
                "margin_max": round(float(margins.max()), 3),
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
                "acc": round(float((np.where(arr == -1, 0, arr) == gold).mean()), 4),
                "unparsable": int((arr == -1).sum()),
            }
            if prev is not None:
                flips += int((arr != prev).sum())
            prev = arr
        results["model"]["sampling_check"] = {
            "config": {"do_sample": True, "temperature": 0.6, "top_p": 0.9},
            "per_seed": samp,
            "rows_changed_between_consecutive_seeds": flips,
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
        pred_frame.to_csv(args.predictions_out, index=False)

        print("[probes] counterfactual", flush=True)
        results["counterfactual_probe"] = {
            p.key: run_probe(scorer, p, gold) for p in counterfactual_probes(test, users)
        }
        if not args.skip_robustness:
            print("[probes] robustness", flush=True)
            results["robustness_probe"] = {p.key: run_probe(scorer, p, gold) for p in robustness_probes(test, users)}
        print("[probes] out-of-distribution", flush=True)
        results["ood_probe"] = {p.key: run_probe(scorer, p, gold) for p in ood_probes()}

    results["provenance"]["total_seconds"] = round(time.time() - t_start, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=1, ensure_ascii=False)
    print(f"wrote {args.out}")
    _print_summary(results)
    return results


def _print_summary(results: dict) -> None:
    lr = results["label_rule"]
    print(f"rule mismatches on {lr['n']} rows: {lr['mismatches']}")
    for name, m in results["baselines_split_v0"].items():
        if isinstance(m, dict) and "acc" in m:
            print(f"  {name:40s} acc {m['acc']:.3f}  f1 {m['f1']:.3f}  auroc {m.get('auroc', 'n/a')}")
    if "model" in results:
        m = results["model"]["test_metrics_greedy"]
        print(
            f"  {'model (greedy)':40s} acc {m['acc']:.3f}  f1 {m['f1']:.3f}  unparsable {results['model']['unparsable']}"  # noqa: E501
        )
        for key, r in results.get("counterfactual_probe", {}).items():
            print(
                f"  probe {key:34s} n={r['n']:3d} pos_rate={r['pred_pos_rate']} matches_expected={r.get('matches_expected')}"  # noqa: E501
            )


if __name__ == "__main__":
    main()
