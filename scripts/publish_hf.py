#!/usr/bin/env python
"""Publish the model card, licences, results, dataset split and demo Space to the Hub.

Requires a *write* token in the local Hugging Face credential store
(`hf auth login` / `huggingface-cli login`). Nothing here touches the model weights.

    python scripts/publish_hf.py --dry-run          # show the plan only
    python scripts/publish_hf.py                    # rename, dataset, space, model card (in that order)

Steps (default order)
  --rename   move Yuchiwang02/DelaySentinel to Yuchiwang02/Llama-3.2-1B-DelaySentinel (the Hub keeps a redirect)
  --dataset  create Yuchiwang02/smart-logistics-delay-split-v0 with the CSV, the frozen JSONL split and SPLIT.md
  --space    create the Gradio Space Yuchiwang02/delaysentinel-leakage-demo from space/ plus src/ and the licences
  --model    upload README.md, licences, results/, runs/, docs/, fixed generation_config.json / config.json;
             delete app.py and templates/ (done last so every link in the card resolves when it goes live)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD_MODEL_ID = "Yuchiwang02/DelaySentinel"
NEW_MODEL_ID = "Yuchiwang02/Llama-3.2-1B-DelaySentinel"
DATASET_ID = "Yuchiwang02/smart-logistics-delay-split-v0"
SPACE_ID = "Yuchiwang02/delaysentinel-leakage-demo"

LICENCE_FILES = [
    "LICENSE",
    "USE_POLICY.md",
    "NOTICE",
    "THIRD_PARTY_LICENSES.md",
    "LICENSE-MIT",
    "LICENSES/Apache-2.0.txt",
]
MODEL_FILES = [
    "README.md",
    *LICENCE_FILES,
    "CHANGELOG.md",
    "results/eval.json",
    "results/test_predictions.csv",
    "results/leakage_audit.json",
    "runs/RUNS.md",
    "runs/sc904/training_config.json",
    "runs/sc904/trainer_state.json",
    "runs/sc904/tensorboard_events.json",
    "data/SPLIT.md",
    "docs/case_study.md",
    "docs/case_study.zh.md",
]
MODEL_DELETE = ["app.py", "templates/index.html"]
SPACE_LICENCE_FILES = ["LICENSE", "USE_POLICY.md", "NOTICE", "LICENSE-MIT"]
GENERATION_CONFIG = {
    "bos_token_id": 128000,
    "eos_token_id": [128001, 128008, 128009],
    "do_sample": False,
    "max_new_tokens": 8,
    "pad_token_id": 128004,
}


def api(token: str | None):
    from huggingface_hub import HfApi

    return HfApi(token=token)


def plan_model(repo_id: str) -> list[str]:
    ops = [f"upload {f}" for f in MODEL_FILES if (ROOT / f).exists()]
    ops += [f"upload docs/figures/{p.name}" for p in sorted((ROOT / "docs" / "figures").glob("*.png"))]
    ops += ["upload generation_config.json (greedy, max_new_tokens 8)", "patch config.json use_cache=true"]
    ops += [f"delete {f}" for f in MODEL_DELETE]
    return [f"[{repo_id}] {op}" for op in ops]


def publish_model(hf, repo_id: str, dry_run: bool) -> None:
    for line in plan_model(repo_id):
        print(line)
    if dry_run:
        return
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, hf_hub_download

    ops = []
    for f in MODEL_FILES:
        if (ROOT / f).exists():
            ops.append(CommitOperationAdd(path_in_repo=f, path_or_fileobj=str(ROOT / f)))
    for p in sorted((ROOT / "docs" / "figures").glob("*.png")):
        ops.append(CommitOperationAdd(path_in_repo=f"docs/figures/{p.name}", path_or_fileobj=str(p)))
    tmp = Path(tempfile.mkdtemp())
    gen = tmp / "generation_config.json"
    gen.write_text(json.dumps(GENERATION_CONFIG, indent=2) + "\n", encoding="utf-8")
    ops.append(CommitOperationAdd(path_in_repo="generation_config.json", path_or_fileobj=str(gen)))
    cfg_path = hf_hub_download(repo_id, "config.json", token=hf.token)
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    cfg["use_cache"] = True
    cfg_new = tmp / "config.json"
    cfg_new.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    ops.append(CommitOperationAdd(path_in_repo="config.json", path_or_fileobj=str(cfg_new)))
    existing = set(hf.list_repo_files(repo_id))
    for f in MODEL_DELETE:
        if f in existing:
            ops.append(CommitOperationDelete(path_in_repo=f))
    hf.create_commit(
        repo_id=repo_id,
        operations=ops,
        commit_message="Rewrite model card as a label-leakage case study; add licences, eval results, probes; remove Flask app",  # noqa: E501
    )
    print(f"model repo updated: https://huggingface.co/{repo_id}")


def publish_dataset(hf, dry_run: bool) -> None:
    files = ["data/smart_logistics_dataset.csv", "data/train.jsonl", "data/test.jsonl", "data/SPLIT.md"]
    for f in files:
        print(f"[{DATASET_ID}] upload {f}")
    if dry_run:
        return
    from huggingface_hub import CommitOperationAdd

    hf.create_repo(DATASET_ID, repo_type="dataset", exist_ok=True)
    card = (ROOT / "data" / "DATASET_CARD.md").read_text(encoding="utf-8")
    tmp = Path(tempfile.mkdtemp())
    (tmp / "README.md").write_text(card, encoding="utf-8")
    ops = [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(tmp / "README.md"))]
    for f in files:
        ops.append(CommitOperationAdd(path_in_repo=Path(f).name, path_or_fileobj=str(ROOT / f)))
    hf.create_commit(
        repo_id=DATASET_ID, repo_type="dataset", operations=ops, commit_message="Frozen split v0 with data card"
    )
    print(f"dataset repo updated: https://huggingface.co/datasets/{DATASET_ID}")


def publish_space(hf, dry_run: bool) -> None:
    print(
        f"[{SPACE_ID}] upload space/app.py, space/requirements.txt, space/README.md, "
        f"{', '.join(SPACE_LICENCE_FILES)}, src/delaysentinel/*"
    )
    if dry_run:
        return
    hf.create_repo(SPACE_ID, repo_type="space", space_sdk="gradio", exist_ok=True)
    tmp = Path(tempfile.mkdtemp())
    for name in ("app.py", "requirements.txt", "README.md"):
        shutil.copy(ROOT / "space" / name, tmp / name)
    for name in SPACE_LICENCE_FILES:
        shutil.copy(ROOT / name, tmp / name)
    shutil.copytree(
        ROOT / "src" / "delaysentinel", tmp / "src" / "delaysentinel", ignore=shutil.ignore_patterns("__pycache__")
    )
    hf.upload_folder(repo_id=SPACE_ID, repo_type="space", folder_path=str(tmp), commit_message="Leakage demo")
    print(f"space updated: https://huggingface.co/spaces/{SPACE_ID}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rename", action="store_true")
    parser.add_argument("--model", action="store_true")
    parser.add_argument("--dataset", action="store_true")
    parser.add_argument("--space", action="store_true")
    parser.add_argument("--token", default=None, help="defaults to the locally stored token")
    args = parser.parse_args(argv)
    if not (args.rename or args.model or args.dataset or args.space):
        args.rename = args.model = args.dataset = args.space = True

    from huggingface_hub import get_token

    token = args.token or get_token()
    if not token and not args.dry_run:
        print("No Hugging Face token found. Run `hf auth login` (or `huggingface-cli login`) with a write token first.")
        return 2
    hf = api(token)
    if not args.dry_run:
        who = hf.whoami()
        print(f"authenticated as {who.get('name')}")

    repo_id = NEW_MODEL_ID
    if args.rename:
        print(f"[rename] {OLD_MODEL_ID} -> {NEW_MODEL_ID}")
        if not args.dry_run:
            existing = {m.id for m in hf.list_models(author="Yuchiwang02")}
            if NEW_MODEL_ID in existing:
                print("  already renamed")
            elif OLD_MODEL_ID in existing:
                hf.move_repo(from_id=OLD_MODEL_ID, to_id=NEW_MODEL_ID, repo_type="model")
                print("  done (the old URL now redirects)")
            else:
                print("  neither id exists under this account; aborting")
                return 1
    elif not args.dry_run:
        existing = {m.id for m in hf.list_models(author="Yuchiwang02")}
        repo_id = NEW_MODEL_ID if NEW_MODEL_ID in existing else OLD_MODEL_ID
    if args.dataset:
        publish_dataset(hf, args.dry_run)
    if args.space:
        publish_space(hf, args.dry_run)
    if args.model:
        publish_model(hf, repo_id, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
