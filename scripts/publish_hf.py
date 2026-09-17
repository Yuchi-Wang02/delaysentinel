#!/usr/bin/env python
"""Publish explicitly selected DelaySentinel artifacts to the Hub.

Requires a *write* token in the local Hugging Face credential store
(`hf auth login` / `huggingface-cli login`). Nothing here touches the model weights.

    python scripts/publish_hf.py --docs-only --model --dataset --dry-run
    python scripts/publish_hf.py --docs-only --model --dataset

No arguments prints help. A documentation update never writes model configurations,
weights, data rows or results, deletes files, renames a repository or creates a Space.
MODEL_CARD.md is the source of the Hub README; README.md is the GitHub landing page.

Explicit full-publication steps (without --docs-only)
  --rename   move Yuchiwang02/DelaySentinel to Yuchiwang02/Llama-3.2-1B-DelaySentinel (the Hub keeps a redirect)
  --dataset  create Yuchiwang02/smart-logistics-delay-split-v0 with the CSV, the frozen JSONL split and SPLIT.md
  --space    create the Gradio Space Yuchiwang02/delaysentinel-leakage-demo from space/ plus src/ and the licences
  --model    upload MODEL_CARD.md as README.md, licences, results/, runs/, docs/,
             fixed generation_config.json / config.json;
             delete app.py and templates/ (done last so every link in the card resolves when it goes live)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath

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
MODEL_DOCUMENT_MAP = [
    ("MODEL_CARD.md", "README.md"),
    *((name, name) for name in LICENCE_FILES),
    ("CHANGELOG.md", "CHANGELOG.md"),
    ("runs/RUNS.md", "runs/RUNS.md"),
    ("data/SPLIT.md", "data/SPLIT.md"),
    ("data/DATASET_CARD.md", "data/DATASET_CARD.md"),
    ("docs/case_study.md", "docs/case_study.md"),
    ("docs/case_study.zh.md", "docs/case_study.zh.md"),
    ("docs/leakage_audit.md", "docs/leakage_audit.md"),
    ("docs/evaluation.md", "docs/evaluation.md"),
    ("docs/olist_reference.md", "docs/olist_reference.md"),
    ("docs/reproduce.md", "docs/reproduce.md"),
]
MODEL_EVIDENCE_FILES = [
    "results/eval.json",
    "results/test_predictions.csv",
    "results/leakage_audit.json",
    "results/olist_positive_control.json",
    "runs/sc904/training_config.json",
    "runs/sc904/trainer_state.json",
    "runs/sc904/tensorboard_events.json",
]
DATASET_DOCUMENT_MAP = [("data/DATASET_CARD.md", "README.md"), ("data/SPLIT.md", "SPLIT.md")]
DATASET_ROW_FILES = ["data/smart_logistics_dataset.csv", "data/train.jsonl", "data/test.jsonl"]
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


def validate_manifest(manifest: list[tuple[str, str]], docs_only: bool = False) -> list[tuple[str, str]]:
    """Fail before publication if a source is missing or two files share a destination."""
    destinations = set()
    for source, destination in manifest:
        for name in (source, destination):
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
                raise ValueError(f"unsafe manifest path: {name}")
        if destination in destinations:
            raise ValueError(f"duplicate publication destination: {destination}")
        if docs_only:
            is_document = destination.endswith(".md") or destination in LICENCE_FILES
            is_figure = destination.startswith("docs/figures/") and destination.lower().endswith((".png", ".svg"))
            if destination.startswith("results/") or not (is_document or is_figure):
                raise ValueError(f"documentation mode cannot write protected artifact: {destination}")
        destinations.add(destination)
        path = (ROOT / source).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
            raise FileNotFoundError(f"required publication source is missing: {source}")
    return manifest


def model_manifest(docs_only: bool = False) -> list[tuple[str, str]]:
    manifest = list(MODEL_DOCUMENT_MAP)
    if not docs_only:
        manifest += [(name, name) for name in MODEL_EVIDENCE_FILES]
    manifest += [
        (path.relative_to(ROOT).as_posix(), path.relative_to(ROOT).as_posix())
        for path in sorted((ROOT / "docs" / "figures").iterdir())
        if path.suffix.lower() in {".png", ".svg"} and path.is_file()
    ]
    return validate_manifest(manifest, docs_only=docs_only)


def dataset_manifest(docs_only: bool = False) -> list[tuple[str, str]]:
    manifest = list(DATASET_DOCUMENT_MAP)
    if not docs_only:
        manifest += [(name, Path(name).name) for name in DATASET_ROW_FILES]
    return validate_manifest(manifest, docs_only=docs_only)


def manifest_plan(repo_id: str, manifest: list[tuple[str, str]]) -> list[str]:
    return [f"[{repo_id}] upload {source} -> {destination}" for source, destination in manifest]


def _model_plan(repo_id: str, manifest: list[tuple[str, str]], docs_only: bool) -> list[str]:
    ops = manifest_plan(repo_id, manifest)
    if not docs_only:
        ops += [
            f"[{repo_id}] upload generation_config.json (greedy, max_new_tokens 8)",
            f"[{repo_id}] patch config.json use_cache=true",
            *(f"[{repo_id}] delete {name} if present" for name in MODEL_DELETE),
        ]
    return ops


def plan_model(repo_id: str, docs_only: bool = False) -> list[str]:
    return _model_plan(repo_id, model_manifest(docs_only), docs_only)


def _add_operations(manifest: list[tuple[str, str]]):
    from huggingface_hub import CommitOperationAdd

    return [
        CommitOperationAdd(path_in_repo=destination, path_or_fileobj=str(ROOT / source))
        for source, destination in manifest
    ]


def publish_model(hf, repo_id: str, dry_run: bool, docs_only: bool = False) -> None:
    manifest = model_manifest(docs_only)
    for line in _model_plan(repo_id, manifest, docs_only):
        print(line)
    if dry_run:
        return
    ops = _add_operations(manifest)
    if docs_only:
        hf.create_commit(
            repo_id=repo_id,
            operations=ops,
            commit_message="Update model card, case studies and presentation assets",
        )
        print(f"model documentation updated: https://huggingface.co/{repo_id}")
        return
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, hf_hub_download

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


def publish_dataset(hf, dry_run: bool, docs_only: bool = False) -> None:
    manifest = dataset_manifest(docs_only)
    for line in manifest_plan(DATASET_ID, manifest):
        print(line)
    if dry_run:
        return
    if not docs_only:
        hf.create_repo(DATASET_ID, repo_type="dataset", exist_ok=True)
    ops = _add_operations(manifest)
    hf.create_commit(
        repo_id=DATASET_ID,
        repo_type="dataset",
        operations=ops,
        commit_message="Update dataset documentation" if docs_only else "Frozen split v0 with data card",
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
    parser.add_argument(
        "--docs-only", action="store_true", help="update documentation in selected existing repositories"
    )
    parser.add_argument("--rename", action="store_true")
    parser.add_argument("--model", action="store_true")
    parser.add_argument("--dataset", action="store_true")
    parser.add_argument("--space", action="store_true")
    parser.add_argument("--token", default=None, help="defaults to the locally stored token")
    args = parser.parse_args(argv)
    if not (args.rename or args.model or args.dataset or args.space):
        parser.print_help()
        return 0
    if args.docs_only and (args.rename or args.space):
        parser.error("--docs-only supports --model and/or --dataset, never --rename or --space")

    # Validate every selected manifest before credentials, network calls or the first commit.
    if args.model:
        model_manifest(args.docs_only)
    if args.dataset:
        dataset_manifest(args.docs_only)
    if args.dry_run:
        if args.rename:
            print(f"[rename] {OLD_MODEL_ID} -> {NEW_MODEL_ID}")
        if args.dataset:
            publish_dataset(None, True, args.docs_only)
        if args.space:
            publish_space(None, True)
        if args.model:
            publish_model(None, NEW_MODEL_ID, True, args.docs_only)
        return 0

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
    elif not args.docs_only:
        existing = {m.id for m in hf.list_models(author="Yuchiwang02")}
        repo_id = NEW_MODEL_ID if NEW_MODEL_ID in existing else OLD_MODEL_ID
    if args.dataset:
        publish_dataset(hf, args.dry_run, args.docs_only)
    if args.space:
        publish_space(hf, args.dry_run)
    if args.model:
        publish_model(hf, repo_id, args.dry_run, args.docs_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
