"""Exercise actual documentation commit operations without network access or Hub credentials."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def publisher(root, tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("publish_hf", root / "scripts" / "publish_hf.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    for source, _ in module.MODEL_DOCUMENT_MAP + module.DATASET_DOCUMENT_MAP:
        path = tmp_path / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"document source: {source}\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("GitHub landing page: never upload as the model card", encoding="utf-8")
    figures = tmp_path / "docs" / "figures"
    figures.mkdir()
    (figures / "example.svg").write_text("<svg/>", encoding="utf-8")
    (figures / "example.png").write_bytes(b"test image")
    (figures / "unrelated.json").write_text("{}", encoding="utf-8")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(CommitOperationAdd=lambda **kwargs: SimpleNamespace(**kwargs), get_token=lambda: "test-token"),
    )
    return module


class CommitOnlyClient:
    """Any create_repo, download, delete, rename or Space operation fails as an absent method."""

    def __init__(self):
        self.commits = []

    def whoami(self):
        return {"name": "test-user"}

    def create_commit(self, **kwargs):
        self.commits.append(kwargs)


def test_model_docs_commit_uses_card_and_never_writes_evidence_or_configs(publisher, capsys):
    client = CommitOnlyClient()
    publisher.publish_model(client, publisher.NEW_MODEL_ID, dry_run=False, docs_only=True)
    assert len(client.commits) == 1
    commit = client.commits[0]
    assert commit["repo_id"] == publisher.NEW_MODEL_ID
    operations = commit["operations"]
    targets = {op.path_in_repo for op in operations}
    card = next(op for op in operations if op.path_in_repo == "README.md")
    assert Path(card.path_or_fileobj).read_text(encoding="utf-8") == "document source: MODEL_CARD.md\n"
    assert {"docs/figures/example.svg", "docs/figures/example.png"} <= targets
    assert "docs/figures/unrelated.json" not in targets
    assert not targets.intersection({"config.json", "generation_config.json", "model.safetensors", "app.py"})
    assert not any(p.startswith("results/") or p.endswith((".json", ".jsonl", ".csv", ".bin")) for p in targets)
    planned = publisher.plan_model(publisher.NEW_MODEL_ID, docs_only=True)
    output = capsys.readouterr().out
    assert all(line in output for line in planned)
    assert len(planned) == len(operations)


def test_dataset_docs_commit_contains_no_data_rows(publisher):
    client = CommitOnlyClient()
    publisher.publish_dataset(client, dry_run=False, docs_only=True)
    assert len(client.commits) == 1
    commit = client.commits[0]
    assert commit["repo_type"] == "dataset"
    assert {op.path_in_repo for op in commit["operations"]} == {"README.md", "SPLIT.md"}
    card = next(op for op in commit["operations"] if op.path_in_repo == "README.md")
    assert Path(card.path_or_fileobj).name == "DATASET_CARD.md"


def test_dry_run_does_not_load_hub_library_or_credentials(publisher, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    assert publisher.main(["--docs-only", "--model", "--dataset", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "MODEL_CARD.md -> README.md" in output
    assert "data/DATASET_CARD.md -> README.md" in output
    assert "config.json" not in output and "delete " not in output and "[rename]" not in output


def test_no_arguments_only_print_help(publisher, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    assert publisher.main([]) == 0
    assert "usage:" in capsys.readouterr().out


@pytest.mark.parametrize("flag", ["--rename", "--space"])
def test_docs_mode_rejects_other_mutations(publisher, flag):
    with pytest.raises(SystemExit) as exc:
        publisher.main(["--docs-only", "--model", flag])
    assert exc.value.code == 2


def test_missing_selected_source_prevents_all_commits(publisher, monkeypatch):
    (publisher.ROOT / "MODEL_CARD.md").unlink()
    client = CommitOnlyClient()
    monkeypatch.setattr(publisher, "api", lambda token: client)
    with pytest.raises(FileNotFoundError, match=r"MODEL_CARD\.md"):
        publisher.main(["--docs-only", "--model", "--dataset"])
    assert client.commits == []


def test_cli_docs_mode_only_commits_to_selected_existing_repositories(publisher, monkeypatch):
    client = CommitOnlyClient()
    monkeypatch.setattr(publisher, "api", lambda token: client)
    assert publisher.main(["--docs-only", "--model", "--dataset"]) == 0
    assert {commit["repo_id"] for commit in client.commits} == {publisher.NEW_MODEL_ID, publisher.DATASET_ID}
    assert len(client.commits) == 2


@pytest.mark.parametrize("target", ["config.json", "model.safetensors", "results/eval.json", "data/test.jsonl"])
def test_docs_mode_rejects_accidental_protected_manifest_entries(publisher, target):
    # A future edit of the manifest must not silently expand a documentation update.
    publisher.MODEL_DOCUMENT_MAP.append(("MODEL_CARD.md", target))
    with pytest.raises(ValueError, match="documentation mode cannot write protected artifact"):
        publisher.model_manifest(docs_only=True)
