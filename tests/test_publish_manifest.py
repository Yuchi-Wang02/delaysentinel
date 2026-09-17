"""Resolve links at their publication locations, including the independently sourced Hub card."""

import importlib.util
import posixpath
import re
from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit

import pytest

LINK_RE = re.compile(r"\]\(<?([^\s)>]+)>?(?:\s+\"[^\"]*\")?\)")


def _publish_module(root):
    spec = importlib.util.spec_from_file_location("publish_hf", root / "scripts" / "publish_hf.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _local_links(text, location):
    for match in LINK_RE.finditer(text):
        url = urlsplit(match.group(1))
        if url.scheme or url.netloc or not url.path:
            continue
        yield posixpath.normpath(posixpath.join(posixpath.dirname(location), unquote(url.path)))


def _relative_extensionless_files(text, location, published_files):
    # HF renders an extensionless relative link as a tree URL even when its target is a file.
    # Exact manifest membership distinguishes files from valid links to published directories.
    return sorted(
        target
        for target in _local_links(text, location)
        if target in published_files and not PurePosixPath(target).suffix
    )


@pytest.mark.parametrize("kind", ["model", "dataset"])
def test_hub_extensionless_files_have_explicit_urls(root, kind):
    module = _publish_module(root)
    manifest = module.model_manifest() if kind == "model" else module.dataset_manifest()
    published_files = {destination for _, destination in manifest}
    for source, destination in manifest:
        if not source.endswith(".md"):
            continue
        ambiguous = _relative_extensionless_files(
            (root / source).read_text(encoding="utf-8"), destination, published_files
        )
        assert not ambiguous, (
            f"{source} at Hub {destination} has extensionless relative file links: {ambiguous}; "
            "use an explicit https://huggingface.co/.../blob/<revision>/... or /resolve/<revision>/... URL"
        )


@pytest.mark.parametrize(
    "text, location, expected",
    [
        ("[license](LICENSE)", "README.md", ["LICENSE"]),
        ("[notice](../NOTICE#attribution)", "docs/evaluation.md", ["NOTICE"]),
        ("[results](results/)", "README.md", []),
        ("[license](https://huggingface.co/user/model/blob/main/LICENSE)", "README.md", []),
        ("[license](https://huggingface.co/user/model/resolve/main/LICENSE)", "README.md", []),
    ],
)
def test_extensionless_file_detection_preserves_directories_and_explicit_urls(text, location, expected):
    published_files = {"LICENSE", "NOTICE", "results/eval.json"}
    assert _relative_extensionless_files(text, location, published_files) == expected


@pytest.mark.parametrize("kind", ["model", "dataset"])
def test_linked_files_are_published(root, kind):
    module = _publish_module(root)
    manifest = module.model_manifest() if kind == "model" else module.dataset_manifest()
    published = {destination for _, destination in manifest}
    checked = []
    for source, destination in manifest:
        if not source.endswith(".md"):
            continue
        text = (root / source).read_text(encoding="utf-8")
        links = set(_local_links(text, destination))
        missing = sorted(
            target
            for target in links
            if target not in published and not any(p.startswith(target.rstrip("/") + "/") for p in published)
        )
        assert not missing, f"{source} at Hub {destination} links to unpublished paths: {missing}"
        checked.append(source)
    assert checked, "the publication manifest must contain documentation"


def test_github_landing_page_links_exist(root):
    text = (root / "README.md").read_text(encoding="utf-8")
    links = set(_local_links(text, "README.md"))
    assert links, "the landing page should link to project artifacts"
    assert not [target for target in links if not (root / target).exists()]


def test_card_maps_to_hub_readme_and_not_github_readme(root):
    module = _publish_module(root)
    for docs_only in (False, True):
        manifest = module.model_manifest(docs_only)
        assert [(source, destination) for source, destination in manifest if destination == "README.md"] == [
            ("MODEL_CARD.md", "README.md")
        ]
        assert all(source != "README.md" for source, _ in manifest)


def test_missing_source_fails_instead_of_being_skipped(root, tmp_path, monkeypatch):
    module = _publish_module(root)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match=r"MODEL_CARD\.md"):
        module.validate_manifest([("MODEL_CARD.md", "README.md")])


def test_duplicate_destinations_are_rejected(root, tmp_path, monkeypatch):
    module = _publish_module(root)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / "b.md").write_text("b", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate publication destination"):
        module.validate_manifest([("a.md", "README.md"), ("b.md", "README.md")])


@pytest.mark.parametrize("path", ["../outside.md", "/absolute.md", "C:/outside.md", "docs\\outside.md"])
def test_manifest_cannot_escape_repository(root, path):
    module = _publish_module(root)
    with pytest.raises(ValueError, match="unsafe manifest path"):
        module.validate_manifest([(path, "README.md")])


def test_link_resolution_uses_remote_location():
    assert list(_local_links("[results](../results/eval.json#key)", "docs/evaluation.md")) == ["results/eval.json"]
    assert list(_local_links("![figure](docs/figures/overview.svg)", "README.md")) == ["docs/figures/overview.svg"]
