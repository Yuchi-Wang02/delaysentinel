import importlib.util
import re


def _load_publish(root):
    spec = importlib.util.spec_from_file_location("publish_hf", root / "scripts" / "publish_hf.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_notice_matches_license_clause_verbatim(root):
    licence = (root / "LICENSE").read_text(encoding="utf-8")
    m = re.search(r"“(Llama 3\.2 is licensed under[^”]+)”", licence)
    assert m, "attribution sentence not found in LICENSE"
    required = re.sub(r"\s+", " ", m.group(1)).strip()
    notice_first_line = (root / "NOTICE").read_text(encoding="utf-8").splitlines()[0].strip()
    assert notice_first_line == required


def test_built_with_llama_and_name(root):
    for rel in ("README.md", "MODEL_CARD.md", "space/README.md", "space/app.py"):
        assert "Built with Llama" in (root / rel).read_text(encoding="utf-8"), rel
    assert "Llama-3.2-1B-DelaySentinel" in (root / "README.md").read_text(encoding="utf-8")
    readme = (root / "MODEL_CARD.md").read_text(encoding="utf-8")
    title = next(line for line in readme.splitlines() if line.startswith("# "))
    assert title.startswith("# Llama")


def test_licence_files_present_and_published(root):
    for rel in (
        "LICENSE",
        "USE_POLICY.md",
        "NOTICE",
        "THIRD_PARTY_LICENSES.md",
        "LICENSE-MIT",
        "LICENSES/Apache-2.0.txt",
    ):
        assert (root / rel).exists(), rel
    assert "LLAMA 3.2 COMMUNITY LICENSE AGREEMENT" in (root / "LICENSE").read_text(encoding="utf-8")
    assert "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION" in (
        root / "LICENSES" / "Apache-2.0.txt"
    ).read_text(encoding="utf-8")
    publish = _load_publish(root)
    manifest = publish.model_manifest()
    published = {destination for _, destination in manifest}
    assert {"LICENSE", "NOTICE", "USE_POLICY.md", "THIRD_PARTY_LICENSES.md", "LICENSE-MIT"} <= published
    assert {"LICENSE", "NOTICE", "USE_POLICY.md"} <= set(publish.SPACE_LICENCE_FILES)
    for source, _ in manifest:
        assert (root / source).is_file(), source


def test_train_script_header(root):
    lines = (root / "scripts" / "train.py").read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("#!/usr/bin/env python3")
    assert "acon96/home-llm" in "\n".join(lines[:20])
