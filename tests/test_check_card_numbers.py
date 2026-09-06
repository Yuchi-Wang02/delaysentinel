import importlib.util


def _load(root):
    spec = importlib.util.spec_from_file_location("check_card_numbers", root / "scripts" / "check_card_numbers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_docs_pass(root):
    mod = _load(root)
    assert mod.check() == []


def test_a_wrong_metric_is_caught(root, tmp_path):
    mod = _load(root)
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "0.452" in readme
    bad = tmp_path / "README.md"
    bad.write_text(readme.replace("0.452", "0.9137", 1), encoding="utf-8")
    missing = mod.check(docs=[bad])
    assert any(token == "0.9137" for _, token, _ in missing)


def test_known_set_is_not_inflated(root):
    mod = _load(root)
    known = mod.known_numbers() | mod.allowlist()
    # values that only exist as unrelated training-config keys or as x100 expansions must not pass
    for token in ("0.999", "1e-08", "1800", "0.9137", "58"):
        assert not (mod.normalise(token) & known), token
