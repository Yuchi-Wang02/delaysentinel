import importlib.util

import pytest


def _load(root):
    spec = importlib.util.spec_from_file_location("check_card_numbers", root / "scripts" / "check_card_numbers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_docs_pass(root):
    mod = _load(root)
    assert mod.check() == []


@pytest.mark.parametrize(
    "original, wrong",
    [
        ("0.452", "0.9137"),  # an invented value
        ("0.452", "0.443"),  # a plausible near-miss on a real metric
        ("1.000", "0.998"),  # a plausible near-miss on a perfect score
    ],
)
def test_a_wrong_metric_is_caught(root, tmp_path, original, wrong):
    mod = _load(root)
    readme = (root / "README.md").read_text(encoding="utf-8")
    if original not in readme:
        pytest.skip(f"{original} no longer appears in README.md")
    bad = tmp_path / "README.md"
    bad.write_text(readme.replace(original, wrong, 1), encoding="utf-8")
    missing = mod.check(docs=[bad])
    assert any(token == wrong for _, token, _ in missing), f"{wrong} slipped through"


def test_known_set_is_not_inflated(root):
    """The check is a presence check, so it cannot catch a wrong number that happens to equal some
    other value in the files (0.5, for instance, really is a score threshold in the Olist sweep).
    What it must catch is a value that appears nowhere."""
    mod = _load(root)
    known = mod.known_numbers() | mod.allowlist()
    for token in ("0.999", "1e-08", "1800", "0.9137", "0.443", "0.4521"):
        assert not (mod.normalise(token) & known), token


def test_scientific_notation_is_tokenized_and_checked(root):
    mod = _load(root)
    known = mod.known_numbers() | mod.allowlist()
    assert mod.NUM_RE.findall("the learning rate is 8.3e-07 at the last step") == ["8.3e-07"]
    assert mod.normalise("8.3e-07") & known, "the last logged learning rate should be backed"
    assert not (mod.normalise("9.9e-07") & known), "an invented exponent must not pass"


def test_derived_counts_match_the_repository(root):
    mod = _load(root)
    counts = mod.derived_counts()
    assert counts["test_functions"] >= 30
    if (root / "results" / "eval.json").exists():
        import json

        data = json.loads((root / "results" / "eval.json").read_text(encoding="utf-8"))
        if "probe_set_counts" in data:
            assert counts["probe_sets_total"] == data["probe_set_counts"]["total"]
