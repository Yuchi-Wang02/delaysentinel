"""The prose claims about the probes must agree with what results/eval.json holds.

scripts/check_card_numbers.py is a presence check. It cannot notice that a cell reads
"11 of 12 ... none fires", which contradicts itself, or that a value the JSON tags
`synonym` is listed among the ones that do not fire. Both mistakes were shipped once.
These assertions derive the counts from the JSON and pin the sentences that state them.
"""

import json
import re

import pytest

DETAIL_DOC = "docs/evaluation.md"
DOCS = [DETAIL_DOC]
PUBLIC_DOCS = [
    "README.md",
    "MODEL_CARD.md",
    DETAIL_DOC,
    "docs/olist_reference.md",
    "docs/case_study.md",
    "docs/case_study.zh.md",
]


def _flat(text):
    """One line, so an assertion is not defeated by where a sentence happens to wrap."""
    return re.sub(r"\s+", " ", text).lower()


@pytest.fixture
def probes(root):
    path = root / "results" / "eval.json"
    assert path.is_file(), "the committed probe evidence is required"
    return json.loads(path.read_text(encoding="utf-8"))["robustness_probe"]


def _control(probes, field):
    return {
        k.rsplit("_", 1)[1]: tuple(sorted(v["pred_counts"].items()))
        for k, v in probes.items()
        if k.startswith(f"control_word_{field}_")
    }


def test_control_words_sit_at_the_baseline_in_shipment_status(root, probes):
    counts = _control(probes, "Shipment_Status")
    assert len(counts) == 12
    assert len(set(counts.values())) == 1, f"not all 12 are identical: {counts}"
    text = _flat((root / DETAIL_DOC).read_text(encoding="utf-8"))
    assert "all 12 give exactly the 23 baseline rows" in text
    assert "11 of 12 give exactly" not in text, "the Shipment_Status cell claims an exception the JSON does not have"


def test_meadow_is_the_only_control_word_exception(root, probes):
    counts = _control(probes, "Traffic_Status")
    assert len(counts) == 12
    common = max(set(counts.values()), key=list(counts.values()).count)
    exceptions = sorted(word for word, c in counts.items() if c != common)
    assert exceptions == ["Meadow"], f"the documents name Meadow as the only exception, but the JSON has {exceptions}"


def test_documents_do_not_call_a_firing_value_a_non_firing_synonym(root, probes):
    fires = {}
    for key, probe in probes.items():
        if not key.startswith(("value_Delayed_to_", "value_Heavy_to_")):
            continue
        kind = (probe.get("meta") or {}).get("variant_kind")
        fires.setdefault(kind, []).append((key.split("_to_")[1], probe["pred_pos_rate"] == 1.0))
    synonyms = fires["synonym"]
    firing = sorted(name for name, f in synonyms if f)
    assert firing == ["Late"], f"expected only Late to fire among the synonyms, got {firing}"
    assert len(synonyms) == 10, "five synonyms per clause"
    for doc in DOCS:
        text = _flat((root / doc).read_text(encoding="utf-8"))
        assert "four of the five synonyms" in text, f"{doc} must not claim all ten synonyms leave the answer alone"


def test_every_non_firing_rewrite_is_listed(root, probes):
    quiet = sorted(
        key.split("_to_")[1].replace("_", " ")
        for key, probe in probes.items()
        if key.startswith("value_Delayed_to_") and probe["pred_pos_rate"] < 1.0
    )
    assert quiet, "expected some Delayed rewrites to leave the answer at the baseline"
    for doc in DOCS:
        text = _flat((root / doc).read_text(encoding="utf-8"))
        missing = [word for word in quiet if word.lower() not in text]
        assert not missing, f"{doc} omits non-firing rewrites: {missing}"


# The published claim "the weights match the strings `Delay` and `Heavy`" was wrong: three
# rewrites fire without containing either string. It reached the card, both case studies and the
# changelog before it was caught, so it gets a test.
TRIGGER_SUBSTRING = {"Delayed": "delay", "Heavy": "heav"}
STRING_MATCH_PHRASES = (
    "string match",
    "matches the strings",
    "match on the strings",
    "string matcher",
    "字符串匹配",
    "两个字符串的匹配",
    "匹配两个字符串",
)
NEGATIONS = ("not", "n't", "refute", "wrong", "不是", "并非", "推翻")


def test_values_fire_without_containing_the_trigger(probes):
    """The counterexamples that make the string-match claim false must still be in the data."""
    counterexamples = []
    for clause, needle in TRIGGER_SUBSTRING.items():
        prefix = f"value_{clause}_to_"
        for key, probe in probes.items():
            if key.startswith(prefix) and probe["pred_pos_rate"] == 1.0:
                value = key[len(prefix) :].replace("_", " ")
                if needle not in value.lower():
                    counterexamples.append(f"{clause}->{value}")
    assert sorted(counterexamples) == ["Delayed->Early", "Delayed->Late", "Heavy->Light"], (
        f"the set of firing rewrites that contain neither trigger has changed: {sorted(counterexamples)}. "
        "The documents' claim that the pattern is not a string match rests on exactly this set."
    )


@pytest.mark.parametrize("doc", PUBLIC_DOCS)
def test_no_public_document_asserts_a_string_match(root, doc):
    text = _flat((root / doc).read_text(encoding="utf-8"))
    for phrase in STRING_MATCH_PHRASES:
        start = 0
        while (i := text.find(phrase, start)) != -1:
            before = text[max(0, i - 90) : i]
            assert any(n in before for n in NEGATIONS), (
                f"{doc} asserts a {phrase!r} without negating it: ...{text[max(0, i - 90) : i + 60]}... "
                "Early, Late and Light fire and contain neither trigger."
            )
            start = i + 1
