"""The three licence statements must not drift apart.

results/olist_positive_control.json is a derivative of a CC BY-NC-SA 4.0 dataset, so it is
the one generated file the author cannot release under MIT. THIRD_PARTY_LICENSES.md said so
while LICENSE-MIT still granted MIT over all of results/, which is the kind of contradiction
only a test catches.
"""

CARVE_OUT = "olist_positive_control.json"


def _positions(text, needle):
    start, out = 0, []
    while (i := text.find(needle, start)) != -1:
        out.append(i)
        start = i + 1
    return out


def test_the_olist_carve_out_appears_in_both_licence_documents(root):
    for name in ("LICENSE-MIT", "THIRD_PARTY_LICENSES.md"):
        text = (root / name).read_text(encoding="utf-8")
        assert CARVE_OUT in text, f"{name} does not carve results/{CARVE_OUT} out of the MIT grant"
        # at least one mention must sit next to the reason, though others may be bare table cells
        windows = [text[max(0, m - 400) : m + 400] for m in _positions(text, CARVE_OUT)]
        assert any("CC BY-NC-SA" in w or "THIRD_PARTY_LICENSES" in w for w in windows), (
            f"{name} names the file without saying why it is carved out"
        )


def test_the_summary_table_agrees_with_the_body(root):
    text = (root / "THIRD_PARTY_LICENSES.md").read_text(encoding="utf-8")
    summary = next(line for line in text.splitlines() if line.startswith("| author's code"))
    assert CARVE_OUT in summary, "the summary row still grants MIT over every generated result file"
