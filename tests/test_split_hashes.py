import re

from delaysentinel.data import sha256_file, sha256_normalized_newlines


def _hashes_from_split_md(root):
    text = (root / "data" / "SPLIT.md").read_text(encoding="utf-8")
    found = {}
    for line in text.splitlines():
        m = re.match(r"\|\s*`([^`]+)`\s*\|.*?`([0-9a-f]{64})`", line)
        if m:
            found[m.group(1)] = m.group(2)
    return found


def test_split_md_hashes_match_the_files(root):
    recorded = _hashes_from_split_md(root)
    assert set(recorded) == {"smart_logistics_dataset.csv", "train.jsonl", "test.jsonl"}
    for name, digest in recorded.items():
        path = root / "data" / name
        assert sha256_normalized_newlines(path) == digest, name
        # on an LF checkout the raw hash is the same; on a CRLF checkout only the normalized one is
        raw = sha256_file(path)
        with open(path, "rb") as handle:
            has_crlf = b"\r\n" in handle.read()
        assert has_crlf or raw == digest, name
