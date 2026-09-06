"""Every in-repository file the card and case studies link to must also be uploaded.

The Hub copy of the card is the same file as README.md, so a link to `results/…` or
`docs/…` that is not in the publish manifest renders as a dead link for every reader who
arrives from the Hub rather than from GitHub.
"""

import importlib.util
import re

import pytest

LINK_RE = re.compile(r"\]\((?:\.\./)?((?:results|docs|runs|data|src|scripts|space)/[^)#\s]+)\)")
DOCS = ["README.md", "docs/case_study.md", "docs/case_study.zh.md"]


def _publish_module(root):
    spec = importlib.util.spec_from_file_location("publish_hf", root / "scripts" / "publish_hf.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("doc", DOCS)
def test_linked_files_are_published(root, doc):
    module = _publish_module(root)
    # the figures are added by a glob in the script rather than listed in MODEL_FILES
    published = set(module.MODEL_FILES) | {
        f"docs/figures/{path.name}" for path in (root / "docs" / "figures").glob("*.png")
    }
    text = (root / doc).read_text(encoding="utf-8")
    linked = {m.group(1) for m in LINK_RE.finditer(text)}
    assert linked, f"{doc} links to no repository file; the regex is probably stale"
    missing = sorted(target for target in linked if target not in published)
    assert not missing, f"{doc} links to files that publish_hf.py does not upload: {missing}"


def test_every_published_file_exists(root):
    module = _publish_module(root)
    missing = sorted(f for f in module.MODEL_FILES if not (root / f).exists())
    assert not missing, f"publish_hf.py would skip files that do not exist: {missing}"
