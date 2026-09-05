import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from delaysentinel.data import frame_from_jsonl, load_csv, load_jsonl  # noqa: E402


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def csv_frame():
    return load_csv(ROOT / "data" / "smart_logistics_dataset.csv")


@pytest.fixture(scope="session")
def train_rows():
    return load_jsonl(ROOT / "data" / "train.jsonl")


@pytest.fixture(scope="session")
def test_rows():
    return load_jsonl(ROOT / "data" / "test.jsonl")


@pytest.fixture(scope="session")
def train_frame(train_rows):
    return frame_from_jsonl(train_rows)


@pytest.fixture(scope="session")
def test_frame(test_rows):
    return frame_from_jsonl(test_rows)
