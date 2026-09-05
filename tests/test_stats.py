import numpy as np

from delaysentinel.stats import bootstrap_ci, classification_metrics, clopper_pearson_ci, confusion, wilson_ci


def test_intervals_at_perfect_accuracy():
    lo, hi = wilson_ci(200, 200)
    assert 0.98 < lo < 0.99 and hi == 1.0
    lo, hi = clopper_pearson_ci(200, 200)
    assert 0.98 < lo < 0.99 and hi == 1.0
    assert clopper_pearson_ci(0, 10)[0] == 0.0


def test_confusion_order():
    assert confusion([0, 0, 1, 1], [0, 1, 0, 1]) == [[1, 1], [1, 1]]


def test_metrics_degenerate_and_bootstrap():
    y = np.array([0] * 84 + [1] * 116)
    m = classification_metrics(y, y)
    assert m["acc"] == 1.0 and m["errors"] == 0
    assert m["f1_ci_bootstrap"] == "degenerate (zero errors)"
    assert m["confusion"] == [[84, 0], [0, 116]]
    p = y.copy()
    p[:10] = 1
    m2 = classification_metrics(y, p, prob=np.where(p == 1, 0.9, 0.1), n_boot=200)
    assert isinstance(m2["f1_ci_bootstrap"], list) and m2["errors"] == 10
    assert "auroc" in m2 and "auroc_note" in m2


def test_bootstrap_ci_shape():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 100)
    p = rng.random(100)
    lo, hi = bootstrap_ci(lambda a, b: float((b > 0.5).astype(int).__eq__(a).mean()), y, p, n_boot=100)
    assert 0 <= lo <= hi <= 1
