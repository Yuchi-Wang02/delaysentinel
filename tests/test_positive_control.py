import numpy as np

from delaysentinel.positive_control import _haversine, _reliability, _threshold_sweep


def test_haversine_known_distance():
    # Sao Paulo (-23.55, -46.63) to Rio de Janeiro (-22.91, -43.17): about 360 km
    d = _haversine(np.array([-23.55]), np.array([-46.63]), np.array([-22.91]), np.array([-43.17]))
    assert 340 < float(d[0]) < 380


def test_threshold_sweep_and_reliability_shapes():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    prob = np.clip(y * 0.6 + rng.random(500) * 0.4, 0, 1)
    sweep = _threshold_sweep(y, prob)
    assert [row["threshold"] for row in sweep] == [0.05, 0.1, 0.2, 0.3, 0.5]
    assert all(0 <= row["recall_of_late_orders"] <= 1 for row in sweep)
    rel = _reliability(y, prob)
    assert sum(r["n"] for r in rel) == 500
