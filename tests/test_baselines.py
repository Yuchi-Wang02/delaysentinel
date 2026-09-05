from delaysentinel.baselines import BASELINE_NAMES, evaluate_split


def test_frozen_split_baselines(train_frame, test_frame):
    res = evaluate_split(train_frame, test_frame, seed=0, n_boot=50)
    assert set(BASELINE_NAMES) <= set(res)
    assert res["rule_delayed_or_heavy"]["acc"] == 1.0
    assert res["decision_tree_depth2"]["acc"] == 1.0
    assert res["logistic_regression"]["acc"] == 1.0
    assert res["all_positive"]["acc"] == 0.58
    assert res["gradient_boosting_without_rule_fields"]["acc"] < 0.7
    assert (
        "Shipment_Status" in res["decision_tree_depth2"]["tree_rules"]
        or "Traffic_Status" in res["decision_tree_depth2"]["tree_rules"]
    )
