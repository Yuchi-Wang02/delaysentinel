import pandas as pd

from delaysentinel.leakage_audit import audit, to_markdown


def test_audit_finds_the_two_clause_rule(root):
    frame = pd.read_csv(root / "data" / "smart_logistics_dataset.csv", keep_default_na=False, na_values=[""])
    report = audit(frame, "Logistics_Delay")
    pure = {c["condition"] for c in report["pure_conditions"]}
    assert 'Shipment_Status == "Delayed"' in pure
    assert 'Traffic_Status == "Heavy"' in pure
    g = report["greedy_or_rule"]
    assert g["mismatches"] == 0 and g["n_conditions"] == 2 and g["accuracy"] == 1.0
    assert report["depth2_tree"]["cv_accuracy_mean"] == 1.0
    assert "Greedy OR-rule" in to_markdown(report)
