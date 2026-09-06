from delaysentinel.data import user_text
from delaysentinel.probes import (
    DELAYED_VARIANTS,
    HEAVY_VARIANTS,
    NON_RULE_EDITS,
    _literal_rule_on_texts,
    _remaining_clause_on_texts,
    counterfactual_probes,
    ood_probes,
    robustness_probes,
    summarize_counterfactuals,
)


def _by_key(probes):
    return {p.key: p for p in probes}


def test_counterfactual_probe_sizes(test_frame, test_rows):
    users = [user_text(r) for r in test_rows]
    probes = _by_key(counterfactual_probes(test_frame, users))
    assert len(probes["status_Delayed_to_InTransit"].texts) == 50
    assert len(probes["traffic_Heavy_to_Clear"].texts) == 43
    assert len(probes["negatives_traffic_to_Heavy"].texts) == 84
    assert len(probes["negatives_status_to_Delayed"].texts) == 84
    assert len(probes["negatives_waiting60_temp30"].texts) == 84
    assert len(probes["positives_waiting10_temp18"].texts) == 116
    assert len(probes["both_rule_fields_removed"].texts) == 200
    assert all(t.count("\n") == 12 for t in probes["both_rule_fields_removed"].texts)
    for fld in NON_RULE_EDITS:
        assert len(probes[f"nonrule_negatives_{fld}"].texts) == 84
        assert len(probes[f"nonrule_positives_{fld}"].texts) == 116
    # each edited text differs from its source only in the edited lines (a row whose value
    # already equals the new value legitimately differs in fewer lines)
    for key, fields in (
        ("status_Delayed_to_InTransit", {"Shipment_Status"}),
        ("negatives_waiting60_temp30", {"Waiting_Time", "Temperature"}),
        ("nonrule_positives_Asset_ID", {"Asset_ID"}),
    ):
        p = probes[key]
        for i, t in zip(p.indices, p.texts, strict=False):
            diff = [a.split(":", 1)[0] for a, b in zip(users[i].split("\n"), t.split("\n"), strict=False) if a != b]
            assert len(diff) <= len(fields) and set(diff) <= fields
    assert set(_literal_rule_on_texts(probes["negatives_traffic_to_Heavy"].texts)) == {1}
    assert set(_literal_rule_on_texts(probes["status_Delayed_to_InTransit"].texts)) == {0}


def test_counterfactual_summary_counts(test_frame, test_rows):
    users = [user_text(r) for r in test_rows]
    cf = counterfactual_probes(test_frame, users)
    fake = {}
    for p in cf:
        fake[p.key] = {"n": len(p.texts), "matches_expected": len(p.texts)}
    s = summarize_counterfactuals(fake, cf)
    assert s["rule_field_edits"] == 50 + 43 + 84 + 84 == 261
    assert s["distinct_rows_with_rule_field_edit"] == 177
    assert s["non_rule_fields_probed_count"] == 13
    assert s["non_rule_field_edits_that_changed_the_prediction"] == 0


def test_robustness_probe_construction(test_frame, test_rows):
    users = [user_text(r) for r in test_rows]
    probes = _by_key(robustness_probes(test_frame, users))
    assert len(probes["rename_columns_both"].texts) == 200
    assert all(t.count("\n") == 14 for t in probes["rename_columns_both"].texts)
    assert all("Shipment_Status" not in t and "Status:" in t for t in probes["rename_columns_both"].texts)
    for suffix, value, _kind in DELAYED_VARIANTS:
        p = probes[f"value_Delayed_to_{suffix}"]
        assert len(p.texts) == 73 and all(f"Shipment_Status: {value}" in t for t in p.texts)
    for suffix, value, _kind in HEAVY_VARIANTS:
        p = probes[f"value_Heavy_to_{suffix}"]
        assert len(p.texts) == 66 and all(f"Traffic_Status: {value}" in t for t in p.texts)
    assert len(probes["shuffle_column_order_seed0"].texts) == 200
    assert all(t.count("\n") == 13 for t in probes["only_status_removed"].texts)
    assert len(probes["trigger_swapped_fields"].texts) == 84
    assert all(
        "Shipment_Status: Heavy" in t and "Traffic_Status: Delayed" in t for t in probes["trigger_swapped_fields"].texts
    )
    # literal rule is undefined once a rule column is renamed; the remaining clause is defined when one is deleted
    assert set(_literal_rule_on_texts(probes["rename_status_only"].texts)) == {None}
    remaining = _remaining_clause_on_texts(probes["only_status_removed"].texts)
    assert None not in remaining and sum(remaining) == 66


def test_delayed_rows_in_test_split(test_frame):
    # 50 Delayed-only + 43 Heavy-only + 23 both = 116 positives
    assert int((test_frame["Shipment_Status"] == "Delayed").sum()) == 73
    assert int((test_frame["Traffic_Status"] == "Heavy").sum()) == 66
    both = int(((test_frame["Shipment_Status"] == "Delayed") & (test_frame["Traffic_Status"] == "Heavy")).sum())
    assert both == 23 and 73 + 66 - both == 116


def test_ood_probes_have_texts():
    for p in ood_probes():
        assert p.texts and p.indices == []
