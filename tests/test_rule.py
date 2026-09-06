from delaysentinel.data import category_counts, rule_crosstab, rule_predict


def test_rule_reproduces_every_label(csv_frame):
    assert len(csv_frame) == 1000
    assert (rule_predict(csv_frame) == csv_frame["Logistics_Delay"].to_numpy()).all()


def test_crosstab_numbers(csv_frame):
    ct = rule_crosstab(csv_frame)
    assert ct["shipment_status_delayed"] == {"rows": 350, "delayed": 350}
    assert ct["traffic_status_heavy"] == {"rows": 327, "delayed": 327}
    assert ct["neither"] == {"rows": 434, "delayed": 0}
    assert ct["delivered_and_heavy"] == {"rows": 118, "delayed": 118}
    assert ct["both"]["rows"] == 111
    assert ct["mismatches"] == 0
    assert len(ct["status_x_traffic_cells"]) == 9


def test_category_counts(csv_frame):
    counts = category_counts(csv_frame)
    assert counts["Shipment_Status"] == {"Delayed": 350, "Delivered": 338, "In Transit": 312}
    assert counts["Traffic_Status"] == {"Detour": 345, "Clear": 328, "Heavy": 327}
    assert counts["Logistics_Delay_Reason"]["None"] == 263


def test_frozen_split_sizes(train_frame, test_frame):
    assert len(train_frame) == 800 and len(test_frame) == 200
    assert int(test_frame["gold"].sum()) == 116
    assert (rule_predict(test_frame) == test_frame["gold"].to_numpy()).all()
    assert (rule_predict(train_frame) == train_frame["gold"].to_numpy()).all()
