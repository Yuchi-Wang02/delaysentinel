import math

import pytest

from delaysentinel.prompting import (
    FIELDS,
    SYSTEM_PROMPT,
    build_messages,
    drop_fields,
    parse_label,
    parse_user_text,
    rename_field,
    reorder_fields,
    serialize_row,
    set_field,
    shuffled_order,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Logistics_Delay: 1", 1),
        ("Logistics_Delay:0", 0),
        ("  Logistics_Delay: 1\n", 1),
        ("Logistics", None),
        ("Logistics_Delay: 2", None),
        ("Logistics_Delay: 10", None),
        ("assistant\nLogistics_Delay: 1", None),
        ("The answer is 1", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_label_is_anchored(text, expected):
    assert parse_label(text) == expected


def test_serialize_round_trip_and_missing():
    row = {f: i for i, f in enumerate(FIELDS)}
    row["Logistics_Delay_Reason"] = math.nan
    row["Temperature"] = " 27.0 "
    text = serialize_row(row)
    assert text.count("\n") == len(FIELDS) - 1
    parsed = parse_user_text(text)
    assert parsed["Logistics_Delay_Reason"] == "None"
    assert parsed["Temperature"] == "27.0"
    assert list(parsed) == list(FIELDS)


def test_build_messages_uses_training_system_prompt():
    msgs = build_messages("x")
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert msgs[1]["role"] == "user"


def test_edits_change_exactly_one_line(test_rows):
    text = test_rows[0]["conversations"][1]["value"]
    edited = set_field(text, "Traffic_Status", "Heavy")
    diff = [(a, b) for a, b in zip(text.split("\n"), edited.split("\n"), strict=False) if a != b]
    assert diff == [("Traffic_Status: Detour", "Traffic_Status: Heavy")]
    with pytest.raises(KeyError):
        set_field(text, "Nope", 1)


def test_drop_rename_reorder(test_rows):
    text = test_rows[0]["conversations"][1]["value"]
    dropped = drop_fields(text, ["Shipment_Status", "Traffic_Status"])
    assert dropped.count("\n") == 12 and "Shipment_Status" not in dropped
    renamed = rename_field(text, "Shipment_Status", "Status")
    assert "Status: In Transit" in renamed and "Shipment_Status" not in renamed
    order = shuffled_order(0)
    assert sorted(order) == sorted(FIELDS) and order != list(FIELDS)
    reordered = reorder_fields(text, order)
    assert sorted(reordered.split("\n")) == sorted(text.split("\n"))
    assert reordered.split("\n")[0].startswith(order[0] + ":")
