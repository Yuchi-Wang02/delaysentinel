"""Counterfactual, robustness and out-of-distribution probes.

Every probe rewrites the *text* of test prompts and re-scores the model, so what is
measured is the behaviour of the published weights on inputs that differ from the
original in one controlled way. Probe construction is pure Python and unit-tested;
scoring needs the model (see :func:`run_probe`).

Groups
------
``counterfactual``  edits to the two rule fields (expect a flip) and to every non-rule
                    field, one value per field (expect no change)
``robustness``      column renames, column order, case variants, synonyms and
                    near-synonyms, antonyms/negations, the trigger value placed in a
                    non-rule field or in the other rule field, one rule field removed,
                    unseen values
``ood``             prompts outside the training schema
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .prompting import (
    FIELDS,
    RULE_FIELDS,
    drop_fields,
    parse_user_text,
    rename_field,
    reorder_fields,
    set_field,
    shuffled_order,
)

README_V0_EXAMPLE = (
    "order_id: 123\norigin_region: OH\ndest_region: CA\ncarrier: A1\nservice_level: ground\n"
    "weight_kg: 10.5\ndistance_km: 3500\nholiday_flag: 0"
)

#: One replacement value per non-rule field (max of the CSV range for numerics, the
#: most distinct categorical value otherwise). Used for the "does anything else move
#: the prediction" probes on negatives and positives.
NON_RULE_EDITS: dict[str, str] = {
    "Timestamp": "2024-12-30 20:21:58",
    "Asset_ID": "Truck_10",
    "Latitude": "0.0",
    "Longitude": "0.0",
    "Inventory_Level": "500",
    "Temperature": "30.0",
    "Humidity": "80.0",
    "Waiting_Time": "60",
    "User_Transaction_Amount": "500",
    "User_Purchase_Frequency": "10",
    "Logistics_Delay_Reason": "Traffic",
    "Asset_Utilization": "100.0",
    "Demand_Forecast": "300",
}

#: Value rewrites for the two rule fields: (key suffix, new value, kind).
DELAYED_VARIANTS = [
    ("DELAYED", "DELAYED", "case"),
    ("delayed", "delayed", "case"),
    ("Late", "Late", "synonym"),
    ("Behind_schedule", "Behind schedule", "synonym"),
    ("Postponed", "Postponed", "synonym"),
    ("Overdue", "Overdue", "synonym"),
    ("Held_up", "Held up", "synonym"),
    ("On_Time", "On Time", "antonym"),
    ("Early", "Early", "antonym"),
    ("Pending", "Pending", "neighbour"),
    ("Not_Delayed", "Not Delayed", "negation"),
    ("Delay", "Delay", "shared_token"),
]
#: 12 capitalised English words that the tokenizer encodes as a single token, unrelated to delay
#: or traffic. They give the base rate at which an unseen value in a rule field makes the model
#: answer 1, which the synonym probes on their own cannot establish.
CONTROL_WORDS = (
    "Copper",
    "Violet",
    "Harbor",
    "Maple",
    "Quartz",
    "Falcon",
    "Meadow",
    "Cobalt",
    "Lantern",
    "Marble",
    "Willow",
    "Amber",
)

HEAVY_VARIANTS = [
    ("HEAVY", "HEAVY", "case"),
    ("heavy", "heavy", "case"),
    ("Congested", "Congested", "synonym"),
    ("Jammed", "Jammed", "synonym"),
    ("Gridlock", "Gridlock", "synonym"),
    ("Slow", "Slow", "synonym"),
    ("Dense", "Dense", "synonym"),
    ("Light", "Light", "antonym"),
    ("Free_flowing", "Free-flowing", "antonym"),
    ("Moderate", "Moderate", "neighbour"),
    ("Not_Heavy", "Not Heavy", "negation"),
    ("Heav", "Heav", "shared_token"),
]


@dataclass
class Probe:
    key: str
    group: str  # "counterfactual" | "robustness" | "ood"
    description: str
    indices: list[int]
    texts: list[str]
    expected: int | None  # label the rule (or common sense) predicts after the edit, if any
    expected_kind: str = ""  # "flip_to_0" | "flip_to_1" | "unchanged" | "gold" | "none"
    notes: str = ""
    meta: dict = field(default_factory=dict)


def _idx(mask: pd.Series) -> list[int]:
    return [int(i) for i in np.flatnonzero(mask.to_numpy())]


# ---------------------------------------------------------------- counterfactual
def counterfactual_probes(test: pd.DataFrame, users: Sequence[str]) -> list[Probe]:
    status = test["Shipment_Status"].astype(str)
    traffic = test["Traffic_Status"].astype(str)
    gold = test["gold"].astype(int)
    neg = _idx(gold == 0)
    pos = _idx(gold == 1)

    probes: list[Probe] = []

    idx = _idx((status == "Delayed") & (traffic != "Heavy"))
    probes.append(
        Probe(
            "status_Delayed_to_InTransit",
            "counterfactual",
            "positives that are positive only because Shipment_Status == Delayed; set it to In Transit",
            idx,
            [set_field(users[i], "Shipment_Status", "In Transit") for i in idx],
            0,
            "flip_to_0",
            meta={"rule_field_edit": True},
        )
    )
    idx = _idx((traffic == "Heavy") & (status != "Delayed"))
    probes.append(
        Probe(
            "traffic_Heavy_to_Clear",
            "counterfactual",
            "positives that are positive only because Traffic_Status == Heavy; set it to Clear",
            idx,
            [set_field(users[i], "Traffic_Status", "Clear") for i in idx],
            0,
            "flip_to_0",
            meta={"rule_field_edit": True},
        )
    )
    probes.append(
        Probe(
            "negatives_traffic_to_Heavy",
            "counterfactual",
            "all negatives; set Traffic_Status to Heavy",
            neg,
            [set_field(users[i], "Traffic_Status", "Heavy") for i in neg],
            1,
            "flip_to_1",
            meta={"rule_field_edit": True},
        )
    )
    probes.append(
        Probe(
            "negatives_status_to_Delayed",
            "counterfactual",
            "all negatives; set Shipment_Status to Delayed",
            neg,
            [set_field(users[i], "Shipment_Status", "Delayed") for i in neg],
            1,
            "flip_to_1",
            meta={"rule_field_edit": True},
        )
    )
    # legacy paired edits kept for continuity with the first evaluation
    probes.append(
        Probe(
            "negatives_waiting60_temp30",
            "counterfactual",
            "all negatives; set Waiting_Time=60 and Temperature=30.0 (two non-rule fields)",
            neg,
            [set_field(set_field(users[i], "Waiting_Time", 60), "Temperature", "30.0") for i in neg],
            0,
            "unchanged",
        )
    )
    probes.append(
        Probe(
            "positives_waiting10_temp18",
            "counterfactual",
            "all positives; set Waiting_Time=10 and Temperature=18.0 (two non-rule fields)",
            pos,
            [set_field(set_field(users[i], "Waiting_Time", 10), "Temperature", "18.0") for i in pos],
            1,
            "unchanged",
        )
    )
    # one probe per non-rule field, on negatives and on positives
    for fld, value in NON_RULE_EDITS.items():
        for grp, idx_, exp in (("negatives", neg, 0), ("positives", pos, 1)):
            probes.append(
                Probe(
                    f"nonrule_{grp}_{fld}",
                    "counterfactual",
                    f"all {grp}; set {fld} to {value!r} (non-rule field)",
                    idx_,
                    [set_field(users[i], fld, value) for i in idx_],
                    exp,
                    "unchanged",
                    meta={"non_rule_field_edit": True, "field": fld, "value": value},
                )
            )
    for value in ("Mechanical Failure", "None"):
        for grp, idx_, exp in (("negatives", neg, 0), ("positives", pos, 1)):
            probes.append(
                Probe(
                    f"nonrule_{grp}_Logistics_Delay_Reason_{value.replace(' ', '_')}",
                    "counterfactual",
                    f"all {grp}; set Logistics_Delay_Reason to {value!r} (non-rule field)",
                    idx_,
                    [set_field(users[i], "Logistics_Delay_Reason", value) for i in idx_],
                    exp,
                    "unchanged",
                    meta={"non_rule_field_edit": True, "field": "Logistics_Delay_Reason", "value": value},
                )
            )
    all_idx = list(range(len(users)))
    probes.append(
        Probe(
            "both_rule_fields_removed",
            "counterfactual",
            "all rows; delete the Shipment_Status and Traffic_Status lines",
            all_idx,
            [drop_fields(users[i], RULE_FIELDS) for i in all_idx],
            None,
            "none",
            notes="no rule applies; report what the model emits",
        )
    )
    return probes


# ------------------------------------------------------------------- robustness
def robustness_probes(test: pd.DataFrame, users: Sequence[str]) -> list[Probe]:
    status = test["Shipment_Status"].astype(str)
    traffic = test["Traffic_Status"].astype(str)
    gold = test["gold"].astype(int)
    all_idx = list(range(len(users)))
    neg = _idx(gold == 0)
    delayed_rows = _idx(status == "Delayed")
    heavy_rows = _idx(traffic == "Heavy")
    probes: list[Probe] = []

    probes.append(
        Probe(
            "rename_columns_both",
            "robustness",
            "all rows; rename Shipment_Status -> Status and Traffic_Status -> Traffic",
            all_idx,
            [
                rename_field(rename_field(users[i], "Shipment_Status", "Status"), "Traffic_Status", "Traffic")
                for i in all_idx
            ],
            None,
            "gold",
            notes="a semantic reader keeps the gold label; a lexical reader may not",
        )
    )
    probes.append(
        Probe(
            "rename_status_only",
            "robustness",
            "all rows; rename Shipment_Status -> Status",
            all_idx,
            [rename_field(users[i], "Shipment_Status", "Status") for i in all_idx],
            None,
            "gold",
        )
    )
    probes.append(
        Probe(
            "rename_traffic_only",
            "robustness",
            "all rows; rename Traffic_Status -> Traffic",
            all_idx,
            [rename_field(users[i], "Traffic_Status", "Traffic") for i in all_idx],
            None,
            "gold",
        )
    )
    for suffix, value, kind in DELAYED_VARIANTS:
        probes.append(
            Probe(
                f"value_Delayed_to_{suffix}",
                "robustness",
                f"rows with Shipment_Status == Delayed; write the value as {value!r} ({kind})",
                delayed_rows,
                [set_field(users[i], "Shipment_Status", value) for i in delayed_rows],
                None,
                "gold",
                notes="gold is 1 for every such row; the literal rule no longer fires",
                meta={"variant_kind": kind, "field": "Shipment_Status", "value": value},
            )
        )
    for suffix, value, kind in HEAVY_VARIANTS:
        probes.append(
            Probe(
                f"value_Heavy_to_{suffix}",
                "robustness",
                f"rows with Traffic_Status == Heavy; write the value as {value!r} ({kind})",
                heavy_rows,
                [set_field(users[i], "Traffic_Status", value) for i in heavy_rows],
                None,
                "gold",
                notes="gold is 1 for every such row; the literal rule no longer fires",
                meta={"variant_kind": kind, "field": "Traffic_Status", "value": value},
            )
        )
    for seed in (0, 1, 2):
        order = shuffled_order(seed)
        probes.append(
            Probe(
                f"shuffle_column_order_seed{seed}",
                "robustness",
                "all rows; permute the 15 lines with a fixed seed",
                all_idx,
                [reorder_fields(users[i], order) for i in all_idx],
                None,
                "gold",
                meta={"order": order},
            )
        )
    # trigger value placed where it never occurred in training
    probes.append(
        Probe(
            "trigger_Heavy_in_Logistics_Delay_Reason",
            "robustness",
            "all negatives; Logistics_Delay_Reason = Heavy (trigger value in a non-rule field)",
            neg,
            [set_field(users[i], "Logistics_Delay_Reason", "Heavy") for i in neg],
            0,
            "unchanged",
            notes="a field-bound rule keeps 0; a bag-of-words trigger fires",
            meta={"trigger_placement": True},
        )
    )
    probes.append(
        Probe(
            "trigger_Delayed_in_Logistics_Delay_Reason",
            "robustness",
            "all negatives; Logistics_Delay_Reason = Delayed (trigger value in a non-rule field)",
            neg,
            [set_field(users[i], "Logistics_Delay_Reason", "Delayed") for i in neg],
            0,
            "unchanged",
            meta={"trigger_placement": True},
        )
    )
    probes.append(
        Probe(
            "trigger_Delayed_in_Asset_ID",
            "robustness",
            "all negatives; Asset_ID = Delayed (trigger value in a non-rule field)",
            neg,
            [set_field(users[i], "Asset_ID", "Delayed") for i in neg],
            0,
            "unchanged",
            meta={"trigger_placement": True},
        )
    )
    probes.append(
        Probe(
            "trigger_swapped_fields",
            "robustness",
            "all negatives; Shipment_Status = Heavy and Traffic_Status = Delayed (values in the other rule field)",
            neg,
            [set_field(set_field(users[i], "Shipment_Status", "Heavy"), "Traffic_Status", "Delayed") for i in neg],
            0,
            "unchanged",
            notes="the literal rule gives 0; a bag-of-words trigger gives 1",
            meta={"trigger_placement": True},
        )
    )
    probes.append(
        Probe(
            "only_status_removed",
            "robustness",
            "all rows; delete the Shipment_Status line",
            all_idx,
            [drop_fields(users[i], ["Shipment_Status"]) for i in all_idx],
            None,
            "none",
            notes="the remaining clause Traffic_Status == Heavy still applies",
        )
    )
    probes.append(
        Probe(
            "only_traffic_removed",
            "robustness",
            "all rows; delete the Traffic_Status line",
            all_idx,
            [drop_fields(users[i], ["Traffic_Status"]) for i in all_idx],
            None,
            "none",
            notes="the remaining clause Shipment_Status == Delayed still applies",
        )
    )
    for word in CONTROL_WORDS:
        probes.append(
            Probe(
                f"control_word_Shipment_Status_{word}",
                "robustness",
                f"rows with Shipment_Status == Delayed; write the value as {word!r} (unrelated single-token word)",
                delayed_rows,
                [set_field(users[i], "Shipment_Status", word) for i in delayed_rows],
                None,
                "gold",
                notes="base rate for an unseen value in this field; gold is 1 for every such row",
                meta={"variant_kind": "control_word", "field": "Shipment_Status", "value": word},
            )
        )
        probes.append(
            Probe(
                f"control_word_Traffic_Status_{word}",
                "robustness",
                f"rows with Traffic_Status == Heavy; write the value as {word!r} (unrelated single-token word)",
                heavy_rows,
                [set_field(users[i], "Traffic_Status", word) for i in heavy_rows],
                None,
                "gold",
                notes="base rate for an unseen value in this field; gold is 1 for every such row",
                meta={"variant_kind": "control_word", "field": "Traffic_Status", "value": word},
            )
        )
    for label in ("Heavy", "Delayed"):
        line = f"Note: the depot supervisor is Mr. {label}"
        probes.append(
            Probe(
                f"trigger_{label}_in_free_text",
                "robustness",
                f"all negatives; append a free-text line containing {label!r} outside any known field",
                neg,
                [users[i] + "\n" + line for i in neg],
                0,
                "unchanged",
                notes="a field-bound rule keeps 0; a detector that scans the whole turn fires",
                meta={"trigger_placement": True, "free_text": True, "value": label},
            )
        )
    probes.append(
        Probe(
            "rule_fields_oov",
            "robustness",
            "all rows; Shipment_Status = Unknown and Traffic_Status = N/A (values never seen in training)",
            all_idx,
            [set_field(set_field(users[i], "Shipment_Status", "Unknown"), "Traffic_Status", "N/A") for i in all_idx],
            None,
            "none",
        )
    )
    return probes


# -------------------------------------------------------------------------- ood
def ood_probes() -> list[Probe]:
    return [
        Probe(
            "readme_v0_schema",
            "ood",
            "the usage example from the original model card: a schema the model never saw",
            [],
            [README_V0_EXAMPLE],
            None,
            "none",
        ),
        Probe("empty_user", "ood", "an empty user turn", [], [""], None, "none"),
        Probe(
            "unrelated_question",
            "ood",
            "a question unrelated to logistics",
            [],
            ["What is the capital of France? Answer in one word."],
            None,
            "none",
        ),
        Probe(
            "header_only",
            "ood",
            "the 15 column names with no values",
            [],
            ["\n".join(f"{f}:" for f in FIELDS)],
            None,
            "none",
        ),
    ]


# ------------------------------------------------------------------- scoring
def _literal_rule_on_texts(texts: Sequence[str]) -> list[int | None]:
    """The two-clause rule read literally off the rewritten text; ``None`` when either
    rule field is absent (renamed or deleted), because the rule is then undefined."""
    out: list[int | None] = []
    for text in texts:
        fields = parse_user_text(text)
        if "Shipment_Status" not in fields or "Traffic_Status" not in fields:
            out.append(None)
            continue
        delayed = fields.get("Shipment_Status", "").strip() == "Delayed"
        heavy = fields.get("Traffic_Status", "").strip() == "Heavy"
        out.append(int(delayed or heavy))
    return out


def _remaining_clause_on_texts(texts: Sequence[str]) -> list[int | None]:
    """When one rule field was *deleted*, the value of the clause that is left.

    A renamed column (``Status``, ``Traffic``) is not a deleted one, so a prompt that still has
    all 15 lines returns ``None``: reading it as a single clause would be meaningless.
    """
    out: list[int | None] = []
    for text in texts:
        fields = parse_user_text(text)
        if len(fields) >= len(FIELDS):  # nothing was removed; a rename, not a deletion
            out.append(None)
            continue
        has_s, has_t = "Shipment_Status" in fields, "Traffic_Status" in fields
        if has_s and not has_t:
            out.append(int(fields["Shipment_Status"].strip() == "Delayed"))
        elif has_t and not has_s:
            out.append(int(fields["Traffic_Status"].strip() == "Heavy"))
        else:
            out.append(None)
    return out


def run_probe(scorer, probe: Probe, gold: Sequence[int]) -> dict:
    raw, labels = scorer.predict(probe.texts)
    n = len(probe.texts)
    preds = np.array([-1 if lab is None else lab for lab in labels])
    result: dict = {
        "group": probe.group,
        "description": probe.description,
        "n": n,
        "unparsable": int((preds == -1).sum()),
        "pred_counts": {"0": int((preds == 0).sum()), "1": int((preds == 1).sum())},
        "pred_pos_rate": round(float((preds == 1).sum() / n), 4) if n else None,
        "expected": probe.expected,
        "expected_kind": probe.expected_kind,
    }
    if probe.notes:
        result["notes"] = probe.notes
    if probe.meta:
        result["meta"] = probe.meta
    if probe.expected is not None:
        result["matches_expected"] = int((preds == probe.expected).sum())
    if probe.indices:
        g = np.array([gold[i] for i in probe.indices])
        result["acc_vs_original_gold"] = round(float((preds == g).mean()), 4)
        literal = _literal_rule_on_texts(probe.texts)
        if all(v is not None for v in literal):
            lit = np.array(literal)
            result["agreement_with_literal_rule_on_rewritten_text"] = round(float((preds == lit).mean()), 4)
        remaining = _remaining_clause_on_texts(probe.texts)
        if all(v is not None for v in remaining):
            rem = np.array(remaining)
            result["agreement_with_remaining_clause"] = round(float((preds == rem).mean()), 4)
    if getattr(scorer, "score_available", False) and n:
        scores = scorer.label_scores(probe.texts)
        if scores:
            margins = np.array([s["margin"] for s in scores])
            result["margin"] = {
                "min": round(float(margins.min()), 3),
                "max": round(float(margins.max()), 3),
                "abs_median": round(float(np.median(np.abs(margins))), 3),
                "abs_min": round(float(np.abs(margins).min()), 3),
            }
    if probe.group == "ood" or n <= 4:
        result["raw_outputs"] = raw
    else:
        result["raw_output_examples"] = raw[:3]
    return result


def summarize_counterfactuals(
    results: dict[str, dict], probes: Sequence[Probe], users: Sequence[str] | None = None
) -> dict:
    """Aggregate counts that the model card quotes: rule-field edits vs non-rule edits.

    ``users`` are the original prompts; when given, rewrites that are byte-identical to the
    original (the row already had that value) are counted separately, because they are not edits.
    """
    by_key = {p.key: p for p in probes}
    rule_edits = rule_matches = 0
    rule_rows: set[int] = set()
    nonrule_edits = nonrule_changed = nonrule_identical = nonrule_pairs = 0
    nonrule_fields: set[str] = set()
    for key, res in results.items():
        p = by_key.get(key)
        if p is None:
            continue
        if p.meta.get("rule_field_edit"):
            rule_edits += res["n"]
            rule_matches += res.get("matches_expected", 0)
            rule_rows.update(p.indices)
        elif p.expected_kind == "unchanged" and p.group == "counterfactual":
            nonrule_edits += res["n"]
            nonrule_changed += res["n"] - res.get("matches_expected", 0)
            if users is not None:
                nonrule_identical += sum(1 for i, t in zip(p.indices, p.texts, strict=False) if t == users[i])
            if p.meta.get("field"):
                nonrule_fields.add(p.meta["field"])
            else:
                nonrule_pairs += res["n"]
                nonrule_fields.update({"Waiting_Time", "Temperature"})
    return {
        "rule_field_edits": rule_edits,
        "rule_field_edits_flipped_as_expected": rule_matches,
        "distinct_rows_with_rule_field_edit": len(rule_rows),
        "non_rule_field_edits": nonrule_edits,
        "non_rule_single_field_edits": nonrule_edits - nonrule_pairs,
        "non_rule_two_field_edits": nonrule_pairs,
        "non_rule_field_edits_identical_to_the_original": nonrule_identical,
        "non_rule_field_edits_that_changed_the_prediction": nonrule_changed,
        "non_rule_fields_probed": sorted(nonrule_fields),
        "non_rule_fields_probed_count": len(nonrule_fields),
    }
