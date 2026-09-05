"""Counterfactual, robustness and out-of-distribution probes.

Every probe rewrites the *text* of test prompts and re-scores the model, so what is
measured is the behaviour of the published weights on inputs that differ from the
original in one controlled way. Probe construction is pure Python and unit-tested;
scoring needs the model (see :func:`run_probe`).
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
        )
    )
    idx = _idx(gold == 0)
    probes.append(
        Probe(
            "negatives_traffic_to_Heavy",
            "counterfactual",
            "all negatives; set Traffic_Status to Heavy",
            idx,
            [set_field(users[i], "Traffic_Status", "Heavy") for i in idx],
            1,
            "flip_to_1",
        )
    )
    probes.append(
        Probe(
            "negatives_status_to_Delayed",
            "counterfactual",
            "all negatives; set Shipment_Status to Delayed",
            idx,
            [set_field(users[i], "Shipment_Status", "Delayed") for i in idx],
            1,
            "flip_to_1",
        )
    )
    probes.append(
        Probe(
            "negatives_waiting60_temp30",
            "counterfactual",
            "all negatives; set Waiting_Time=60 and Temperature=30.0 (non-rule fields)",
            idx,
            [set_field(set_field(users[i], "Waiting_Time", 60), "Temperature", "30.0") for i in idx],
            0,
            "unchanged",
        )
    )
    idx = _idx(gold == 1)
    probes.append(
        Probe(
            "positives_waiting10_temp18",
            "counterfactual",
            "all positives; set Waiting_Time=10 and Temperature=18.0 (non-rule fields)",
            idx,
            [set_field(set_field(users[i], "Waiting_Time", 10), "Temperature", "18.0") for i in idx],
            1,
            "unchanged",
        )
    )
    idx = list(range(len(users)))
    probes.append(
        Probe(
            "both_rule_fields_removed",
            "counterfactual",
            "all rows; delete the Shipment_Status and Traffic_Status lines",
            idx,
            [drop_fields(users[i], RULE_FIELDS) for i in idx],
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
    all_idx = list(range(len(users)))
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
    for key, old, new, column in (
        ("synonym_Delayed_to_Late", "Delayed", "Late", "Shipment_Status"),
        ("synonym_Delayed_to_DELAYED", "Delayed", "DELAYED", "Shipment_Status"),
        ("synonym_Delayed_to_delayed", "Delayed", "delayed", "Shipment_Status"),
        ("synonym_Heavy_to_Congested", "Heavy", "Congested", "Traffic_Status"),
        ("synonym_Heavy_to_HEAVY", "Heavy", "HEAVY", "Traffic_Status"),
        ("synonym_Heavy_to_heavy", "Heavy", "heavy", "Traffic_Status"),
    ):
        series = status if column == "Shipment_Status" else traffic
        idx = _idx(series == old)
        probes.append(
            Probe(
                key,
                "robustness",
                f"rows with {column} == {old}; write the value as {new!r}",
                idx,
                [set_field(users[i], column, new) for i in idx],
                None,
                "gold",
                notes="gold is 1 for every such row; the literal rule no longer fires",
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
    out: list[int | None] = []
    for text in texts:
        fields = parse_user_text(text)
        if "Shipment_Status" not in fields and "Traffic_Status" not in fields:
            out.append(None)
            continue
        delayed = fields.get("Shipment_Status", "").strip() == "Delayed"
        heavy = fields.get("Traffic_Status", "").strip() == "Heavy"
        out.append(int(delayed or heavy))
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
    if probe.group == "ood" or n <= 4:
        result["raw_outputs"] = raw
    else:
        result["raw_output_examples"] = raw[:3]
    return result
