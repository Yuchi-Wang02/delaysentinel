"""Gradio demo: watch a 1.24B-parameter model behave like a two-clause rule.

The point of this Space is not that the model predicts delays. It is that you can
edit the two fields the Kaggle label was built from and watch the prediction flip,
edit any other field and watch nothing happen, or delete the two fields and watch
the model answer anyway.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import gradio as gr
import pandas as pd

HERE = Path(__file__).resolve().parent
for candidate in (HERE / "src", HERE.parent / "src"):
    if candidate.exists():
        sys.path.insert(0, str(candidate))

from delaysentinel.data import rule_predict  # noqa: E402
from delaysentinel.prompting import FIELDS, RULE_FIELDS, drop_fields, parse_label, serialize_row  # noqa: E402

PRIMARY_ID = os.environ.get("MODEL_ID", "Yuchiwang02/Llama-3.2-1B-DelaySentinel")
FALLBACK_ID = os.environ.get(
    "MODEL_ID_FALLBACK", "Yuchiwang02/DelaySentinel"
)  # pre-rename id, kept for the redirect window
RULE = 'Logistics_Delay = 1  iff  Shipment_Status == "Delayed"  OR  Traffic_Status == "Heavy"'
MAX_ROWS = 200

EXAMPLE = {  # first row of data/test.jsonl, gold label 0
    "Timestamp": "2024-07-18 12:19:20",
    "Asset_ID": "Truck_10",
    "Latitude": "20.2969",
    "Longitude": "124.1885",
    "Inventory_Level": "298",
    "Shipment_Status": "In Transit",
    "Temperature": "18.6",
    "Humidity": "74.4",
    "Traffic_Status": "Detour",
    "Waiting_Time": "41",
    "User_Transaction_Amount": "248",
    "User_Purchase_Frequency": "1",
    "Logistics_Delay_Reason": "None",
    "Asset_Utilization": "86.3",
    "Demand_Forecast": "283",
}
STATUS_CHOICES = ["Delayed", "Delivered", "In Transit"]
TRAFFIC_CHOICES = ["Heavy", "Detour", "Clear"]
REASON_CHOICES = ["None", "Weather", "Traffic", "Mechanical Failure"]

_scorer = None
_load_error = None


def get_scorer():
    global _scorer, _load_error
    if _scorer is not None:
        return _scorer
    from delaysentinel.model import Scorer

    for model_id in (PRIMARY_ID, FALLBACK_ID):
        try:
            _scorer = Scorer(model_id, dtype="bf16", batch_size=8)
            return _scorer
        except Exception as exc:
            _load_error = f"{model_id}: {exc}"
    raise RuntimeError(_load_error)


def _fields_from_inputs(values: list) -> dict:
    return {name: value for name, value in zip(FIELDS, values, strict=False)}


def _flipped(fields: dict) -> tuple[dict, str]:
    """A copy of the record with the rule fields edited so that the rule's answer flips."""
    swapped = fields.copy()
    if fields["Shipment_Status"] == "Delayed" or fields["Traffic_Status"] == "Heavy":
        notes = []
        if fields["Shipment_Status"] == "Delayed":
            swapped["Shipment_Status"] = "In Transit"
            notes.append("`Shipment_Status` → In Transit")
        if fields["Traffic_Status"] == "Heavy":
            swapped["Traffic_Status"] = "Clear"
            notes.append("`Traffic_Status` → Clear")
        return swapped, " and ".join(notes) + " (both clauses cleared)" if len(notes) == 2 else " and ".join(notes)
    swapped["Shipment_Status"] = "Delayed"
    return swapped, "`Shipment_Status` → Delayed"


def predict_single(*values):
    fields = _fields_from_inputs(list(values))
    scorer = get_scorer()
    full = serialize_row(fields)
    without = drop_fields(full, RULE_FIELDS)
    swapped, flip_note = _flipped(fields)
    swapped_text = serialize_row(swapped)
    t0 = time.time()
    raw_full, raw_without, raw_swapped = scorer.generate([full, without, swapped_text])
    dt = time.time() - t0
    rule = int(rule_predict(pd.DataFrame([fields]))[0])
    label = parse_label(raw_full)
    verdict = {None: "unparsable", 0: "0 = no delay", 1: "1 = delay"}[label]
    agree = "n/a" if label is None else ("yes" if label == rule else "NO")
    summary = (
        f"**Model output:** `{raw_full}` → {verdict}\n\n"
        f"**Two-clause rule says:** {rule}  ·  model agrees with the rule: **{agree}**\n\n"
        f"**Same row with {flip_note}:** `{raw_swapped}`\n\n"
        f"**Same row with both rule fields deleted from the prompt:** `{raw_without}` "
        f"(no rule can apply here; the model still answers)\n\n"
        f"<small>3 generations in {dt:.1f} s on {scorer.device}.</small>"
    )
    return summary, full


def predict_csv(file, drop_rule):
    if file is None:
        return None, "Upload a CSV with the 15 input columns (the Kaggle file works as-is)."
    frame = pd.read_csv(file.name, keep_default_na=False, na_values=[""], dtype=str)
    missing = [f for f in FIELDS if f not in frame.columns]
    if missing:
        return None, f"Missing columns: {missing}"
    if frame.empty:
        return None, "The CSV has no data rows."
    n_total = len(frame)
    frame = frame.head(MAX_ROWS)
    scorer = get_scorer()
    texts = [serialize_row(r) for r in frame[list(FIELDS)].to_dict("records")]
    if drop_rule:
        texts = [drop_fields(t, RULE_FIELDS) for t in texts]
    t0 = time.time()
    raw, labels = scorer.predict(texts)
    dt = time.time() - t0
    out = frame[list(FIELDS)].copy()
    out["rule_label"] = rule_predict(frame)
    out["model_raw"] = raw
    out["model_label"] = ["" if lab is None else lab for lab in labels]
    out["agrees_with_rule"] = [
        ("" if lab is None else ("yes" if lab == r else "no"))
        for lab, r in zip(labels, out["rule_label"], strict=False)
    ]
    agree_rate = sum(1 for lab, r in zip(labels, out["rule_label"], strict=False) if lab == r) / len(out)
    pos = sum(1 for lab in labels if lab == 1) / len(out)
    prefix = f"(showing the first {MAX_ROWS} of {n_total} rows) " if n_total > MAX_ROWS else ""
    msg = (
        f"{prefix}{len(out)} rows scored in {dt:.1f} s. Model agrees with the rule on {agree_rate:.0%} of rows; "
        f"model positive rate {pos:.0%}."
    )
    if "Logistics_Delay" in frame.columns:
        gold = pd.to_numeric(frame["Logistics_Delay"], errors="coerce")
        scored = [(lab, g) for lab, g in zip(labels, gold, strict=False) if lab is not None and pd.notna(g)]
        if scored:
            acc = sum(1 for lab, g in scored if lab == int(g)) / len(scored)
            msg += f" Accuracy against the file's own label column: {acc:.1%} ({len(scored)} scored rows)."
    if drop_rule:
        msg += " (Rule fields were removed from every prompt before scoring.)"
    return out, msg


with gr.Blocks(title="Llama-3.2-1B-DelaySentinel: a leakage demo") as demo:
    gr.Markdown(
        """
# Llama-3.2-1B-DelaySentinel — a label-leakage demo, not a delay predictor

**Built with Llama.** This 1.24B-parameter model was fine-tuned to answer `Logistics_Delay: 0|1`
from 15 fields of a synthetic Kaggle logistics table. It scores 100% on its test split because
the label *is* the rule below, and a depth-2 decision tree scores the same:

```
Logistics_Delay = 1  iff  Shipment_Status == "Delayed"  OR  Traffic_Status == "Heavy"
```

Try it: change **Shipment_Status** or **Traffic_Status** and the answer flips; change anything
else and it does not. The result also shows the same row with the rule fields flipped the other
way and with both rule fields deleted. Model card, evaluation JSON and probes:
[Yuchiwang02/Llama-3.2-1B-DelaySentinel](https://huggingface.co/Yuchiwang02/Llama-3.2-1B-DelaySentinel)
· code: [github.com/Yuchi-Wang02/delaysentinel](https://github.com/Yuchi-Wang02/delaysentinel)
"""
    )
    with gr.Tab("One record"):
        inputs = []  # order must match FIELDS
        with gr.Row():
            with gr.Column():
                inputs.append(gr.Textbox(label="Timestamp", value=EXAMPLE["Timestamp"]))
                inputs.append(gr.Textbox(label="Asset_ID", value=EXAMPLE["Asset_ID"]))
                inputs.append(gr.Textbox(label="Latitude", value=EXAMPLE["Latitude"]))
                inputs.append(gr.Textbox(label="Longitude", value=EXAMPLE["Longitude"]))
                inputs.append(gr.Textbox(label="Inventory_Level", value=EXAMPLE["Inventory_Level"]))
                inputs.append(
                    gr.Dropdown(
                        STATUS_CHOICES,
                        label="Shipment_Status (rule field)",
                        value=EXAMPLE["Shipment_Status"],
                        allow_custom_value=True,
                    )
                )
                inputs.append(gr.Textbox(label="Temperature", value=EXAMPLE["Temperature"]))
                inputs.append(gr.Textbox(label="Humidity", value=EXAMPLE["Humidity"]))
            with gr.Column():
                inputs.append(
                    gr.Dropdown(
                        TRAFFIC_CHOICES,
                        label="Traffic_Status (rule field)",
                        value=EXAMPLE["Traffic_Status"],
                        allow_custom_value=True,
                    )
                )
                inputs.append(gr.Textbox(label="Waiting_Time", value=EXAMPLE["Waiting_Time"]))
                inputs.append(gr.Textbox(label="User_Transaction_Amount", value=EXAMPLE["User_Transaction_Amount"]))
                inputs.append(gr.Textbox(label="User_Purchase_Frequency", value=EXAMPLE["User_Purchase_Frequency"]))
                inputs.append(
                    gr.Dropdown(
                        REASON_CHOICES,
                        label="Logistics_Delay_Reason",
                        value=EXAMPLE["Logistics_Delay_Reason"],
                        allow_custom_value=True,
                    )
                )
                inputs.append(gr.Textbox(label="Asset_Utilization", value=EXAMPLE["Asset_Utilization"]))
                inputs.append(gr.Textbox(label="Demand_Forecast", value=EXAMPLE["Demand_Forecast"]))
        run = gr.Button("Score this record", variant="primary")
        result = gr.Markdown()
        prompt_box = gr.Code(label="Exact prompt sent to the model (user turn)", language=None)
        run.click(predict_single, inputs=inputs, outputs=[result, prompt_box])
    with gr.Tab(f"CSV (up to {MAX_ROWS} rows)"):
        gr.Markdown(
            "Upload the Kaggle CSV or any file with the same 15 columns. "
            "Optionally strip the two rule fields from every prompt first."
        )
        file_in = gr.File(label="CSV", file_types=[".csv"])
        drop = gr.Checkbox(
            label="Delete Shipment_Status and Traffic_Status from every prompt before scoring", value=False
        )
        go = gr.Button("Score file")
        table = gr.Dataframe(label="Predictions", wrap=True)
        status = gr.Markdown()
        go.click(predict_csv, inputs=[file_in, drop], outputs=[table, status])
    gr.Markdown(
        """
<small>Weights: Llama 3.2 Community License; copies of LICENSE, USE_POLICY.md and NOTICE are in this Space's
Files tab and in the model repo. Code: MIT (LICENSE-MIT). Data: Kaggle ziya07, CC0. Every number in the model
card comes from <code>results/eval.json</code> in the GitHub repository. This demo downloads the published bf16
weights (2.47 GB) on the first request and runs them on the Space's CPU.</small>
"""
    )

if __name__ == "__main__":
    demo.launch()
