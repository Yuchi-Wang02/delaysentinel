"""Local Gradio demo for inspecting model responses to record edits.

Compare an original record, a rule-field rewrite and a prompt with those fields
removed. The dataset's label rule supplies a reference; the generated outputs
show what the published checkpoint does on the specific inputs entered.
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
)  # historical model id; the Hub redirects it to the current repository
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
    verdict = {None: "unparsable", 0: "dataset label 0", 1: "dataset label 1"}[label]
    agree = "n/a" if label is None else ("yes" if label == rule else "NO")
    summary = (
        f"**Model output:** `{raw_full}` → {verdict}\n\n"
        f"**Two-clause rule says:** {rule}  ·  model agrees with the rule: **{agree}**\n\n"
        f"**Same row with {flip_note}:** `{raw_swapped}`\n\n"
        f"**Same row with both rule fields deleted from the prompt:** `{raw_without}` "
        f"(the two-field reference rule is undefined on this reduced prompt)\n\n"
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
        f"{prefix}{len(out)} rows scored in {dt:.1f} s. "
        f"Model agrees with the original rows' rule labels on {agree_rate:.0%} of rows; "
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


with gr.Blocks(title="Llama-3.2-1B-DelaySentinel: model behavior demo") as demo:
    gr.Markdown(
        """
# Llama-3.2-1B-DelaySentinel: explore model behavior

**Built with Llama.** This 1.24B-parameter model was fine-tuned to answer `Logistics_Delay: 0|1`
from 15 fields of a Kaggle logistics table. On the historical 200-row evaluation split,
the checkpoint and a depth-2 decision tree both score 100%. The rule below reconstructs
the label across all 1,000 source rows:

```
Logistics_Delay = 1  iff  Shipment_Status == "Delayed"  OR  Traffic_Status == "Heavy"
```

**Try a comparison.** Edit a field, score the record, and inspect the exact prompt. Each run
also shows a rewrite that changes the rule's answer and a version with both rule fields removed.
Custom values let you explore where a model response differs from the literal rule.

This is a local exploration tool; a hosted Space has not been created. Outputs describe this
checkpoint's behavior on the entered text and are not delivery-risk estimates. The historical
split was partly exposed during training-time evaluation. Saved results and tested rewrites:
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
<small>Weights: Llama 3.2 Community License; LICENSE, USE_POLICY.md and NOTICE are in the model repository.
Code: MIT (LICENSE-MIT). Source data: Kaggle ziya07, listed as CC0. Saved results and training records
are linked from the model card. The first prediction downloads the published bf16 weights (2.47 GB);
the result reports the device used by the model wrapper.</small>
"""
    )

if __name__ == "__main__":
    demo.launch()
