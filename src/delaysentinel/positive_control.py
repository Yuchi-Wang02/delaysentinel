"""Positive control on real order-level data: late-delivery classification on Olist.

The DelaySentinel table cannot support a delay model (its label is a rule over two of its
own columns). This module shows what the *same* evaluation discipline produces on a public
dataset that has a promised date and an actual date: the Olist Brazilian e-commerce orders
(≈100k orders, 2016-2018). It is deliberately a classical-baseline study, not an LLM one.

    python -m delaysentinel.positive_control --out results/olist_positive_control.json

What it does
- downloads the Olist CSVs from a public Hub mirror (cached under data/olist/, git-ignored);
- keeps delivered orders with a purchase date at least 60 days before the last purchase in
  the data (right-censoring guard) and defines ``late = delivered_customer_date >
  estimated_delivery_date`` at day granularity;
- uses only fields known at checkout/approval: purchase month, weekday and hour, promised
  lead time (estimated date minus purchase date), customer state, seller state and product
  category of the first item, item count, total price, total freight, payment type and
  instalments, and the great-circle distance between customer and seller zip-code
  centroids (from the public geolocation table);
- time-based split: train on purchases before 2018-03-01, test on 2018-03-01 .. cut-off;
- reports prevalence, logistic regression and HistGradientBoosting with AUROC / AUPRC /
  Brier and bootstrap intervals, plus a cost-threshold sweep for the decision "expedite if
  flagged", where the threshold is ``p* = C_expedite / C_chargeback`` because the expedite
  cost is paid on every flagged order (see README, *What would make this non-trivial*).
"""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .stats import bootstrap_ci

ROOT = Path(__file__).resolve().parents[2]
MIRROR = "aviahYadler/Olist_Ecommerce_Dataset"
FILES = [
    "olist_orders_dataset.csv",
    "olist_customers_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_sellers_dataset.csv",
    "olist_products_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_geolocation_dataset.csv",
]
TRAIN_END = pd.Timestamp("2018-03-01")
CENSOR_DAYS = 60
NUMERIC = ["promised_lead_days", "n_items", "total_price", "total_freight", "payment_installments", "distance_km"]
CATEGORICAL = [
    "purchase_month",
    "purchase_weekday",
    "purchase_hour_bin",
    "customer_state",
    "seller_state",
    "category",
    "payment_type",
]


def download(data_dir: Path) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    data_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in FILES:
        target = data_dir / name
        if not target.exists():
            got = hf_hub_download(MIRROR, name, repo_type="dataset", local_dir=str(data_dir))
            target = Path(got)
        out[name] = target
    return out


def _haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def build_table(paths: dict[str, Path]) -> tuple[pd.DataFrame, dict]:
    orders = pd.read_csv(
        paths["olist_orders_dataset.csv"],
        parse_dates=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
    )
    customers = pd.read_csv(paths["olist_customers_dataset.csv"])
    items = pd.read_csv(paths["olist_order_items_dataset.csv"])
    sellers = pd.read_csv(paths["olist_sellers_dataset.csv"])
    products = pd.read_csv(paths["olist_products_dataset.csv"])
    payments = pd.read_csv(paths["olist_order_payments_dataset.csv"])
    geo = pd.read_csv(paths["olist_geolocation_dataset.csv"])

    notes: dict = {"orders_total": len(orders)}
    last_purchase = orders["order_purchase_timestamp"].max()
    cutoff = last_purchase - pd.Timedelta(days=CENSOR_DAYS)
    notes["last_purchase_in_data"] = str(last_purchase.date())
    notes["cutoff_purchase_date"] = str(cutoff.date())
    delivered = orders[(orders["order_status"] == "delivered") & orders["order_delivered_customer_date"].notna()]
    notes["orders_delivered"] = len(delivered)
    delivered = delivered[delivered["order_purchase_timestamp"] <= cutoff].copy()
    notes["orders_delivered_before_cutoff"] = len(delivered)
    delivered["late"] = (
        delivered["order_delivered_customer_date"].dt.normalize()
        > delivered["order_estimated_delivery_date"].dt.normalize()
    ).astype(int)

    # first item defines seller and category; totals over all items
    items_sorted = items.sort_values(["order_id", "order_item_id"])
    first = items_sorted.groupby("order_id").first()[["seller_id", "product_id"]]
    totals = items_sorted.groupby("order_id").agg(
        n_items=("order_item_id", "count"), total_price=("price", "sum"), total_freight=("freight_value", "sum")
    )
    pay = (
        payments.sort_values(["order_id", "payment_sequential"])
        .groupby("order_id")
        .agg(payment_type=("payment_type", "first"), payment_installments=("payment_installments", "max"))
    )
    geo_centroid = geo.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].median()

    table = delivered.merge(
        customers[["customer_id", "customer_state", "customer_zip_code_prefix"]], on="customer_id", how="left"
    )
    table = (
        table.merge(first, on="order_id", how="left")
        .merge(totals, on="order_id", how="left")
        .merge(pay, on="order_id", how="left")
    )
    table = table.merge(sellers[["seller_id", "seller_state", "seller_zip_code_prefix"]], on="seller_id", how="left")
    table = table.merge(products[["product_id", "product_category_name"]], on="product_id", how="left")
    table = table.merge(
        geo_centroid.rename(columns={"geolocation_lat": "c_lat", "geolocation_lng": "c_lng"}),
        left_on="customer_zip_code_prefix",
        right_index=True,
        how="left",
    )
    table = table.merge(
        geo_centroid.rename(columns={"geolocation_lat": "s_lat", "geolocation_lng": "s_lng"}),
        left_on="seller_zip_code_prefix",
        right_index=True,
        how="left",
    )
    table["distance_km"] = _haversine(table["c_lat"], table["c_lng"], table["s_lat"], table["s_lng"])

    ts = table["order_purchase_timestamp"]
    table["purchase_month"] = ts.dt.month.astype(str)
    table["purchase_weekday"] = ts.dt.weekday.astype(str)
    table["purchase_hour_bin"] = pd.cut(
        ts.dt.hour, bins=[-1, 5, 11, 17, 23], labels=["night", "morning", "afternoon", "evening"]
    ).astype(str)
    table["promised_lead_days"] = (table["order_estimated_delivery_date"].dt.normalize() - ts.dt.normalize()).dt.days
    table["category"] = table["product_category_name"].fillna("unknown")
    table["seller_state"] = table["seller_state"].fillna("unknown")
    table["payment_type"] = table["payment_type"].fillna("unknown")
    table["payment_installments"] = table["payment_installments"].fillna(1)
    for c in ("n_items", "total_price", "total_freight"):
        table[c] = table[c].fillna(0)
    table["distance_km"] = table["distance_km"].fillna(table["distance_km"].median())
    table = table.dropna(subset=["customer_state", "promised_lead_days"])
    notes["rows_modelled"] = len(table)
    notes["late_rate_overall"] = round(float(table["late"].mean()), 4)
    return table, notes


def _pipeline(estimator) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False), CATEGORICAL),
        ]
    )
    return Pipeline([("pre", pre), ("clf", estimator)])


def _metrics(y: np.ndarray, prob: np.ndarray, seed: int, n_boot: int) -> dict:
    return {
        "n": len(y),
        "positives": int(y.sum()),
        "prevalence": round(float(y.mean()), 4),
        "auroc": round(float(roc_auc_score(y, prob)), 4),
        "auroc_ci_bootstrap": [round(v, 4) for v in bootstrap_ci(roc_auc_score, y, prob, n_boot, seed)],
        "auprc": round(float(average_precision_score(y, prob)), 4),
        "auprc_ci_bootstrap": [round(v, 4) for v in bootstrap_ci(average_precision_score, y, prob, n_boot, seed)],
        "brier": round(float(brier_score_loss(y, prob)), 4),
    }


def _threshold_sweep(y: np.ndarray, prob: np.ndarray) -> list[dict]:
    """Decision 'expedite if p >= p*', with p* = C_expedite / C_chargeback."""
    rows = []
    for ratio in (0.05, 0.1, 0.2, 0.3, 0.5):
        flag = prob >= ratio
        tp = int((flag & (y == 1)).sum())
        rows.append(
            {
                "cost_ratio_expedite_over_chargeback": ratio,
                "threshold": ratio,
                "flagged_share": round(float(flag.mean()), 4),
                "recall_of_late_orders": round(float(tp / max(1, y.sum())), 4),
                "precision_among_flagged": round(float(tp / max(1, flag.sum())), 4),
            }
        )
    return rows


def _reliability(y: np.ndarray, prob: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in itertools.pairwise(edges):
        mask = (prob >= lo) & (prob < hi if hi < 1 else prob <= hi)
        if mask.sum() == 0:
            continue
        out.append(
            {
                "bin": [round(float(lo), 2), round(float(hi), 2)],
                "n": int(mask.sum()),
                "mean_pred": round(float(prob[mask].mean()), 4),
                "observed_rate": round(float(y[mask].mean()), 4),
            }
        )
    return out


def _pr_curve(y: np.ndarray, prob: np.ndarray, n_points: int = 30) -> list[dict]:
    """Precision / recall / flagged share at evenly spaced thresholds (for the figure)."""
    rows = []
    for thr in np.linspace(0.02, 0.6, n_points):
        flag = prob >= thr
        tp = int((flag & (y == 1)).sum())
        rows.append(
            {
                "threshold": round(float(thr), 4),
                "flagged_share": round(float(flag.mean()), 4),
                "recall": round(float(tp / max(1, y.sum())), 4),
                "precision": round(float(tp / max(1, flag.sum())), 4) if flag.sum() else None,
            }
        )
    return rows


def run(data_dir: Path, seed: int = 0, n_boot: int = 500) -> dict:
    paths = download(data_dir)
    table, notes = build_table(paths)
    train = table[table["order_purchase_timestamp"] < TRAIN_END]
    test = table[table["order_purchase_timestamp"] >= TRAIN_END]
    y_tr, y_te = train["late"].to_numpy(), test["late"].to_numpy()
    results: dict = {
        "provenance": {
            "source": "Olist Brazilian E-Commerce Public Dataset (Kaggle olistbr/brazilian-ecommerce), downloaded from the Hub mirror "  # noqa: E501
            + MIRROR,
            "licence_note": "Olist is published on Kaggle under CC BY-NC-SA 4.0 (Kaggle page); used here non-commercially with attribution; the mirror carries no licence statement of its own",  # noqa: E501
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed": seed,
            "split": f"time-based: train purchases < {TRAIN_END.date()}, test purchases >= {TRAIN_END.date()} and <= cutoff",  # noqa: E501
            "label": "late = delivered_customer_date > estimated_delivery_date (day granularity), delivered orders only",  # noqa: E501
            "censoring_guard": f"orders purchased within {CENSOR_DAYS} days of the last purchase are excluded",
            "features_known_at_checkout": NUMERIC + CATEGORICAL,
        },
        "data": {
            **notes,
            "train_n": len(train),
            "test_n": len(test),
            "train_late_rate": round(float(y_tr.mean()), 4),
            "test_late_rate": round(float(y_te.mean()), 4),
        },
        "models": {},
    }
    prevalence = np.full(len(y_te), y_tr.mean())
    results["models"]["prevalence_constant"] = {
        "n": len(y_te),
        "auroc": 0.5,
        "auprc": round(float(y_te.mean()), 4),
        "brier": round(float(brier_score_loss(y_te, prevalence)), 4),
        "note": "constant probability equal to the training late rate; AUPRC of a random ranker equals prevalence",
    }
    for name, est in (
        ("logistic_regression", LogisticRegression(max_iter=3000, class_weight=None)),
        (
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(
                random_state=seed, learning_rate=0.05, max_iter=300, early_stopping=True, validation_fraction=0.1
            ),
        ),
    ):
        pipe = _pipeline(est).fit(train[NUMERIC + CATEGORICAL], y_tr)
        prob = pipe.predict_proba(test[NUMERIC + CATEGORICAL])[:, 1]
        entry = _metrics(y_te, prob, seed, n_boot)
        entry["threshold_sweep"] = _threshold_sweep(y_te, prob)
        entry["reliability"] = _reliability(y_te, prob)
        entry["pr_curve"] = _pr_curve(y_te, prob)
        results["models"][name] = entry
    # single-feature reference: the promised lead time alone
    lead = test["promised_lead_days"].to_numpy(dtype=float)
    results["models"]["promised_lead_days_only"] = {
        "auroc": round(float(roc_auc_score(y_te, -lead)), 4),
        "auprc": round(float(average_precision_score(y_te, -lead)), 4),
        "note": "ranking by shorter promised lead time only (no model); a sanity reference for what one checkout-time field carries",  # noqa: E501
    }
    return results


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "olist"))
    parser.add_argument("--out", default=str(ROOT / "results" / "olist_positive_control.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=500)
    args = parser.parse_args(argv)
    results = run(Path(args.data_dir), seed=args.seed, n_boot=args.n_boot)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(results, handle, indent=1, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({k: v for k, v in results["data"].items()}, indent=1))
    for name, m in results["models"].items():
        print(f"  {name:28s} auroc {m.get('auroc')}  auprc {m.get('auprc')}  brier {m.get('brier', 'n/a')}")
    print(f"wrote {args.out}")
    return results


if __name__ == "__main__":
    main()
