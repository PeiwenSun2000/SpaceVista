"""
SpaceVista-Bench metrics for bench_all_clean.json format.

Computes per-category (Indoor/Outdoor/TinyTabletop/Tabletop),
per-original_type, and overall scores.
"""

import json
from collections import defaultdict

import numpy as np
import pandas as pd

from .common import (
    exact_match,
    extract_answer,
    fuzzy_matching,
    fuzzy_matching_num,
    mra_5_95_5,
    to_float,
)


def _score_one(doc: dict) -> dict:
    """Compute per-sample metric based on answer_type."""
    pred_raw = doc["predicted_answer"]
    gt = doc["ground_truth"]
    answer_type = doc.get("answer_type", "")

    pred = extract_answer(pred_raw)

    if answer_type == "mc":
        doc["score"] = exact_match(fuzzy_matching(pred), gt)
    elif answer_type == "numerical":
        try:
            doc["score"] = mra_5_95_5(
                to_float(fuzzy_matching_num(pred)), to_float(gt)
            )
        except (TypeError, ValueError):
            doc["score"] = 0.0
    elif answer_type == "text":
        doc["score"] = exact_match(extract_answer(pred), gt)
    else:
        doc["score"] = 0.0

    return doc


def compute_metrics(jsonl_path: str) -> dict:
    """
    Compute metrics from a SpaceVista-Bench results JSONL file.

    Returns a dict with:
      - per_category: {category: {mc_accuracy, num_MRA, text_accuracy, count, overall}}
      - per_type: {original_type: {score, count}}
      - overall: float
    """
    docs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                doc = _score_one(json.loads(line))
                docs.append(doc)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: skipping malformed line — {e}")

    if not docs:
        print("No valid results found.")
        return {"overall": 0.0}

    df = pd.DataFrame(docs)
    output = {}

    # --- Per-category breakdown ---
    per_category = {}
    for category in ["Indoor", "Outdoor", "Tabletop", "TinyTabletop"]:
        cat_df = df[df["category"] == category]
        if cat_df.empty:
            continue
        cat_result = {"count": len(cat_df)}

        mc_df = cat_df[cat_df["answer_type"] == "mc"]
        if not mc_df.empty:
            cat_result["mc_accuracy"] = float(mc_df["score"].mean())
            cat_result["mc_count"] = len(mc_df)

        num_df = cat_df[cat_df["answer_type"] == "numerical"]
        if not num_df.empty:
            cat_result["num_MRA"] = float(num_df["score"].mean())
            cat_result["num_count"] = len(num_df)

        text_df = cat_df[cat_df["answer_type"] == "text"]
        if not text_df.empty:
            cat_result["text_accuracy"] = float(text_df["score"].mean())
            cat_result["text_count"] = len(text_df)

        cat_result["overall"] = float(cat_df["score"].mean())
        per_category[category] = cat_result

    output["per_category"] = per_category

    # --- Per-original_type breakdown ---
    per_type = {}
    for otype, grp in df.groupby("original_type"):
        per_type[otype] = {
            "score": float(grp["score"].mean()),
            "count": len(grp),
        }
    output["per_type"] = per_type

    # --- Overall ---
    output["overall"] = float(df["score"].mean())
    output["total_count"] = len(df)

    # --- Print summary table ---
    _print_summary(output)

    return output


def _print_summary(output: dict):
    """Print a formatted summary of the evaluation results."""
    print("\n" + "=" * 70)
    print("SpaceVista-Bench Evaluation Results")
    print("=" * 70)

    # Per-category table
    print(f"\n{'Category':<15} {'Count':>6} {'MC Acc':>8} {'Num MRA':>8} {'Text Acc':>9} {'Overall':>8}")
    print("-" * 60)
    for cat in ["Indoor", "Outdoor", "Tabletop", "TinyTabletop"]:
        cr = output.get("per_category", {}).get(cat)
        if cr is None:
            continue
        mc = f"{cr.get('mc_accuracy', 0):.4f}" if "mc_accuracy" in cr else "  N/A "
        nm = f"{cr.get('num_MRA', 0):.4f}" if "num_MRA" in cr else "  N/A "
        tx = f"{cr.get('text_accuracy', 0):.4f}" if "text_accuracy" in cr else "   N/A  "
        ov = f"{cr['overall']:.4f}"
        print(f"{cat:<15} {cr['count']:>6} {mc:>8} {nm:>8} {tx:>9} {ov:>8}")

    # Overall
    print("-" * 60)
    print(f"{'OVERALL':<15} {output.get('total_count', 0):>6} {'':>8} {'':>8} {'':>9} {output['overall']:.4f}")

    # Per-type table
    print(f"\n{'Question Type':<35} {'Count':>6} {'Score':>8}")
    print("-" * 52)
    per_type = output.get("per_type", {})
    for otype in sorted(per_type.keys()):
        info = per_type[otype]
        print(f"{otype:<35} {info['count']:>6} {info['score']:.4f}")

    print("=" * 70)
