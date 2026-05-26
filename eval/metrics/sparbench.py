"""SPAR-Bench scoring: multi-type metrics with Low/Middle/High skill groupings."""

import json
import logging
import math
import re

import numpy as np
import pandas as pd

from .common import exact_match, to_float, abs_dist_norm, mean_relative_accuracy
from functools import partial

logger = logging.getLogger(__name__)

mra_5_95_5 = partial(mean_relative_accuracy, start=0.5, end=0.95, interval=0.05)

MCA_QUESTION_TYPES = [
    "obj_spatial_relation_oo",
    "obj_spatial_relation_oc_mv",
    "obj_spatial_relation_oo_mv",
    "spatial_imagination_oc",
    "spatial_imagination_oo",
    "spatial_imagination_oc_mv",
    "spatial_imagination_oo_mv",
    "position_matching",
    "camera_motion_infer",
    "distance_infer_center_oo",
    "distance_infer_center_oo_mv",
]

NA_QUESTION_TYPES = [
    "depth_prediction_oc",
    "depth_prediction_oo",
    "distance_prediction_oc",
    "distance_prediction_oo",
    "depth_prediction_oc_mv",
    "depth_prediction_oo_mv",
    "distance_prediction_oo_mv",
    "distance_prediction_oc_mv",
]

SPECIAL_QUESTION_TYPES = ["view_change_infer"]

WORST_CASE = {"accuracy": 0.0, "MRA:.5:.95:.05": 0.0}

# Skill-level groupings for summary output
Low = NA_QUESTION_TYPES[:]
Middle = ["view_change_infer", "position_matching", "camera_motion_infer"]
High = [
    "obj_spatial_relation_oo", "obj_spatial_relation_oc_mv", "obj_spatial_relation_oo_mv",
    "spatial_imagination_oc", "spatial_imagination_oo", "spatial_imagination_oc_mv",
    "spatial_imagination_oo_mv", "distance_infer_center_oo", "distance_infer_center_oo_mv",
]


def _fuzzy_matching(pred: str) -> str:
    """Strip to first whitespace-delimited token (SPAR answers are not always single letters)."""
    return pred.split(" ")[0].rstrip(".").strip()


def _extract_answer(text: str) -> str:
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, str(text), re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _process_na(pred: str, task: str) -> float:
    """Extract the relevant number from a free-form NA answer."""
    numbers = re.findall(r'(?<!\^)\d+\.\d+|(?<!\^)\d+', pred)
    extracted = [float(n) if '.' in n else int(n) for n in numbers]
    mv_tasks = {
        "depth_prediction_oc_mv", "depth_prediction_oo_mv",
        "distance_prediction_oc_mv", "distance_prediction_oo_mv",
    }
    if task in mv_tasks:
        extracted = [extracted[-1]] if extracted else [-1]
    return extracted[0] if extracted else -1


def _parse_instruction(instruction: str) -> dict:
    return {k: float(v) for k, v in [item.split(":") for item in instruction.split(",")]}


def _compute_vci_metric(pred: str, answer: str) -> float:
    """View-change inference metric: mean MRA across 5 action dimensions."""
    action_order = [
        "move_right_left", "move_up_down", "move_forward_backward",
        "rotate_right_left", "rotate_up_down",
    ]
    try:
        pred_dict = _parse_instruction(pred)
        gt_dict = _parse_instruction(answer)
    except Exception:
        return 0.0

    pred_vals, gt_vals = [], []
    for pair in action_order:
        a, b = pair.split("_", 2)[1], pair.split("_", 2)[2]  # e.g. "right", "left"
        # Reconstruct keys from pair name
        if pair == "move_right_left":
            pred_vals.append(pred_dict.get("move_right", 0) - pred_dict.get("move_left", 0))
            gt_vals.append(gt_dict.get("move_right", 0) - gt_dict.get("move_left", 0))
        elif pair == "move_up_down":
            pred_vals.append(pred_dict.get("move_up", 0) - pred_dict.get("move_down", 0))
            gt_vals.append(gt_dict.get("move_up", 0) - gt_dict.get("move_down", 0))
        elif pair == "move_forward_backward":
            pred_vals.append(pred_dict.get("move_forward", 0) - pred_dict.get("move_backward", 0))
            gt_vals.append(gt_dict.get("move_forward", 0) - gt_dict.get("move_backward", 0))
        elif pair == "rotate_right_left":
            pred_vals.append(pred_dict.get("rotate_right", 0) - pred_dict.get("rotate_left", 0))
            gt_vals.append(gt_dict.get("rotate_right", 0) - gt_dict.get("rotate_left", 0))
        elif pair == "rotate_up_down":
            pred_vals.append(pred_dict.get("rotate_up", 0) - pred_dict.get("rotate_down", 0))
            gt_vals.append(gt_dict.get("rotate_up", 0) - gt_dict.get("rotate_down", 0))

    mra_vals = [mra_5_95_5(p, g) for p, g in zip(pred_vals, gt_vals)]
    return float(np.mean(mra_vals))


def _parse_cmi(text: str) -> list:
    pattern = r"\([0-9\.]+,[0-9\.]+\)|[0-9\.]+"
    matches = re.findall(pattern, text)
    if len(matches) < 2:
        if len(matches) == 1 and "(" in matches[0]:
            matches.append("0.0")
        elif len(matches) == 1:
            matches.insert(0, "(0.0,0.0)")
    result = []
    for m in matches:
        if "(" in m:
            n1, n2 = m.strip("()").split(",")
            result.extend([float(n1), float(n2)])
        else:
            result.append(float(m))
    return result


def _process(doc: dict) -> dict:
    task = doc["task"]
    pred = _extract_answer(doc["predicted_answer"])
    doc["predicted_answer"] = pred

    if task in MCA_QUESTION_TYPES:
        doc["accuracy"] = exact_match(_fuzzy_matching(pred), doc["ground_truth"])
    elif task in NA_QUESTION_TYPES:
        try:
            doc["MRA:.5:.95:.05"] = mra_5_95_5(
                to_float(_process_na(pred, task)), to_float(doc["ground_truth"])
            )
        except Exception:
            doc["MRA:.5:.95:.05"] = WORST_CASE["MRA:.5:.95:.05"]
    elif task in SPECIAL_QUESTION_TYPES:
        if task == "view_change_infer":
            try:
                doc["vci_metric"] = _compute_vci_metric(pred, doc["ground_truth"])
            except Exception:
                doc["vci_metric"] = 0.0
    else:
        raise ValueError(f"Unknown SPAR-Bench task type: {task!r}")
    return doc


def _aggregate(results: list) -> dict:
    df = pd.DataFrame(results)
    overall = {}

    for task, idx in df.groupby("task").groups.items():
        per_task = df.iloc[idx]
        if task in MCA_QUESTION_TYPES:
            overall[f"{task}_accuracy"] = per_task["accuracy"].mean()
        elif task in NA_QUESTION_TYPES:
            overall[f"{task}_MRA:.5:.95:.05"] = per_task["MRA:.5:.95:.05"].mean()
        elif task == "view_change_infer":
            overall[f"{task}_vci_metric"] = per_task["vci_metric"].mean()

    overall["overall_accuracy"] = sum(overall.values()) / len(overall) if overall else 0.0

    # Per-image-type breakdown
    by_img_type = {}
    if "image_type" in df.columns:
        for img_type, img_group in df.groupby("image_type"):
            img_group = img_group.reset_index(drop=True)
            img_out = {}
            for task, idx in img_group.groupby("task").groups.items():
                per_task = img_group.iloc[idx]
                if task in MCA_QUESTION_TYPES:
                    img_out[f"{task}_accuracy"] = per_task["accuracy"].mean()
                elif task in NA_QUESTION_TYPES:
                    img_out[f"{task}_MRA:.5:.95:.05"] = per_task["MRA:.5:.95:.05"].mean()
                elif task == "view_change_infer":
                    img_out[f"{task}_vci_metric"] = per_task["vci_metric"].mean()
            img_out["overall_accuracy"] = (
                sum(img_out.values()) / len(img_out) if img_out else 0.0
            )
            by_img_type[img_type] = img_out

    # Skill-level summaries (overall)
    def _skill_mean(skill_tasks, data):
        vals = [data[f"{t}_accuracy"] for t in skill_tasks if f"{t}_accuracy" in data]
        vals += [data[f"{t}_MRA:.5:.95:.05"] for t in skill_tasks if f"{t}_MRA:.5:.95:.05" in data]
        vals += [data[f"{t}_vci_metric"] for t in skill_tasks if f"{t}_vci_metric" in data]
        return float(np.mean(vals)) if vals else 0.0

    overall["Low"] = _skill_mean(Low, overall)
    overall["Middle"] = _skill_mean(Middle, overall)
    overall["High"] = _skill_mean(High, overall)

    for img_type, img_out in by_img_type.items():
        img_out["Low"] = _skill_mean(Low, img_out)
        img_out["Middle"] = _skill_mean(Middle, img_out)
        img_out["High"] = _skill_mean(High, img_out)

    output = {"overall": overall, "by_img_type": by_img_type}
    logger.info(f"SPAR-Bench results: {output}")
    return output


def calculate_json(jsonl_file: str) -> dict:
    """Score a SPAR-Bench prediction JSONL. Returns overall and per-image-type metrics."""
    docs = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(_process(json.loads(line)))
    return _aggregate(docs)
