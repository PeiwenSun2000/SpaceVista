"""SpaceVista-Bench scoring: per-task-type accuracy/MRA and overall."""

import json

import pandas as pd

from .common import (
    exact_match, extract_answer, fuzzy_matching, fuzzy_matching_num,
    to_float, mra_5_95_5,
)

# Fine-grained question categories stored in result['question_type']
MCA_QUESTION_TYPES = [
    "EXISTENCE",
    "COMPARISON_HEIGHT",
    "COMPARISON_WIDTH",
    "POSITION_ACROSS_FRAME",
    "APPEAR_ORDER",
    "CAMERA_MOVING",
    "SPATIAL_RELATION",
    "OBJECT_MATCHING",
]

NA_QUESTION_TYPES = [
    "COUNTING",
    "DEPTH_all",
    "DEPTH",
    "HEIGHT",
    "WIDTH",
    "OBJECT_DISTANCE",
    "OBJECT_ROTATION",
    "ROOM_SIZE",
    "OBJECT_RELATIVE_DISTANCE",
    "CAMERA_ROTATION",
]

WORST_CASE = {"accuracy": 0.0, "MRA:.5:.95:.05": 0.0}


def _process(doc: dict) -> dict:
    """Compute per-sample metric. Expects doc['question_type'] = row['TaskType']."""
    qt = doc["question_type"]
    pred = extract_answer(doc["predicted_answer"])
    if qt in MCA_QUESTION_TYPES:
        doc["predicted_answer"] = pred
        doc["accuracy"] = exact_match(fuzzy_matching(pred), doc["ground_truth"])
    elif qt in NA_QUESTION_TYPES:
        doc["predicted_answer"] = pred
        try:
            doc["MRA:.5:.95:.05"] = mra_5_95_5(
                to_float(fuzzy_matching_num(pred)), to_float(doc["ground_truth"])
            )
        except TypeError:
            doc["MRA:.5:.95:.05"] = WORST_CASE["MRA:.5:.95:.05"]
    else:
        raise ValueError(f"Unknown SpaceVista question type: {qt!r}")
    return doc


def _aggregate(results: list) -> dict:
    """
    Groups by doc['task_type'] (row['Question Type']: 'multiple choice' / 'regression').
    Computes mean accuracy and mean MRA per group, then overall across all values.
    """
    if not results:
        return {"overall_accuracy": 0.0}

    df = pd.DataFrame(results)
    output = {}

    if "task_type" in df.columns:
        for task, idx in df.groupby("task_type").groups.items():
            per_task = df.iloc[idx]
            if "accuracy" in per_task.columns:
                val = per_task["accuracy"].dropna().mean()
                if not pd.isna(val):
                    output[f"{task}_accuracy"] = val
            if "MRA:.5:.95:.05" in per_task.columns:
                val = per_task["MRA:.5:.95:.05"].dropna().mean()
                if not pd.isna(val):
                    output[f"{task}_MRA"] = val

    # Overall: mean of all individual per-sample metric values
    all_vals = []
    if "accuracy" in df.columns:
        all_vals.extend(df["accuracy"].dropna().tolist())
    if "MRA:.5:.95:.05" in df.columns:
        all_vals.extend(df["MRA:.5:.95:.05"].dropna().tolist())
    output["overall_accuracy"] = float(pd.Series(all_vals).mean()) if all_vals else 0.0

    print("SpaceVista Evaluation Results:")
    print(json.dumps(output, indent=2))
    return output


def calculate_json(jsonl_file: str) -> dict:
    """Score a SpaceVista-Bench prediction JSONL. Returns per-task metrics and overall_accuracy."""
    docs = []
    try:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    docs.append(_process(json.loads(line)))
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: skipping malformed line — {e}")
    except FileNotFoundError:
        print(f"Error: file not found: {jsonl_file}")
        return {}
    return _aggregate(docs)
