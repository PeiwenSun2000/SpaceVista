"""VSI-Bench scoring: per-question-type accuracy and MRA, plus overall."""

import json

import numpy as np
import pandas as pd

from .common import (
    exact_match, extract_answer, fuzzy_matching, fuzzy_matching_num,
    to_float, mra_5_95_5,
)

MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]

NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]

WORST_CASE = {"accuracy": 0.0, "MRA:.5:.95:.05": 0.0}


def _process(doc: dict) -> dict:
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
        raise ValueError(f"Unknown VSI-Bench question type: {qt!r}")
    return doc


def _aggregate(results: list) -> dict:
    df = pd.DataFrame(results)
    output = {}

    for qt, idx in df.groupby("question_type").groups.items():
        per_qt = df.iloc[idx]
        if qt in MCA_QUESTION_TYPES:
            output[f"{qt}_accuracy"] = per_qt["accuracy"].mean()
        elif qt in NA_QUESTION_TYPES:
            output[f"{qt}_MRA:.5:.95:.05"] = per_qt["MRA:.5:.95:.05"].mean()

    # Collapse easy/medium/hard direction subtypes into one score
    direction_keys = [
        "object_rel_direction_easy_accuracy",
        "object_rel_direction_medium_accuracy",
        "object_rel_direction_hard_accuracy",
    ]
    present = [k for k in direction_keys if k in output]
    if present:
        output["object_rel_direction_accuracy"] = np.mean([output.pop(k) for k in present])

    output["overall_accuracy"] = np.mean(list(output.values()))
    return output


def calculate_json(jsonl_file: str) -> dict:
    """Score a VSI-Bench prediction JSONL. Returns per-type metrics and overall_accuracy."""
    docs = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(_process(json.loads(line)))
    result = _aggregate(docs)
    print(json.dumps(result, indent=2))
    return result
