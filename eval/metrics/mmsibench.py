"""MMSI-Bench scoring: per-question-type accuracy and overall."""

import json

import pandas as pd

from .common import exact_match, extract_answer, fuzzy_matching

# All MMSI-Bench question types are multiple-choice
MCA_QUESTION_TYPES = [
    "Positional Relationship (Cam.–Obj.)",
    "MSR",
    "Motion (Cam.)",
    "Positional Relationship (Cam.–Reg.)",
    "Attribute (Appr.)",
    "Positional Relationship (Obj.–Reg.)",
    "Positional Relationship (Reg.–Reg.)",
    "Motion (Obj.)",
    "Positional Relationship (Obj.–Obj.)",
    "Positional Relationship (Cam.–Cam.)",
    "Attribute (Meas.)",
]


def _process(doc: dict) -> dict:
    pred = extract_answer(doc["predicted_answer"])
    doc["predicted_answer"] = pred
    doc["accuracy"] = exact_match(fuzzy_matching(pred), doc["ground_truth"])
    return doc


def _aggregate(results: list) -> dict:
    df = pd.DataFrame(results)
    output = {}

    for qt, idx in df.groupby("question_type").groups.items():
        per_qt = df.iloc[idx]
        output[f"{qt}_accuracy"] = per_qt["accuracy"].mean()

    if output:
        output["overall_accuracy"] = sum(output.values()) / len(output)
    else:
        output["overall_accuracy"] = 0.0

    print(json.dumps(output, indent=2))
    return output


def calculate_json(jsonl_file: str) -> dict:
    """Score an MMSI-Bench prediction JSONL. Returns per-type accuracy and overall_accuracy."""
    docs = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(_process(json.loads(line)))
    return _aggregate(docs)
