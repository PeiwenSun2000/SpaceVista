"""STI-Bench scoring: per-task accuracy, SR sub-task mean, and overall."""

import json
import logging
import re

import numpy as np
import pandas as pd

from .common import exact_match, extract_answer, fuzzy_matching_num, to_float

logger = logging.getLogger(__name__)

# Spatial Reasoning sub-tasks that form their own sub-score
SR_SUB_TASKS = [
    "Dimensional Measurement",
    "Displacement & Path Length",
    "Ego-Centric Orientation",
    "Spatial Relation",
    "Speed & Acceleration",
    "Trajectory Description",
]

PROMPT_TEMPLATES = {
    "default": {
        "pre_prompt": "Question: {Question}\n",
        "mca_post_prompt": "Answer with the option's letter from the given choices directly.",
        "na_post_prompt": "Please answer the question using a numerical value (e.g., 42 or 3.1).",
    },
    "thinking": {
        "pre_prompt": (
            "Question: {Question}\n"
            "Please think about this question as if you were a human pondering deeply. "
            "Engage in an internal dialogue using expressions such as 'let me think', 'wait', "
            "'Hmm', 'oh, I see', 'let\\'s break it down', etc. "
            "It's encouraged to include self-reflection or verification in the reasoning process. "
            "Provide your detailed reasoning between the <think> </think> tags, and then give "
            "your final answer between the <answer> </answer> tags."
        ),
        "mca_post_prompt": "Please provide only the single option letter (e.g., A, B, C, D, E) within the <answer> </answer> tags.",
        "na_post_prompt": "Please provide the numerical value (e.g., 42 or 3.1) within the <answer> </answer> tags.",
        "special_post_prompt": "First output the thinking process in <think> </think> tags and then output the answer in <answer> </answer> tags.",
    },
}


def _normalize_choice(text) -> str:
    """Extract a single capital letter A-E from free-form text. Returns None on failure."""
    if text is None:
        return None
    patterns = [
        r'\(([A-E])\)',
        r'Ans\s*=\s*[\'"]?([A-E])[\'"]?',
        r'Answer\s*[:=]\s*([A-E])',
        r'Option\s+([A-E])',
        r'\b([A-E])\s*(?:is|was)\s*correct',
        r'\b([A-E])[\.\)]\s*$',
        r'^\s*([A-E])\s*$',
    ]
    for pat in patterns:
        m = re.search(pat, str(text), re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    head = re.match(r'^\s*([A-E])\b', str(text).strip(), re.IGNORECASE)
    return head.group(1).upper() if head else None


def _process(doc: dict, mode: str = "thinking") -> dict:
    pred = doc.get("predicted_answer", "")
    gt = doc.get("ground_truth", "")

    if mode == "thinking" and isinstance(pred, str) and "<answer>" in pred:
        pred = extract_answer(pred)

    norm = _normalize_choice(pred) or str(pred).strip()
    doc["predicted_answer"] = norm
    doc["accuracy"] = exact_match(norm, gt)
    return doc


def _aggregate(results: list) -> dict:
    df = pd.DataFrame(results) if results else pd.DataFrame()
    output = {}

    if df.empty or "task" not in df.columns or "accuracy" not in df.columns:
        output["sr_sub_accuracy"] = 0.0
        output["overall_accuracy"] = 0.0
        return output

    for task, idx in df.groupby("task").groups.items():
        output[task] = float(df.iloc[idx]["accuracy"].mean())

    sr_vals = [output[t] for t in SR_SUB_TASKS if t in output]
    output["sr_sub_accuracy"] = float(np.mean(sr_vals)) if sr_vals else 0.0

    all_vals = list(output.values())
    output["overall_accuracy"] = float(np.mean(all_vals)) if all_vals else 0.0

    logger.info(f"STI-Bench results: {output}")
    return output


def calculate_json(jsonl_file: str, mode: str = "thinking") -> dict:
    """Score an STI-Bench prediction JSONL. Returns per-task accuracy, SR sub-score, overall."""
    if not jsonl_file or not __import__("os").path.exists(jsonl_file):
        print(f"Error: file not found: {jsonl_file}")
        return {}
    docs = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(_process(json.loads(line), mode=mode))
            except json.JSONDecodeError as e:
                print(f"Warning: skipping malformed line — {e}")
    return _aggregate(docs)


# --- Prompt assembly helpers (useful for external callers) ---

def build_mca_prompt(question_text: str, candidates: dict,
                     prompt_type: str = "default", time_span=None) -> str:
    """Assemble a multiple-choice prompt string."""
    tpl = PROMPT_TEMPLATES.get(prompt_type, PROMPT_TEMPLATES["default"])
    prefix = f"From {time_span[0]} seconds to {time_span[1]} seconds. " if time_span else ""
    cands = "\n".join(f"{k} {v}" for k, v in candidates.items()) if isinstance(candidates, dict) else ""
    q = prefix + question_text + (("\n" + cands) if cands else "")
    return tpl["pre_prompt"].format(Question=q) + "\n" + tpl["mca_post_prompt"]


def build_na_prompt(question_text: str, prompt_type: str = "default", time_span=None) -> str:
    """Assemble a numerical-answer prompt string."""
    tpl = PROMPT_TEMPLATES.get(prompt_type, PROMPT_TEMPLATES["default"])
    prefix = f"From {time_span[0]} seconds to {time_span[1]} seconds. " if time_span else ""
    return tpl["pre_prompt"].format(Question=prefix + question_text) + "\n" + tpl["na_post_prompt"]
