"""
Shared metric utilities used across all benchmark evaluators.

Each function is the best-available version across the original util files:
- null-safe where inputs may be None
- negative-number-aware fuzzy numeric matching
- zero-denominator-safe relative distance
"""

import re
from functools import partial

import numpy as np


def exact_match(pred, target) -> float:
    """Case-insensitive exact string match. Returns 0.0 if either argument is None."""
    if pred is None or target is None:
        return 0.0
    return 1.0 if str(pred).strip().lower() == str(target).strip().lower() else 0.0


def extract_answer(text) -> str:
    """
    Extract content from <answer>...</answer> tags.
    Returns the original text unchanged if no tag is found.
    """
    if text is None:
        return text
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, str(text), re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


def fuzzy_matching(pred) -> str:
    """
    Normalize multiple-choice letter answers to a bare capital letter.
    Handles formats like 'A.', 'a)', 'A some text' → 'A'.
    Returns original string unchanged if no A-D letter is found at the start.
    """
    match = re.search(r'^\s*([A-D])\b', pred.strip(), re.IGNORECASE)
    return match.group(1).upper() if match else pred.strip()


def fuzzy_matching_num(pred) -> str:
    """
    Extract a numeric string from free-form text.
    Handles written-out numbers ('two', 'thirty'), bare digits, and negatives.
    Returns the string 'None' if no number is found.
    """
    if pred is None:
        return "None"
    s = str(pred).strip().lower()
    number_words = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
        'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
        'eighteen': '18', 'nineteen': '19', 'twenty': '20',
        'thirty': '30', 'forty': '40', 'fifty': '50', 'sixty': '60',
        'seventy': '70', 'eighty': '80', 'ninety': '90',
        'a': '1', 'an': '1',
    }
    for word, digit in number_words.items():
        if re.search(r'\b' + re.escape(word) + r'\b', s):
            return digit
    m = re.search(r'(-?\d+(\.\d+)?)', s)
    return m.group(1) if m else "None"


def to_float(pred):
    """Convert a string to float. Returns None on failure."""
    try:
        return float(pred)
    except (ValueError, TypeError):
        return None


def abs_dist_norm(pred, target) -> float:
    """Normalized absolute distance. Guards against zero denominator."""
    if target == 0.0:
        return abs(pred - target)
    return abs((pred - target) / target)


def mean_relative_accuracy(pred, target, start, end, interval) -> float:
    """Mean accuracy across a range of relative-error thresholds (MRA metric)."""
    num_pts = (end - start) / interval + 2
    thresholds = np.linspace(start, end, int(num_pts))
    return (abs_dist_norm(pred, target) <= 1 - thresholds).mean()


# Pre-built MRA variant used by most benchmarks
mra_5_95_5 = partial(mean_relative_accuracy, start=0.5, end=0.95, interval=0.05)
