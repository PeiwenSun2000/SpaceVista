import re
from functools import partial
import os
import json
import pandas as pd
import numpy as np

# Original VSiBench constants
MCA_QUESTION_TYPES = [
    'Positional Relationship (Cam.–Obj.)', 
    'MSR', 'Motion (Cam.)', 
    'Positional Relationship (Cam.–Reg.)', 
    'Attribute (Appr.)', 
    'Positional Relationship (Obj.–Reg.)',
    'Positional Relationship (Reg.–Reg.)', 
    'Motion (Obj.)', 
    'Positional Relationship (Obj.–Obj.)', 
    'Positional Relationship (Cam.–Cam.)', 
    'Attribute (Meas.)'
]
NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]

METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}

METRICS_FOR_NA = {
    "MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
}

WORST_CASE_FOR_METRICS = {
    "accuracy": 0.,
    "MRA:.5:.95:.05": 0.,
}
EXAMPLE_MAP={"table":[[0,3],[5,7]],"chair":[[9,3]],"window":[[6,5]]}
COGMAP_TEMPLATE=(
        "Question: {Question}\n"
        "Please think about this question as if you were a human pondering deeply. "
        "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
        "It's encouraged to include self-reflection or verification in the reasoning process.\n"
        "If generating a cognitive map for the video can help you answer the question, you could follow the below steps to generate a cognitive map in <map> </map> tags\n"
        "[Steps] Identify specific objects within the **video scene**, understand the spatial arrangement of the scene, and estimate the center point of each object, assuming the entire scene is represented by a 10x10 grid. These information should be summarized in <map> </map> tags.\n"
        "[Rule]1. We provide the categories to care about in this scene: {object_list}. Focus ONLY on these categories for the entire video scene.\n2. Estimate the center location of each instance within the provided categories, assuming the entire scene is represented by a 10x10 grid, considering the information from all frames.\n3. If a category contains multiple instances across all frames, include all of them.\n"
        "Present the map using dict format. Here is an example: <map>{map_example}</map>.\n"
        "If you generate a cognitive map, please put it in <map> </map> tags. Provide your detailed reasoning process between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."
    )
OBJECT_LIST=[
    "ceiling light", "trash can", "bed", "heater", "closet", "pillow", "backpack", "chair", "refrigerator",
    "tv", "nightstand", "keyboard", "computer tower", "coat hanger", "table", "trash bin", "whiteboard",
    "monitor", "sofa", "clock", "computer mouse", "radiator", "telephone"
]
PROMPT_TEMPLATES = {
    "default": {
        "pre_prompt": "Question: {Question}\n",
        "mca_post_prompt": "Answer with the option's letter from the given choices directly.",
        "na_post_prompt": "Please answer the question using a numerical value (e.g., 42 or 3.1)."
    },
    "thinking":
    {   "pre_prompt": COGMAP_TEMPLATE,
        "mca_post_prompt": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "na_post_prompt": " Please provide the numerical value (e.g., 42 or 3.1) within the <answer> </answer> tags.",
    }
}

# --- Utility Functions (shared across evaluators) ---
def exact_match(pred, target):
    """
    Performs a case-insensitive comparison of prediction and target strings.
    """
    if pred is None or target is None:
        return 0.
    return 1. if str(pred).strip().lower() == str(target).strip().lower() else 0.

def abs_dist_norm(pred, target):
    return abs(pred - target) / target

def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()

def to_float(pred):
    try:
        pred = float(pred)
    except (ValueError, TypeError):
        pred = None
    return pred

def fuzzy_matching_num(pred):
    pred = pred.strip().lower() 
    number_words = {
        'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
        'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
        'zero': '0', 'a': '1', 'an': '1'  
    }
    for word, digit in number_words.items():
        if re.search(r'\b' + word + r'\b', pred):  
            return digit  
    number_match = re.search(r'(\d+(\.\d+)?)', pred)  
    if number_match:
        return number_match.group(1)  
    return "None"  

def extract_answer(text):
    """
    Extracts content from within <answer>...</answer> tags.
    """
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text # Return original text if no tag is found

def fuzzy_matching(pred):
    """
    Normalizes single-letter multiple-choice answers (e.g., "A.", "B ") to "A", "B".
    """
    # Search for a single character A, B, C, or D, possibly followed by a dot.
    match = re.search(r'^\s*([A-D])\b', pred.strip(), re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return pred.strip()

# --- VSiBench / MMSI Bench Evaluation Functions ---
def vsibench_process_results(doc): 
    if doc['question_type'] in MCA_QUESTION_TYPES:
        for key, value in METRICS_FOR_MCA.items():
            doc['predicted_answer'] = extract_answer(doc['predicted_answer'])
            doc[key] = eval(value)(fuzzy_matching(doc['predicted_answer']), doc['ground_truth'])
    elif doc['question_type'] in NA_QUESTION_TYPES:
        for key, value in METRICS_FOR_NA.items():
            doc['predicted_answer'] = extract_answer(doc['predicted_answer'])
            try:
                doc[key] = eval(value)(to_float(fuzzy_matching_num(doc['predicted_answer'])), to_float(doc['ground_truth']))
            except TypeError:
                doc[key] = WORST_CASE_FOR_METRICS[key]
    else:
        # Gracefully handle unknown question types
        print(f"Warning: Unknown question type '{doc['question_type']}' for VSiBench. Skipping metrics calculation.")
    return doc 

def mmsi_process_results(doc): 
    for key, value in METRICS_FOR_MCA.items():
        doc['predicted_answer'] = extract_answer(doc['predicted_answer'])
        doc[key] = eval(value)(fuzzy_matching(doc['predicted_answer']), doc['ground_truth'])
    return doc 

def caluculate_json(json_file):
    doc=[]
    with open(json_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Note: You would choose one of these process functions based on the dataset
            # doc.append(vsibench_process_results(json.loads(line))) 
            doc.append(mmsi_process_results(json.loads(line)))
        doc = vsibench_aggregate_results(doc)
    return doc

def vsibench_aggregate_results(results):
    results = pd.DataFrame(results)
    output = {}
    for question_type, question_type_indexes in results.groupby('question_type').groups.items():
        per_question_type = results.iloc[question_type_indexes]
        if question_type in MCA_QUESTION_TYPES:
            for metric in METRICS_FOR_MCA.keys():
                output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
        elif question_type in NA_QUESTION_TYPES:
            for metric in METRICS_FOR_NA.keys():
                output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
        else:
            print(f"Warning: Unknown question type '{question_type}' during VSiBench aggregation.")
    
    if output:
        output['overall_accuracy'] = sum(output.values()) / len(output)
    else:
        output['overall_accuracy'] = 0.0
    print(output)
    return output

# --- NEW: SpaceVista Evaluation Functions ---

def spacevista_process_results(doc):
    """
    Processes a single result entry for the SpaceVista dataset.
    It calculates accuracy using a general exact match.
    """
    # The prompt for SpaceVista also uses <answer> tags
    predicted_answer = extract_answer(doc['predicted_answer'])
    ground_truth = doc['ground_truth']
    
    # Use fuzzy_matching for single-letter answers, otherwise use the extracted text
    final_prediction = fuzzy_matching(predicted_answer)

    # Calculate accuracy using a simple exact match
    doc['accuracy'] = exact_match(final_prediction, ground_truth)
    return doc

def spacevista_aggregate_results(results):
    """
    Aggregates results for the SpaceVista dataset, grouping by 'task_type'.
    """
    if not results:
        print("Warning: No results to aggregate for SpaceVista.")
        return {"overall_accuracy": 0.0}

    results_df = pd.DataFrame(results)
    output = {}

    # Group by 'task_type' which is the high-level category in SpaceVista
    if 'task_type' in results_df.columns:
        for task_type, group_indexes in results_df.groupby('task_type').groups.items():
            per_task_type_df = results_df.iloc[group_indexes]
            # Calculate mean accuracy for this task type
            mean_accuracy = per_task_type_df['accuracy'].mean()
            output[f"{task_type}_accuracy"] = mean_accuracy
    else:
        print("Warning: 'task_type' column not found in results. Cannot provide a breakdown.")

    # Calculate the overall accuracy across all entries
    if 'accuracy' in results_df.columns:
        output['overall_accuracy'] = results_df['accuracy'].mean()
    else:
        print("Error: 'accuracy' column not calculated. Overall accuracy is 0.")
        output['overall_accuracy'] = 0.0

    print("SpaceVista Evaluation Results:")
    print(json.dumps(output, indent=2))
    return output

def caluculate_json_spacevista(json_file):
    """
    Top-level function to calculate and aggregate results from a SpaceVista JSONL file.
    """
    processed_docs = []
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_docs.append(spacevista_process_results(data))
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from line: {line.strip()}")
                except KeyError as e:
                    print(f"Warning: Missing key {e} in line: {line.strip()}")

        aggregated_results = spacevista_aggregate_results(processed_docs)
        return aggregated_results
    except FileNotFoundError:
        print(f"Error: The file {json_file} was not found.")
        return {}