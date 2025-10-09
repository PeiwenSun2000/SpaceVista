
import os
from pathlib import Path
import yaml
from loguru import logger as eval_logger
from functools import partial
import numpy as np
import pandas as pd
import re
import math
import json

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
    "distance_infer_center_oo_mv"
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

SPECIAL_QUESTION_TYPES = [
    "view_change_infer",
]

METRICS_FOR_NA = {
    "MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
}

METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}

Low = [
    "depth_prediction_oc",
    "depth_prediction_oo",
    "distance_prediction_oc",
    "distance_prediction_oo",
    "depth_prediction_oc_mv",
    "depth_prediction_oo_mv",
    "distance_prediction_oo_mv",
    "distance_prediction_oc_mv",  
]

Middle = [
    "view_change_infer",
    "position_matching",
    "camera_motion_infer",
]

High = [
    "obj_spatial_relation_oo",
    "obj_spatial_relation_oc_mv",
    "obj_spatial_relation_oo_mv",
    "spatial_imagination_oc",
    "spatial_imagination_oo",
    "spatial_imagination_oc_mv",
    "spatial_imagination_oo_mv",
    "distance_infer_center_oo",
    "distance_infer_center_oo_mv"
]

# hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
# base_cache_dir = os.path.expanduser(hf_home)
# base_cache_dir = "/cache/data/sparbench_tiny_image"
# base_cache_dir = "/cache/data/sparbench_image"


def sparbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
        
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") # or "These are frames of a video."
    
    if doc['task'] in NA_QUESTION_TYPES:
        post_prompt = lmms_eval_specific_kwargs.get("na_post_prompt", "") or "Please answer the question using a single word or phrase."
        return pre_prompt + "\n" + question + "\n" + post_prompt
    elif doc['task'] in MCA_QUESTION_TYPES:
        post_prompt = ""
        if doc['task'] in ['position_matching', "camera_motion_infer"]:
            post_prompt = "The values represent the bounding box coordinates normalized to a 0-1000 scale, with the top-left corner as the origin of the image."
        post_prompt2 = "Answer with the option's letter from the given choices directly."
        return pre_prompt + "\n" + question + "\n" + post_prompt + "\n" + post_prompt2
    elif doc['task'] in SPECIAL_QUESTION_TYPES:
        post_prompt1 = ""
        post_prompt2 = ""
        return pre_prompt + "\n" + question + "\n" + post_prompt1 + "\n" + post_prompt2
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")



def fuzzy_matching(pred):
    return pred.split(' ')[0].rstrip('.').strip()

def process_na(pred, task):
    numbers = re.findall(r'(?<!\^)\d+\.\d+|(?<!\^)\d+', pred)

    # Convert the matched numbers to float or int
    extracted_numbers = [float(num) if '.' in num else int(num) for num in numbers]
    if task in ["depth_prediction_oc_mv", 
                "depth_prediction_oo_mv",
                "distance_prediction_oc_mv",
                "distance_prediction_oo_mv",
                ]:
        if len(extracted_numbers) == 0:
            extracted_numbers = [-1]
        extracted_numbers = [extracted_numbers[-1]]
    return extracted_numbers[0]


def calculate_distance(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

def parse_instruction(instruction):
    return {k: float(v) for k, v in [item.split(":") for item in instruction.split(",")]}

def compute_vci_metric(pred, answer):

    acion_list = ["move_right", "move_left", 
                  "move_forward", "move_backward", 
                  "move_up", "move_down", 
                  "rotate_right", "rotate_left",
                  "rotate_up", "rotate_down"]
    action_order = ["move_right_left",
                    "move_up_down",
                    "move_forward_backward",
                    "rotate_right_left",
                    "rotate_up_down"]

    answer_dict = parse_instruction(pred)
    gt_dict = parse_instruction(answer)

    answer_list = []
    gt_list = []

    for action_pair in action_order:
        if action_pair == "move_right_left":
            answer_list.append(answer_dict.get("move_right", 0) - answer_dict.get("move_left", 0))
            gt_list.append(gt_dict.get("move_right", 0) - gt_dict.get("move_left", 0))
        elif action_pair == "move_up_down":
            answer_list.append(answer_dict.get("move_up", 0) - answer_dict.get("move_down", 0))
            gt_list.append(gt_dict.get("move_up", 0) - gt_dict.get("move_down", 0))
        elif action_pair == "move_forward_backward":
            answer_list.append(answer_dict.get("move_forward", 0) - answer_dict.get("move_backward", 0))
            gt_list.append(gt_dict.get("move_forward", 0) - gt_dict.get("move_backward", 0))
        elif action_pair == "rotate_right_left":
            answer_list.append(answer_dict.get("rotate_right", 0) - answer_dict.get("rotate_left", 0))
            gt_list.append(gt_dict.get("rotate_right", 0) - gt_dict.get("rotate_left", 0))
        elif action_pair == "rotate_up_down":
            answer_list.append(answer_dict.get("rotate_up", 0) - answer_dict.get("rotate_down", 0))
            gt_list.append(gt_dict.get("rotate_up", 0) - gt_dict.get("rotate_down", 0))
    
    mra_list = []
    for gt, answer in zip(gt_list, answer_list):
        mra = mean_relative_accuracy(gt, answer, start=.5, end=.95, interval=.05)
        mra_list.append(mra)

    return np.mean(mra_list)

def compute_dic_metric(pred, answer):
    answer = pred
    answer_gt = answer
    if answer == answer_gt:
        return 1
    elif answer_gt in answer:  # TODO: This is a hacky way to handle the case where the answer is a subset of the predicted answer
        return 1
    
    return 0

def parse_cmi(text):
    pattern = r"\([0-9\.]+,[0-9\.]+\)|[0-9\.]+"

    matches = re.findall(pattern, text)

    if len(matches) < 2:
        if len(matches) == 1 and "(" in matches[0]:
            matches.append("0.0")
        elif len(matches) == 1 and "." in matches[0]:
            matches.insert(0, "(0.0,0.0)")

    result = []
    for match in matches:
        if "(" in match and ")" in match:
            num1, num2 = match.strip("()").split(",")
            result.extend([float(num1), float(num2)])
        else:
            result.append(float(match))

    return result

def compute_cmi_metric(pred, answer):

    pred_process = parse_cmi(pred)
    ans_process = parse_cmi(answer)
    dist = math.sqrt(
        (pred_process[0]/1000 - ans_process[0]/1000) ** 2 + 
        (pred_process[1]/1000 - ans_process[1]/1000) ** 2 + 
        (pred_process[2] - ans_process[2]) ** 2
    )
    return dist


def exact_match(pred, target):
    # return 1. if pred.lower() == target.lower() else 0.
    pred = pred.lower()
    target = target.lower()
    if pred.lower() == target.lower():
        return 1.
    elif pred in target:
        return 1.
    elif pred[0] == target:
        return 1.
    else:
        return 0

def abs_dist_norm(pred, target):
    if target == 0.0:
        return abs(pred - target)
    else:
        return abs((pred - target) / target)

def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()

WORST_CASE_FOR_METRICS = {
    "accuracy": 0.,
    "MRA:.5:.95:.05": 0.,
}

def to_float(pred):
    try:
        pred = float(pred)
    except BaseException as e:
        pred = None
    return pred

def caluculate_json(json_file):
    doc=[]
    with open(json_file, 'r', encoding='utf-8') as f:
        for line in f:
            # doc.append(vsibench_process_results(json.loads(line)))
            doc.append(sparbench_process_results(json.loads(line)))
        # print(doc)
        doc = sparbench_aggregate_results(doc)
    return doc

def extract_answer(text):
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def sparbench_process_results(doc):
    
    if doc['task'] in MCA_QUESTION_TYPES:
        for key, value in METRICS_FOR_MCA.items():
            doc['predicted_answer']=extract_answer(doc['predicted_answer'])
            doc[key] = eval(value)(fuzzy_matching(doc['predicted_answer']), doc['ground_truth'])
        pass
    elif doc['task'] in NA_QUESTION_TYPES:
        for key, value in METRICS_FOR_NA.items():
            try:
                doc['predicted_answer']=extract_answer(doc['predicted_answer'])
                doc[key] = eval(value)(to_float(process_na(doc['predicted_answer'], doc['task'])), to_float(doc['ground_truth']))
            except:
                doc[key] = WORST_CASE_FOR_METRICS[key]
    elif doc['task'] in SPECIAL_QUESTION_TYPES:
        if doc['task'] == "view_change_infer":
            try:
                doc['vci_metric'] = compute_vci_metric(doc['predicted_answer'], doc['ground_truth'])
            except:
                doc['vci_metric'] = 0

    else:
        raise ValueError(f"Unknown question type: {doc['task']}")

    return doc



import pandas as pd
import numpy as np

def sparbench_aggregate_results(results):
    print(results)
    results = pd.DataFrame(results)
    output = {}
    
    # 不按 img_type 分类的统计
    overall_output = {}
    for question_type, question_type_indexes in results.groupby('task').groups.items():
        per_question_type = results.iloc[question_type_indexes]
        if question_type in MCA_QUESTION_TYPES:
            for metric in METRICS_FOR_MCA.keys():
                overall_output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
        elif question_type in NA_QUESTION_TYPES:
            for metric in METRICS_FOR_NA.keys():
                if metric == 'success_rate':
                    overall_output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
                else:
                    overall_output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
        elif question_type in SPECIAL_QUESTION_TYPES:
            if question_type == "view_change_infer":
                overall_output[f"{question_type}_vci_metric"] = per_question_type["vci_metric"].mean()

    overall_output['overall_accuracy'] = sum([_ for _ in overall_output.values()]) / len(overall_output)
    
    # 按 img_type 分类的统计
    img_type_output = {}
    for img_type, img_type_group in results.groupby('image_type'):
        img_type_output[img_type] = {}
        
        # 重置索引，使其从 0 开始
        img_type_group = img_type_group.reset_index(drop=True)
        
        for question_type, question_type_indexes in img_type_group.groupby('task').groups.items():
            per_question_type = img_type_group.iloc[question_type_indexes]

            if question_type in MCA_QUESTION_TYPES:
                for metric in METRICS_FOR_MCA.keys():
                    img_type_output[img_type][f"{question_type}_{metric}"] = per_question_type[metric].mean()
            elif question_type in NA_QUESTION_TYPES:
                for metric in METRICS_FOR_NA.keys():
                    if metric == 'success_rate':
                        img_type_output[img_type][f"{question_type}_{metric}"] = per_question_type[metric].mean()
                    else:
                        img_type_output[img_type][f"{question_type}_{metric}"] = per_question_type[metric].mean()
            elif question_type in SPECIAL_QUESTION_TYPES:
                if question_type == "view_change_infer":
                    img_type_output[img_type]["{question_type}_vci_metric"] = per_question_type["vci_metric"].mean()

        # 计算该 img_type 的总体准确率
        img_type_output[img_type]['overall_accuracy'] = sum([_ for _ in img_type_output[img_type].values()]) / len(img_type_output[img_type])

    # 合并输出结果
    output['overall'] = overall_output
    output['by_img_type'] = img_type_output

    # 计算 Low, Middle, High 的平均值
    low_list = []
    middle_list = []
    high_list = []
    for task in overall_output:
        if task == 'overall_accuracy':
            continue
        task_name = "_".join(task.split("_")[:-1])
        if task_name in Low:
            low_list.append(overall_output[task])
        elif task_name in Middle:
            middle_list.append(overall_output[task])
        elif task_name in High:
            high_list.append(overall_output[task])

    output['overall']['Low'] = np.mean(low_list)
    output['overall']['Middle'] = np.mean(middle_list)
    output['overall']['High'] = np.mean(high_list)

    # 对每个 img_type 计算 Low, Middle, High 的平均值
    for img_type in img_type_output:
        low_list = []
        middle_list = []
        high_list = []
        for task in img_type_output[img_type]:
            if task == 'overall_accuracy':
                continue
            task_name = "_".join(task.split("_")[:-1])
            if task_name in Low:
                low_list.append(img_type_output[img_type][task])
            elif task_name in Middle:
                middle_list.append(img_type_output[img_type][task])
            elif task_name in High:
                high_list.append(img_type_output[img_type][task])

        img_type_output[img_type]['Low'] = np.mean(low_list)
        img_type_output[img_type]['Middle'] = np.mean(middle_list)
        img_type_output[img_type]['High'] = np.mean(high_list)

    eval_logger.info(f"Evaluation results: {output}")
    return output