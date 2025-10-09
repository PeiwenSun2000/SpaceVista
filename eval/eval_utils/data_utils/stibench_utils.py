import re
import os
import json
import logging
import pandas as pd
import numpy as np

# -------------------------------
# STI-Bench constants and prompts
# -------------------------------

# 子任务名称需与数据文件中的 'Task' 字段一致
SR_SUB_TASKS = [
    'Dimensional Measurement',
    'Displacement & Path Length',
    'Ego-Centric Orientation',
    'Spatial Relation',
    'Speed & Acceleration',
    'Trajectory Description',
]

# 思维链模板（与主代码中的 QUESTION_TEMPLATE 和 PROMPT_TEMPLATES 保持一致）
QUESTION_TEMPLATE = (
    "Question: {Question}\n"
    "Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
    "It's encouraged to include self-reflection or verification in the reasoning process. "
    "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."
)

PROMPT_TEMPLATES = {
    "default": {
        "pre_prompt": "Question: {Question}\n",
        "mca_post_prompt": "Answer with the option's letter from the given choices directly.",
        "na_post_prompt": "Please answer the question using a numerical value (e.g., 42 or 3.1)."
    },
    "thinking": {
        "pre_prompt": QUESTION_TEMPLATE,
        "mca_post_prompt": "Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "na_post_prompt": "Please provide the numerical value (e.g., 42 or 3.1) within the <answer> </answer> tags.",
        "special_post_prompt": "First output the thinking process in <think> </think> tags and then output the answer in <answer> </answer> tags.",
    },
}

# -------------------------------
# Utility functions
# -------------------------------

def exact_match(pred, target):
    """
    Performs a case-insensitive comparison of prediction and target strings.
    """
    if pred is None or target is None:
        return 0.0
    return 1.0 if str(pred).strip().lower() == str(target).strip().lower() else 0.0

def extract_answer(text):
    """
    Extracts content from within <answer>...</answer> tags.
    若不存在，则返回原文。
    """
    if text is None:
        return text
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, str(text), re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text

def extract_answer_text(text):
    """
    与主评估代码 stibench_eval 中一致的 <answer> 抽取函数占位。
    如果上游已导入同名函数，可不使用此函数。
    """
    return extract_answer(text)

def normalize_choice_answer(text):
    """
    Attempts to extract a capital letter answer (A-E) from the text.
    Unifies the format to single capital letter 'A'...'E'.
    若未匹配返回 None。
    """
    if text is None:
        return None
    patterns = [
        r"\(([A-E])\)",                          # (A)
        r"Ans\s*=\s*['\"]?([A-E])['\"]?",        # Ans='C'
        r"Answer\s*[:=]\s*([A-E])",              # Answer: B
        r"Option\s+([A-E])",                     # Option D
        r"\b([A-E])\s*(?:is|was)\s*correct",     # A is correct
        r"\b([A-E])[\.\)]\s*$",                  # C.  /  D)
        r"^\s*([A-E])\s*$",                      # just a single letter
    ]
    for pattern in patterns:
        match = re.search(pattern, str(text), flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    # 兜底：句首的 A-E
    head = re.match(r'^\s*([A-E])\b', str(text).strip(), re.IGNORECASE)
    if head:
        return head.group(1).upper()
    return None

def fuzzy_matching_num(pred):
    """
    将文字数字或文本中的数字提取为字符串数值。
    若找不到则返回 'None'（字符串）。
    """
    if pred is None:
        return "None"
    s = str(pred).strip().lower()
    number_words = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'a': '1', 'an': '1'
    }
    for word, digit in number_words.items():
        if re.search(r'\b' + re.escape(word) + r'\b', s):
            return digit
    number_match = re.search(r'(-?\d+(\.\d+)?)', s)
    if number_match:
        return number_match.group(1)
    return "None"

def to_float(pred):
    """
    尝试将字符串转为 float；失败返回 None。
    """
    try:
        return float(pred)
    except (ValueError, TypeError):
        return None

def format_time(seconds):
    """
    简单时间格式化，供日志打印使用。
    """
    seconds = float(seconds or 0)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m {s:.1f}s"
    if m > 0:
        return f"{m}m {s:.1f}s"
    return f"{s:.1f}s"

# -------------------------------
# STI-Bench evaluation helpers
# -------------------------------

def stibench_process_result_entry(doc, mode="thinking"):
    """
    处理单条结果：
    - 根据 mode 可选地先从 <answer> 标签中抽取。
    - 归一化多选答案为大写字母 A-E。
    - 计算 accuracy（严格匹配）。
    输入字段要求：
      doc: {
        'predicted_answer': str,
        'ground_truth': str (通常为 A/B/C/D/E),
        'task': str,
        ... 其他任意字段
      }
    输出：添加或更新
      doc['predicted_answer'] 归一化后答案（如 'A'）
      doc['accuracy'] 浮点 0/1
    """
    pred = doc.get("predicted_answer", "")
    gt = doc.get("ground_truth", "")

    # 当使用 thinking 模式时，先抽取 <answer> 内容
    if mode == "thinking" and isinstance(pred, str) and "<answer>" in pred:
        pred = extract_answer(pred)

    # 标准化为单个选项字母
    norm_pred = normalize_choice_answer(pred) or str(pred).strip()
    if norm_pred is None or norm_pred == "":
        norm_pred = "NORMALIZE_ERROR"

    doc["predicted_answer"] = norm_pred
    doc["accuracy"] = exact_match(norm_pred, gt)
    return doc

def stibench_aggregate_results(results):
    """
    与主代码中的 stibench_aggregate_results 对齐，但更健壮：
    - 按 task 聚合 mean(accuracy)
    - 针对 SR 子任务单独计算 sr_sub_accuracy
    - overall_accuracy 为所有键值的平均（含各 task accuracy 与 sr_sub_accuracy）
    """
    results_df = pd.DataFrame(results) if results else pd.DataFrame()
    output = {}

    if results_df.empty:
        output['sr_sub_accuracy'] = 0.0
        output['overall_accuracy'] = 0.0
        return output

    if 'task' not in results_df.columns or 'accuracy' not in results_df.columns:
        logging.getLogger('eval_logger').warning("Results missing 'task' or 'accuracy' columns.")
        output['sr_sub_accuracy'] = 0.0
        output['overall_accuracy'] = 0.0
        return output

    # 每个任务的准确率
    for task, idx in results_df.groupby('task').groups.items():
        per_task = results_df.iloc[idx]
        output[f"{task}"] = float(per_task['accuracy'].mean())

    # SR 子任务平均
    sr_vals = [output[t] for t in SR_SUB_TASKS if t in output]
    output['sr_sub_accuracy'] = float(np.mean(sr_vals)) if sr_vals else 0.0

    # overall：包括所有键（各 task + sr_sub_accuracy）
    all_vals = [v for v in output.values()]
    output['overall_accuracy'] = float(np.mean(all_vals)) if all_vals else 0.0

    eval_logger = logging.getLogger('eval_logger')
    eval_logger.info(f"Evaluation results: {output}")
    return output

def caluculate_json(jsonl_file_path, mode="thinking"):
    """
    顶层评估入口：
    - 逐行读取 JSONL
    - 根据 mode 解析并计算 accuracy
    - 返回分任务聚合结果
    """
    results = []
    if not os.path.exists(jsonl_file_path):
        print(f"Error: File not found: {jsonl_file_path}")
        return {}

    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON: {line[:200]}...")
                continue

            doc = stibench_process_result_entry(doc, mode=mode)
            results.append(doc)

    aggregated = stibench_aggregate_results(results)
    return aggregated

# -------------------------------
# Prompt assembly helpers
# -------------------------------

def build_mca_prompt(question_text, candidates_dict, prompt_type="default", time_span=None):
    """
    组装选择题 Prompt。
    - question_text: 原始问题描述
    - candidates_dict: 形如 {'A': 'xxx', 'B': 'yyy'} 的选项字典
    - time_span: 可选 (time_start, time_end) 秒
    - prompt_type: 'default' or 'thinking'
    """
    tpl = PROMPT_TEMPLATES.get(prompt_type, PROMPT_TEMPLATES["default"])
    cand_lines = []
    if isinstance(candidates_dict, dict):
        for k, v in candidates_dict.items():
            cand_lines.append(f"{k} {v}")
    candidate_block = "\n".join(cand_lines)

    prefix = ""
    if time_span and len(time_span) == 2:
        prefix = f"From {time_span[0]} seconds to {time_span[1]} seconds. "

    q = prefix + question_text + ("\n" + candidate_block if candidate_block else "")
    prompt_text = tpl["pre_prompt"].format(Question=q)
    prompt_text += "\n" + tpl["mca_post_prompt"]
    return prompt_text

def build_na_prompt(question_text, prompt_type="default", time_span=None):
    """
    组装数值题 Prompt。
    """
    tpl = PROMPT_TEMPLATES.get(prompt_type, PROMPT_TEMPLATES["default"])
    prefix = ""
    if time_span and len(time_span) == 2:
        prefix = f"From {time_span[0]} seconds to {time_span[1]} seconds. "
    q = prefix + question_text
    prompt_text = tpl["pre_prompt"].format(Question=q)
    prompt_text += "\n" + tpl["na_post_prompt"]
    return prompt_text

# -------------------------------
# Lightweight logger helper
# -------------------------------

def setup_basic_logger(name="eval_logger", level=logging.INFO):
    """
    如果外部未配置 loguru，这里提供一个基础 logger。
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger