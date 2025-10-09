import os
import json
from pandas.core.frame import DataFrame
import torch
import pandas as pd
import numpy as np
import logging
import multiprocessing as mp
from tqdm import tqdm
from PIL import Image
from decord import VideoReader, cpu
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from accelerate import Accelerator
from qwen_vl_utils import process_vision_info
from multiprocessing import set_start_method
import time
from datetime import datetime, timedelta
import argparse
import glob
import io
import cv2
from natsort import natsorted
from typing import Optional, List, Union, Tuple
from pathlib import Path

# Import evaluation metric functions
from eval_utils.data_utils.vsi_util import caluculate_json as calculate_vsi_metrics
from eval_utils.data_utils.vsi_util_3 import caluculate_json as calculate_mmsi_metrics
from eval_utils.data_utils.vsi_util_2 import caluculate_json_spacevista as calculate_spacevista_metrics
from eval_utils.data_utils.spar_util import caluculate_json as calculate_spar_metrics
from eval_utils.data_utils.stibench_utils import caluculate_json as calculate_sti_metrics

# --- Constants and Configuration ---

MCA_QUESTION_TYPES = [
    "object_rel_direction_easy", "object_rel_direction_medium",
    "object_rel_direction_hard", "object_rel_distance", "route_planning",
    "obj_appearance_order"
]
NA_QUESTION_TYPES = [
    "object_abs_distance", "object_counting", "object_size_estimation",
    "room_size_estimation"
]

PROMPT_TEMPLATES = {
    "default": {
        "pre_prompt": "",
        "mca_post_prompt": "Answer with the option's letter from the given choices directly.",
        "na_post_prompt": "Please answer the question using a single word or phrase."
    },
    "thinking": {
        "pre_prompt": "",
        "mca_post_prompt": "First output the thinking process in <think> </think> tags and then output an option letter in <answer> </answer> tags.",
        "na_post_prompt": "First output the thinking process in <think> </think> tags and then output a number in <answer> </answer> tags."
    },
    "gemini_api": {
        "pre_prompt": "",
        "mca_post_prompt": "Answer with the option's letter from the given choices directly.",
        "na_post_prompt": "Do not response anything other than a single number!"
    },
    "gpt4v": {
        "pre_prompt": "",
        "mca_post_prompt": "Answer with the option's letter from the given choices directly.",
        "na_post_prompt": "Do not response anything other than a single number!"
    }
}

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

def process_and_pad_images(geometry_encoder_inputs: List[List[Image.Image]], extra_type: str) -> Tuple[List[List[Image.Image]], List[torch.Tensor]]:
    """
    Returns:
      - processed_batches_364: List[List[PIL.Image]]; each image is square 364x364, used as video frames for processor.video
      - processed_batches_420: List[torch.Tensor], shape (N, 3, 420, 420), used as geometry_encoder_inputs
    extra_type: 'vggt' or 'dinov3'
    """
    if not geometry_encoder_inputs or not geometry_encoder_inputs[0]:
        raise ValueError("Input list cannot be empty")

    processed_batches_364: List[List[Image.Image]] = []
    processed_batches_420: List[torch.Tensor] = []

    to_tensor = T.ToTensor()
    if extra_type == "vggt":
        target_sizes = [364, 364]
    else:
        target_sizes = [364, 420]

    def ensure_rgb(img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            raise TypeError(f"Expected a PIL Image, but got {type(img)}")
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        return img.convert("RGB")

    def compute_resized_dims(width: int, height: int, target_size: int):
        if width >= height:
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14
        else:
            new_height = target_size
            new_width = round(width * (new_height / height) / 14) * 14
        new_width = max(14, new_width)
        new_height = max(14, new_height)
        return new_width, new_height

    def pad_pil_to_square(img_pil: Image.Image, target_size: int) -> Image.Image:
        bg = Image.new("RGB", (target_size, target_size), (255, 255, 255))
        w, h = img_pil.size
        if w > target_size or h > target_size:
            left = max(0, (w - target_size) // 2)
            top = max(0, (h - target_size) // 2)
            img_pil = img_pil.crop((left, top, left + min(target_size, w), top + min(target_size, h)))
            w, h = img_pil.size
        left = (target_size - w) // 2
        top = (target_size - h) // 2
        bg.paste(img_pil, (left, top))
        return bg

    def preprocess_364_pil(img: Image.Image, target_size: int) -> Image.Image:
        img = ensure_rgb(img)
        w, h = img.size
        new_w, new_h = compute_resized_dims(w, h, target_size)
        img_resized = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        img_padded = pad_pil_to_square(img_resized, target_size)
        return img_padded

    def preprocess_420_tensor(img: Image.Image, target_size: int) -> torch.Tensor:
        img = ensure_rgb(img)
        w, h = img.size
        new_w, new_h = compute_resized_dims(w, h, target_size)
        img_resized = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        img_tensor = to_tensor(img_resized)

        h_padding = target_size - img_tensor.shape[1]
        w_padding = target_size - img_tensor.shape[2]

        if h_padding < 0 or w_padding < 0:
            top = max(0, (img_tensor.shape[1] - target_size) // 2)
            left = max(0, (img_tensor.shape[2] - target_size) // 2)
            img_tensor = img_tensor[:, top:top + target_size, left:left + target_size]
            h_padding = 0
            w_padding = 0

        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            img_tensor = torch.nn.functional.pad(
                img_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
            )

        return img_tensor

    for image_list in geometry_encoder_inputs:
        processed_images_364_pil: List[Image.Image] = []
        processed_images_420_tensor: List[torch.Tensor] = []

        for img in image_list:
            img_364_pil = preprocess_364_pil(img, target_sizes[0])
            img_420_tensor = preprocess_420_tensor(img, target_sizes[1])

            processed_images_364_pil.append(img_364_pil)
            processed_images_420_tensor.append(img_420_tensor)

        processed_batches_364.append(processed_images_364_pil)
        images_tensor_420 = torch.stack(processed_images_420_tensor)
        processed_batches_420.append(images_tensor_420)

    return processed_batches_364, processed_batches_420

def format_time(elapsed_seconds: float) -> str:
    time_delta = timedelta(seconds=int(elapsed_seconds))
    hours = time_delta.seconds // 3600
    minutes = (time_delta.seconds % 3600) // 60
    seconds = time_delta.seconds % 60
    return f"{hours:02}h{minutes:02}m{seconds:02}s"

def setup_logger(rank: int, log_file: str, params_dict: dict) -> logging.Logger:
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_with_timestamp = log_file.replace(
        ".log", f"_{timestamp_str}_rank_{rank}.log")
    logging.basicConfig(
        filename=log_file_with_timestamp,
        level=logging.INFO,
        format=f'%(asctime)s - [Rank {rank}] - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(f"Starting process with rank {rank}")
    logger.info("Running with parameters:")
    for key, value in params_dict.items():
        logger.info(f"  {key}: {value}")
    return logger

def allocate_gpu(rank: int, gpu_ids: str, world_size: int) -> str:
    gpu_ids_list = [gpu_id.strip() for gpu_id in gpu_ids.split(',')]
    num_gpus_available = len(gpu_ids_list)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids_list)

    if world_size > num_gpus_available:
        logging.warning(
            f"Rank {rank}: Not enough GPUs for all processes. GPUs will be reused."
        )

    selected_gpu_index = rank % num_gpus_available
    selected_gpu = gpu_ids_list[selected_gpu_index]
    
    if world_size > 1:
        torch.cuda.set_device(int(selected_gpu))

    logger = logging.getLogger(__name__)
    logger.info(
        f"Rank {rank}: Allocated GPU: {selected_gpu}. CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}"
    )
    return selected_gpu

def resize_image(image: Image.Image, max_size: int = 448) -> Image.Image:
    w, h = image.size
    if max(h, w) <= max_size:
        return image
    if h > w:
        new_h = max_size
        new_w = int(w * (max_size / h))
    else:
        new_w = max_size
        new_h = int(h * (max_size / w))
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

def load_video_frames_from_file(video_path: str,
                             num_frames: int = 16,
                             target_resolution: tuple = (448, 448)):
    try:
        vr = VideoReader(video_path, ctx=cpu())
        total_frames = len(vr)
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames_np = vr.get_batch(frame_indices).asnumpy()
        frames_pil = [
            resize_image(Image.fromarray(f), max(target_resolution))
            for f in frames_np
        ]
        return frames_pil
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to load video {video_path}: {e}")
        return None

def evaluate_vsibench(rank, world_size, dataset_path, video_dir, model_name,
                      output_dir, log_file, gpu_ids, num_frames,
                      target_resolution, debug, batch_size,
                      debug_size, params_dict):
    logger = setup_logger(rank, log_file, params_dict)
    start_time_process = time.time()

    allocate_gpu(rank, gpu_ids, world_size)
    accelerator = Accelerator()
    device = accelerator.device
    logger.info(f"Rank {rank} using device: {device}")

    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    processor.tokenizer.padding_side = 'left'
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto" if world_size == 1 and len(gpu_ids.split(',')) > 1 else None
    )
    if not (world_size == 1 and len(gpu_ids.split(',')) > 1):
        model.to(device)
    model = accelerator.prepare(model)
    model.eval()

    if not isinstance(dataset_path, list):
        df = pd.read_parquet(dataset_path)
    else:
        dfs = [pd.read_parquet(file) for file in dataset_path]
        df = pd.concat(dfs, ignore_index=True)
    if debug:
        df = df.head(debug_size)
        logger.info(f"Rank {rank} [Debug Mode]: Processing the first {debug_size} samples.")

    df_shard = np.array_split(df, world_size)[rank] if world_size > 1 else df
    logger.info(f"Rank {rank} shard size: {len(df_shard)}")

    results = []
    total_samples = len(df_shard)
    if total_samples == 0:
        logger.info(f"Rank {rank} has an empty shard. Skipping.")
        return os.path.join(output_dir, f"results_rank_{rank}.jsonl"), 0

    QUESTION_TEMPLATE = "Please think about this question as if you were a human pondering deeply. \nPlease give your final answer between the <answer> </answer> tags."
    TYPE_TEMPLATE = {
        "multiple choice": "Question: \n{Question}\n Options: \n{Option} Please provide the thinking process within the <think> </think> tags. \nPlease provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "regression": "Question:  \n{Question}\n Please provide the thinking process within the <think> </think> tags. Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
    }
    
    use_geometry_encoder = 'dinov3'

    for start_index in tqdm(range(0, total_samples, batch_size), desc=f"Rank {rank} Processing"):
        batch_df = df_shard.iloc[start_index:min(start_index + batch_size, total_samples)]
        
        batch_messages = []
        batch_metadata = []
        batch_frames_for_geom = []

        for _, row in batch_df.iterrows():
            video_path = os.path.join(video_dir, row['dataset'], f"{row['scene_name']}.mp4")
            if not os.path.exists(video_path):
                logger.warning(f"Video not found: {video_path}, skipping.")
                continue

            frames = load_video_frames_from_file(video_path, num_frames, target_resolution)
            if frames is None:
                continue

            question = row['question']
            options = row.get('options')

            if row['question_type'] in NA_QUESTION_TYPES:
                prompt_text = "\n" + QUESTION_TEMPLATE + TYPE_TEMPLATE["regression"].format(Question=question)
            elif row['question_type'] in MCA_QUESTION_TYPES:
                options_text = " ".join(options.tolist())
                prompt_text = "\n" + QUESTION_TEMPLATE + TYPE_TEMPLATE["multiple choice"].format(Question=question, Option=options_text)
            else:
                prompt_text = "\n" + QUESTION_TEMPLATE + "Question: \n" + question

            frames_for_geom = frames
            processed_364_list, processed_420_tensor_list = process_and_pad_images([frames_for_geom], use_geometry_encoder)
            frames_364 = processed_364_list[0]
            geom_tensor = processed_420_tensor_list[0]
            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": frames_364},
                    {"type": "text", "text": prompt_text},
                ],
            }]

            batch_messages.append(messages)
            batch_metadata.append(row)
            batch_frames_for_geom.append(geom_tensor)

        if not batch_messages:
            continue

        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        _, video_inputs = process_vision_info(batch_messages)

        geometry_encoder_inputs = batch_frames_for_geom
        
        inputs = processor(
            text=texts,
            videos=video_inputs,
            geometry_encoder_inputs=geometry_encoder_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=512, do_sample=False)
        trimmed_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        predictions = processor.batch_decode(trimmed_ids, skip_special_tokens=True)

        for i, (row, pred) in enumerate(zip(batch_metadata, predictions)):
            results.append({
                'id': row['id'],
                'dataset': row['dataset'],
                'scene_name': row['scene_name'],
                'question': row['question'],
                'ground_truth': row['ground_truth'],
                'predicted_answer': pred.strip(),
                'question_type': row['question_type'],
                'prompt': texts[i]
            })

    process_output_file = os.path.join(output_dir, f"results_rank_{rank}.jsonl")
    with open(process_output_file, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
            
    elapsed_time = time.time() - start_time_process
    logger.info(f"Rank {rank} finished in {format_time(elapsed_time)}. Results saved to {process_output_file}")
    
    return process_output_file

def evaluate_mmsibench(rank, world_size, dataset_path, video_dir, model_name,
                       output_dir, log_file, gpu_ids, num_frames,
                       target_resolution, debug, batch_size,
                       debug_size, params_dict):
    logger = setup_logger(rank, log_file, params_dict)
    start_time_process = time.time()

    allocate_gpu(rank, gpu_ids, world_size)
    accelerator = Accelerator()
    device = accelerator.device
    logger.info(f"Rank {rank} using device: {device}")

    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    processor.tokenizer.padding_side = 'left'
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto" if world_size == 1 and len(gpu_ids.split(',')) > 1 else None
    )
    if not (world_size == 1 and len(gpu_ids.split(',')) > 1):
        model.to(device)
    model = accelerator.prepare(model)
    model.eval()

    df = pd.read_parquet(dataset_path)
    if debug:
        df = df.head(debug_size)
        logger.info(f"Rank {rank} [Debug Mode]: Processing the first {debug_size} samples.")

    df_shard = np.array_split(df, world_size)[rank] if world_size > 1 else df
    logger.info(f"Rank {rank} shard size: {len(df_shard)}")

    results = []
    total_samples = len(df_shard)
    if total_samples == 0:
        logger.info(f"Rank {rank} has an empty shard. Skipping.")
        return os.path.join(output_dir, f"results_rank_{rank}.jsonl"), 0

    QUESTION_TEMPLATE = "Please think about this question as if you were a human pondering deeply. \nPlease give your final answer between the <answer> </answer> tags."
    TYPE_TEMPLATE = {
        "multiple choice": "Question: \n{Question}\n Options: \n{Option} Please provide the thinking process within the <think> </think> tags. \nPlease provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "regression": "Question:  \n{Question}\n Please provide the thinking process within the <think> </think> tags. Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
    }

    use_geometry_encoder = 'dinov3'

    for start_index in tqdm(range(0, total_samples, batch_size), desc=f"Rank {rank} Processing"):
        batch_df = df_shard.iloc[start_index:min(start_index + batch_size, total_samples)]
        
        batch_messages = []
        batch_metadata = []
        batch_frames_for_geom = []

        for _, row in batch_df.iterrows():
            frames = []
            for img_bytes in row['images']:
                img = Image.open(io.BytesIO(img_bytes))
                img = resize_image(img, max(target_resolution))
                frames.append(img)
            
            if not frames:
                logger.warning(f"No frames found for id {row['id']}, skipping.")
                continue

            processed_364_list, processed_420_tensor_list = process_and_pad_images([frames], use_geometry_encoder)
            frames_364 = processed_364_list[0]
            geom_tensor = processed_420_tensor_list[0]
            question_full = row['question']
            try:
                question, options_str = question_full.split("\nOptions: ")
                options_text = options_str.strip()
            except Exception:
                question = question_full
                options_text = row.get('options', "")

            prompt_text = "\n" + QUESTION_TEMPLATE + TYPE_TEMPLATE["multiple choice"].format(Question=question, Option=options_text)
            frames_364 = [x for x in frames_364 for _ in range(2)]
            geom_tensor = geom_tensor.repeat_interleave(2, dim=0) 
            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": frames_364},
                    {"type": "text", "text": prompt_text},
                ],
            }]
            batch_messages.append(messages)
            batch_metadata.append(row)
            batch_frames_for_geom.append(geom_tensor)

        if not batch_messages:
            continue

        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        _, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=texts,
            videos=video_inputs,
            geometry_encoder_inputs=batch_frames_for_geom,
            padding=True,
            return_tensors="pt"
        ).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=512, do_sample=False)
        trimmed_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        predictions = processor.batch_decode(trimmed_ids, skip_special_tokens=True)
        
        for i, (row, pred) in enumerate(zip(batch_metadata, predictions)):
            results.append({
                'id': row['id'],
                'dataset': 'mmsi-bench',
                'question': row['question'],
                'ground_truth': row['answer'],
                'predicted_answer': pred.strip(),
                'question_type': row['question_type'],
                'prompt': texts[i],
            })
            
    process_output_file = os.path.join(output_dir, f"results_rank_{rank}.jsonl")
    with open(process_output_file, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
            
    elapsed_time = time.time() - start_time_process
    logger.info(f"Rank {rank} finished in {format_time(elapsed_time)}. Results saved to {process_output_file}")

    return process_output_file

def evaluate_spacevista(rank, world_size, dataset_path, frames_root_dir, model_name,
                       output_dir, log_file, gpu_ids, num_frames,
                       target_resolution, debug, batch_size,
                       debug_size, params_dict, model_path=None, local_infer_callable=None):
    logger = setup_logger(rank, log_file, params_dict)
    start_time_process = time.time()

    allocate_gpu(rank, gpu_ids, world_size)
    accelerator = Accelerator()
    device = accelerator.device
    logger.info(f"Rank {rank} using device: {device}")

    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    processor.tokenizer.padding_side = 'left'
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto" if world_size == 1 and len(gpu_ids.split(',')) > 1 else None
    )
    if not (world_size == 1 and len(gpu_ids.split(',')) > 1):
        model.to(device)
    model = accelerator.prepare(model)
    model.eval()

    with open(dataset_path, 'r') as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame(data)

    if debug:
        df = df.head(debug_size)
        logger.info(f"Rank {rank} [Debug Mode]: Processing the first {debug_size} samples.")

    df_shard = np.array_split(df, world_size)[rank] if world_size > 1 else df
    logger.info(f"Rank {rank} shard size: {len(df_shard)}")

    results = []
    total_samples = len(df_shard)
    if total_samples == 0:
        logger.info(f"Rank {rank} has an empty shard. Skipping.")
        return os.path.join(output_dir, f"results_rank_{rank}.jsonl"), 0

    QUESTION_TEMPLATE = "Please think about this question as if you were a human pondering deeply. \nPlease give your final answer between the <answer> </answer> tags."
    TYPE_TEMPLATE = {
        "multiple choice": "Question: \n{Question}\n Options: \n{Option} Please provide the thinking process within the <think> </think> tags. \nPlease provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "regression": "Question:  \n{Question}\n Please provide the thinking process within the <think> </think> tags. Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
    }

    use_geometry_encoder = 'dinov3'

    for index, row in tqdm(df_shard.iterrows(), total=total_samples, desc=f"Rank {rank} Processing"):
        frames_dir = os.path.join(frames_root_dir, row["Frames Path"])
        if not os.path.isdir(frames_dir):
            logger.warning(f"Frames directory not found: {frames_dir}, skipping.")
            continue

        image_files = natsorted(os.listdir(frames_dir))
        if not image_files:
            logger.warning(f"No frames found in: {frames_dir}, skipping.")
            continue
        
        indices = np.linspace(0, len(image_files) - 1, num_frames, dtype=int)
        frames = []
        for i in indices:
            try:
                img_path = os.path.join(frames_dir, image_files[i])
                img = Image.open(img_path).convert("RGB")
                frames.append(resize_image(img, max(target_resolution)))
            except Exception as e:
                logger.error(f"Error loading image {img_path}: {e}")
        
        if not frames:
            logger.warning(f"Could not load valid frames from {frames_dir}, skipping.")
            continue

        processed_364_list, processed_420_tensor_list = process_and_pad_images([frames], use_geometry_encoder)
        frames_364 = processed_364_list[0]
        geom_tensor = processed_420_tensor_list[0]

        question = row['Question']
        if row['Question Type'] == "regression":
            prompt_text = "\n" + QUESTION_TEMPLATE + TYPE_TEMPLATE["regression"].format(Question=question)
        elif row['Question Type'] == "multiple choice":
            options = row['options']
            options_text = " ".join(options)
            prompt_text = "\n" + QUESTION_TEMPLATE + TYPE_TEMPLATE["multiple choice"].format(Question=question, Option=options_text)
        else:
            prompt_text = "\n" + QUESTION_TEMPLATE + "Question: \n" + question
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": frames_364},
                {"type": "text", "text": prompt_text},
            ],
        }]

        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        _, video_inputs = process_vision_info([messages])
        print(geometry_encoder_inputs.shape)
        inputs = processor(
            text=[text_prompt],
            videos=video_inputs,
            geometry_encoder_inputs=[geom_tensor],
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                use_cache=True,
                max_new_tokens=512,
                do_sample=False
            )
        gen_ids = output_ids[0][inputs.input_ids.shape[1]:]
        prediction = processor.decode(gen_ids, skip_special_tokens=True).strip()

        results.append({
            'Question Number': row['Question Number'],
            'Scene Source': row['Scene Source'],
            'Frames Path': row['Frames Path'],
            'question': row['Question'],
            'ground_truth': row['Answer'],
            'predicted_answer': prediction,
            'question_type': row['TaskType'],
            'task_type': row['Question Type'],
            'prompt': text_prompt
        })

    process_output_file = os.path.join(output_dir, f"results_rank_{rank}.jsonl")
    with open(process_output_file, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
            
    elapsed_time = time.time() - start_time_process
    logger.info(f"Rank {rank} finished in {format_time(elapsed_time)}. Results saved to {process_output_file}")

    return process_output_file


def evaluate_sparbench(rank, world_size, dataset_path, frames_root_dir, model_name,
                       output_dir, log_file, gpu_ids, num_frames,
                       target_resolution, debug, batch_size,
                       debug_size, params_dict, model_path=None, local_infer_callable=None):
    """
    Main evaluation function for SPARBench.
    """
    logger = setup_logger(rank, log_file, params_dict)
    start_time_process = time.time()

    allocate_gpu(rank, gpu_ids, world_size)
    accelerator = Accelerator()
    device = accelerator.device
    logger.info(f"Rank {rank} using device: {device}")

    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    processor.tokenizer.padding_side = 'left'
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto" if world_size == 1 and len(gpu_ids.split(',')) > 1 else None
    )
    if not (world_size == 1 and len(gpu_ids.split(',')) > 1):
        model.to(device)
    model = accelerator.prepare(model)
    model.eval()

    df = pd.read_parquet(dataset_path)
    print(len(df))
    if debug:
        df = df.head(debug_size)
        logger.info(f"Rank {rank} [Debug Mode]: Processing the first {debug_size} samples.")

    df_shard = np.array_split(df, world_size)[rank] if world_size > 1 else df
    logger.info(f"Rank {rank} shard size: {len(df_shard)}")

    results = []
    total_samples = len(df_shard)
    if total_samples == 0:
        logger.info(f"Rank {rank} has an empty shard. Skipping.")
        return os.path.join(output_dir, f"results_rank_{rank}.jsonl"), 0

    QUESTION_TEMPLATE = "Please think about this question as if you were a human pondering deeply.\nPlease give your final answer between the <answer> </answer> tags."
    TYPE_TEMPLATE = {
        "multiple choice": "\nQuestion:\n{Question}\n Please provide the thinking process within the <think> </think> tags. \nPlease provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "regression": "\nQuestion:\n{Question}\n Please provide the thinking process within the <think> </think> tags. Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
    }

    use_geometry_encoder = 'dinov3'

    for start_index in tqdm(range(0, total_samples, batch_size), desc=f"Rank {rank} Processing"):
        batch_df = df_shard.iloc[start_index:min(start_index + batch_size, total_samples)]
        
        batch_messages = []
        batch_metadata = []
        batch_frames_for_geom = []

        for _, row in batch_df.iterrows():
            frames = []
            for img_bytes in row['image']:
                img = Image.open(io.BytesIO(img_bytes["bytes"]))
                img = resize_image(img, max(target_resolution))
                frames.append(img)
            
            if not frames:
                logger.warning(f"No frames found for id {row['id']}, skipping.")
                continue

            format_type = row['format_type']
            processed_364_list, processed_420_tensor_list = process_and_pad_images([frames], use_geometry_encoder)
            frames_364 = processed_364_list[0]
            geom_tensor = processed_420_tensor_list[0]
            question_full = row['question']
            if row['task'] in ['position_matching', "camera_motion_infer"]:
                question_full += "\nThe values represent the bounding box coordinates normalized to a 0-1000 scale, with the top-left corner as the origin of the image."
            if format_type == "select":
                format_type = "multiple choice"
                question = question_full
                prompt_text = "\n" + QUESTION_TEMPLATE + TYPE_TEMPLATE["multiple choice"].format(Question=question)
            elif format_type == "fill":
                format_type = "regression"
                question = question_full
                prompt_text = "\n" + QUESTION_TEMPLATE + TYPE_TEMPLATE["regression"].format(Question=question)
            if len(frames_364) <= 3:
                frames_364 = [x for x in frames_364 for _ in range(2)]
                geom_tensor = geom_tensor.repeat_interleave(2, dim=0) 
            elif len(frames_364) % 2:
                frames_364.append(frames_364[-1])
                last = geom_tensor[-1:,...]
                geom_tensor = np.concatenate([geom_tensor, last], axis=0)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": frames_364},
                    {"type": "text", "text": prompt_text},
                ],
            }]
            batch_messages.append(messages)
            batch_metadata.append(row)
            batch_frames_for_geom.append(geom_tensor)

        if not batch_messages:
            continue

        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        _, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=texts,
            videos=video_inputs,
            geometry_encoder_inputs=batch_frames_for_geom,
            padding=True,
            return_tensors="pt"
        ).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=512, do_sample=False)
        trimmed_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        predictions = processor.batch_decode(trimmed_ids, skip_special_tokens=True)
        
        for i, (row, pred) in enumerate(zip(batch_metadata, predictions)):
            results.append({
                'id': row['id'],
                'dataset': 'spar-bench',
                'question': row['question'],
                'ground_truth': row['answer'],
                'predicted_answer': pred.strip(),
                'question_type': format_type,
                'prompt': texts[i],
                'task': row['task'],
                'image_type': row['img_type']
            })
            
    process_output_file = os.path.join(output_dir, f"results_rank_{rank}.jsonl")
    with open(process_output_file, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
            
    elapsed_time = time.time() - start_time_process
    logger.info(f"Rank {rank} finished in {format_time(elapsed_time)}. Results saved to {process_output_file}")

    return process_output_file

def load_frames_by_id(raw_id: str | int,
                      frame_num: int,
                      root: str = "your_path/sti-bench/frames",
                      exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")):
    """
    Load frames by ID:
      1) zero-fill to 6 chars
      2) join to directory under root
      3) list and naturally sort frame files
      4) uniformly sample frame_num
      5) load with PIL and return list
    Returns:
      padded_id: str
      frame_paths_sampled: list[Path]
      images_sampled: list[PIL.Image.Image]
    """
    padded_id = str(raw_id).zfill(6)
    dir_path = Path(root) / padded_id

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {dir_path}")

    frame_paths = []
    for p in dir_path.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            frame_paths.append(p)

    def natural_key(path_obj: Path):
        import re
        s = path_obj.stem
        parts = re.split(r"(\d+)", s)
        return [int(t) if t.isdigit() else t.lower() for t in parts] + [path_obj.suffix.lower()]

    frame_paths.sort(key=natural_key)

    if not frame_paths:
        raise FileNotFoundError(f"No image files in directory: {dir_path}")

    n = len(frame_paths)
    if frame_num is None or frame_num >= n:
        indices = list(range(n))
    else:
        if frame_num <= 0:
            raise ValueError(f"frame_num must be positive, got {frame_num}")
        if frame_num == 1:
            indices = [n // 2]
        else:
            step = (n - 1) / (frame_num - 1)
            indices = [round(i * step) for i in range(frame_num)]
            dedup = []
            seen = set()
            for idx in indices:
                if idx not in seen:
                    dedup.append(idx)
                    seen.add(idx)
            indices = dedup
            i = 0
            while len(indices) < frame_num:
                cand = min(n - 1, max(0, round(i * step)))
                if cand not in seen:
                    indices.append(cand)
                    seen.add(cand)
                i += 1
            indices.sort()

    frame_paths_sampled = [frame_paths[i] for i in indices]

    images_sampled = []
    for fp in frame_paths_sampled:
        with Image.open(fp) as im:
            images_sampled.append(im.copy())

    return padded_id, frame_paths_sampled, images_sampled

def evaluate_stibench(rank, world_size, dataset_path, frames_root_dir, model_name,
                       output_dir, log_file, gpu_ids, num_frames,
                       target_resolution, debug, batch_size,
                       debug_size, params_dict, model_path=None, local_infer_callable=None):

    logger = setup_logger(rank, log_file, params_dict)
    start_time_process = time.time()

    allocate_gpu(rank, gpu_ids, world_size)
    accelerator = Accelerator()
    device = accelerator.device
    logger.info(f"Rank {rank} using device: {device}")

    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    processor.tokenizer.padding_side = 'left'
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto" if world_size == 1 and len(gpu_ids.split(',')) > 1 else None
    )
    if not (world_size == 1 and len(gpu_ids.split(',')) > 1):
        model.to(device)
    model = accelerator.prepare(model)
    model.eval()

    df = pd.read_parquet(dataset_path)
    if debug:
        df = df.head(debug_size)
        logger.info(f"Rank {rank} [Debug Mode]: Processing the first {debug_size} samples.")

    df_shard = np.array_split(df, world_size)[rank] if world_size > 1 else df
    logger.info(f"Rank {rank} shard size: {len(df_shard)}")

    results = []
    total_samples = len(df_shard)
    if total_samples == 0:
        logger.info(f"Rank {rank} has an empty shard. Skipping.")
        return os.path.join(output_dir, f"results_rank_{rank}.jsonl"), 0

    QUESTION_TEMPLATE = "Please think about this question as if you were a human pondering deeply.\nEngage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions. \nIt's encouraged to include self-reflection or verification in the reasoning process. \nPlease give your final answer between the <answer> </answer> tags."

    TYPE_TEMPLATE = {
        "multiple choice": "Question: \n{Question}\n Options: \n{Option} Please provide the thinking process within the <think> </think> tags. \nPlease provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "regression": "Question:  \n{Question}\n Please provide the thinking process within the <think> </think> tags. Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
    }

    use_geometry_encoder = 'dinov3'

    for start_index in tqdm(range(0, total_samples, batch_size), desc=f"Rank {rank} Processing"):
        batch_df = df_shard.iloc[start_index:min(start_index + batch_size, total_samples)]
        
        batch_messages = []
        batch_metadata = []
        batch_frames_for_geom = []

        for _, row in batch_df.iterrows():
            try:
                _,_,frames = load_frames_by_id(row['ID'], num_frames)
            except:
                print("ID ", row['ID'], " not found.")
                continue

            if not frames:
                logger.warning(f"No frames found for id {row['ID']}, skipping.")
                continue
            
            if len(frames) % 2:
                frames.append(frames[-1])

            processed_364_list, processed_420_tensor_list = process_and_pad_images([frames], use_geometry_encoder)
            frames_364 = processed_364_list[0]
            geom_tensor = processed_420_tensor_list[0]
            question_full = row['Question']

            format_type = "multiple choice"

            question = question_full
            options_text = row['Candidates']

            prompt_text = "\n" + QUESTION_TEMPLATE + TYPE_TEMPLATE["multiple choice"].format(Question=question, Option=options_text)
                
            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": frames_364},
                    {"type": "text", "text": prompt_text},
                ],
            }]
            batch_messages.append(messages)
            batch_metadata.append(row)
            batch_frames_for_geom.append(geom_tensor)

        if not batch_messages:
            continue

        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        _, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=texts,
            videos=video_inputs,
            geometry_encoder_inputs=batch_frames_for_geom,
            padding=True,
            return_tensors="pt"
        ).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=512, do_sample=False)
        trimmed_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        predictions = processor.batch_decode(trimmed_ids, skip_special_tokens=True)
        
        for i, (row, pred) in enumerate(zip(batch_metadata, predictions)):
            results.append({
                'id': row['ID'],
                'dataset': 'sti-bench',
                'question': row['Question'],
                'ground_truth': row['Answer'],
                'predicted_answer': pred.strip(),
                'question_type': format_type,
                'prompt': texts[i],
                'task': row['Task'],
            })
            
    process_output_file = os.path.join(output_dir, f"results_rank_{rank}.jsonl")
    with open(process_output_file, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
            
    elapsed_time = time.time() - start_time_process
    logger.info(f"Rank {rank} finished in {format_time(elapsed_time)}. Results saved to {process_output_file}")

    return process_output_file


def merge_results(world_size: int, output_dir: str, final_filename: str):
    final_output_path = os.path.join(output_dir, final_filename)
    with open(final_output_path, 'w') as outfile:
        for rank in range(world_size):
            process_file = os.path.join(output_dir, f"results_rank_{rank}.jsonl")
            if os.path.exists(process_file):
                with open(process_file, 'r') as infile:
                    for line in infile:
                        outfile.write(line)
                os.remove(process_file)
    print(f"Merged results into {final_output_path}")

def main():
    parser = argparse.ArgumentParser(description="Run evaluation on spatial reasoning video benchmarks.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model or directory with checkpoints.")
    parser.add_argument("--dataset", type=str, required=True, choices=["vsibench", "mmsibench", "spacevista","sparbench","stibench"], help="Name of the dataset to evaluate.")
    parser.add_argument("--output_dir", type=str, default="./eval_results", help="Directory to save results and logs.")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3", help="Comma-separated list of GPU IDs to use (e.g., '0,1,2').")
    parser.add_argument("--num_processes", type=int, default=4, help="Number of parallel processes to use.")
    parser.add_argument("--num_frames", type=int, default=32, help="Number of frames to sample from each video.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode on a small subset of data.")
    parser.add_argument("--debug_size", type=int, default=20, help="Number of samples to use in debug mode.")
    args = parser.parse_args()

    DATASET_CONFIGS = {
        "vsibench": {
            "dataset_path": "your_path/vsi-bench/test-00000-of-00001.parquet",
            "video_dir": "your_path/vsi-bench",
            "evaluation_fn": evaluate_vsibench,
            "metric_fn": calculate_vsi_metrics,
        },
        "mmsibench": {
            "dataset_path": "your_path/vsi-bench/MMSI_Bench.parquet",
            "video_dir": "",
            "evaluation_fn": evaluate_mmsibench,
            "metric_fn": calculate_mmsi_metrics,
        },
        "spacevista": {
            "dataset_path": "your_path/our_outdoor/self_collect_data/from_pc/unified_qa.jsonl",
            "video_dir": "your_path/our_outdoor/self_collect_data/frames/all",
            "evaluation_fn": evaluate_spacevista,
            "metric_fn": calculate_spacevista_metrics,
        },
        "sparbench": {
            "dataset_path": ["your_path/SPAR-Bench/data/test-00000-of-00004.parquet","your_path/SPAR-Bench/data/test-00001-of-00004.parquet",\
                "your_path/SPAR-Bench/data/test-00002-of-00004.parquet","your_path/SPAR-Bench/data/test-00003-of-00004.parquet"],
            "video_dir": "",
            "evaluation_fn": evaluate_sparbench,
            "metric_fn": calculate_spar_metrics,
        },
        "stibench": {
            "dataset_path": "your_path/sti-bench/qa.parquet",
            "video_dir": "",
            "evaluation_fn": evaluate_stibench,
            "metric_fn": calculate_sti_metrics,
        }
    }

    set_start_method('spawn', force=True)
    main_start_time = time.time()
    
    config = DATASET_CONFIGS[args.dataset]
    
    model_path = args.model_path
    if "checkpoint-" in os.path.basename(model_path):
        model_name = model_path
    else:
        checkpoint_folders = glob.glob(os.path.join(model_path, "checkpoint-*"))
        if checkpoint_folders:
            latest_folder = max(checkpoint_folders, key=lambda x: int(x.split("-")[-1]))
            model_name = os.path.join(model_path, latest_folder)
            print(f"Found checkpoints. Using the latest: {os.path.basename(latest_folder)}")
        else:
            model_name = model_path
            print(f"No checkpoint folders found. Using path directly: {model_name}")

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = os.path.basename(os.path.normpath(model_name))
    output_dir = os.path.join(args.output_dir, args.dataset, model_id, timestamp_str)
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, "evaluation.log")
    final_results_file = f"final_results_{args.dataset}.jsonl"

    params_to_log = {
        "model_name": model_name,
        "dataset": args.dataset,
        "num_frames": args.num_frames,
        "target_resolution": (448, 448),
        "debug_mode": args.debug,
        "batch_size": args.batch_size,
        "debug_size": args.debug_size,
        "gpu_ids": args.gpu_ids,
        "num_processes": args.num_processes
    }
    
    evaluation_function = config["evaluation_fn"]
    
    if args.num_processes > 1:
        with mp.Pool(processes=args.num_processes) as pool:
            pool.starmap(evaluation_function, [
                (rank, args.num_processes, config["dataset_path"], config["video_dir"],
                 model_name, output_dir, log_file, args.gpu_ids, args.num_frames,
                 (448, 448), args.debug, args.batch_size, args.debug_size, params_to_log)
                for rank in range(args.num_processes)
            ])
        merge_results(args.num_processes, output_dir, final_results_file)
    else:
        process_output_file = evaluation_function(
            0, 1, config["dataset_path"], config["video_dir"], model_name, output_dir,
            log_file, args.gpu_ids, args.num_frames, (448, 448), args.debug,
            args.batch_size, args.debug_size, params_to_log)
        os.rename(process_output_file, os.path.join(output_dir, final_results_file))

    main_end_time = time.time()
    print(f"\nEvaluation finished. Total time: {format_time(main_end_time - main_start_time)}")
    
    final_results_path = os.path.join(output_dir, final_results_file)
    if os.path.exists(final_results_path):
        print(f"\nCalculating metrics from: {final_results_path}")
        metric_function = config["metric_fn"]
        evaluation_results = metric_function(final_results_path)
        
        print("\n--- Evaluation Results ---")
        print(json.dumps(evaluation_results, indent=2))
        print("--------------------------")
        
        log_str = f"Final metrics for {args.dataset}:\n{json.dumps(evaluation_results, indent=2)}"
        with open(log_file, 'a') as f:
            f.write("\n" + log_str)
    else:
        print("Could not find final results file to calculate metrics.")

if __name__ == "__main__":
    main()