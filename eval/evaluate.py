"""
SpaceVista Evaluation Harness
Evaluates Qwen2.5-VL (or compatible) models on 5 spatial reasoning benchmarks:
  vsibench | mmsibench | spacevista | sparbench | stibench
"""

import argparse
import glob
import io
import json
import logging
import multiprocessing as mp
import os
import re
import time
from datetime import datetime, timedelta
from multiprocessing import set_start_method
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from natsort import natsorted
from PIL import Image
from tqdm import tqdm

from metrics import (
    calculate_mmsi_metrics,
    calculate_spar_metrics,
    calculate_spacevista_metrics,
    calculate_sti_metrics,
    calculate_vsi_metrics,
)

# Heavy inference imports are deferred into _import_inference_deps() so that
# --help and --metrics_only work without GPU / torchvision installed.
def _import_inference_deps():
    """Lazily import packages only needed for model inference."""
    global Accelerator, AutoProcessor, Qwen2_5_VLForConditionalGeneration, process_vision_info
    from accelerate import Accelerator  # noqa: F811
    from qwen_vl_utils import process_vision_info  # noqa: F811
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: F811

# ---------------------------------------------------------------------------
# Prompt templates (evaluation-only; kept here, not in metrics/)
# ---------------------------------------------------------------------------

QUESTION_TEMPLATE = (
    "Please think about this question as if you were a human pondering deeply.\n"
    "Please give your final answer between the <answer> </answer> tags."
)

TYPE_TEMPLATE = {
    "multiple choice": (
        "Question: \n{Question}\n Options: \n{Option} "
        "Please provide the thinking process within the <think> </think> tags. \n"
        "Please provide only the single option letter (e.g., A, B, C, D, etc.) "
        "within the <answer> </answer> tags."
    ),
    "regression": (
        "Question:  \n{Question}\n "
        "Please provide the thinking process within the <think> </think> tags. "
        "Please provide the numerical value (e.g., 42 or 3.14) "
        "within the <answer> </answer> tags."
    ),
}

# VSI-Bench question type constants (used for prompt routing only)
_MCA_QT = {
    "object_rel_direction_easy", "object_rel_direction_medium",
    "object_rel_direction_hard", "object_rel_distance",
    "route_planning", "obj_appearance_order",
}
_NA_QT = {
    "object_abs_distance", "object_counting",
    "object_size_estimation", "room_size_estimation",
}

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def format_time(elapsed_seconds: float) -> str:
    td = timedelta(seconds=int(elapsed_seconds))
    h = td.seconds // 3600
    m = (td.seconds % 3600) // 60
    s = td.seconds % 60
    return f"{h:02}h{m:02}m{s:02}s"


def setup_logger(rank: int, log_file: str, params: dict) -> logging.Logger:
    """Per-process logger that writes to a timestamped file."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = log_file.replace(".log", f"_{ts}_rank_{rank}.log")
    logging.basicConfig(
        filename=path, level=logging.INFO,
        format=f"%(asctime)s - [Rank {rank}] - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting process rank={rank}")
    for k, v in params.items():
        logger.info(f"  {k}: {v}")
    return logger


def allocate_gpu(rank: int, gpu_ids: str, world_size: int) -> str:
    """Round-robin GPU allocation across processes. Also triggers lazy inference imports."""
    _import_inference_deps()
    gpu_list = [g.strip() for g in gpu_ids.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_list)
    if world_size > len(gpu_list):
        logging.warning(f"Rank {rank}: fewer GPUs than processes; GPUs will be shared.")
    selected = gpu_list[rank % len(gpu_list)]
    if world_size > 1:
        torch.cuda.set_device(int(selected))
    logging.getLogger(__name__).info(f"Rank {rank}: GPU={selected}")
    return selected


def resize_image(image: Image.Image, max_size: int = 448) -> Image.Image:
    """Resize keeping aspect ratio so the longest side equals max_size."""
    w, h = image.size
    if max(h, w) <= max_size:
        return image
    if h > w:
        return image.resize((int(w * max_size / h), max_size), Image.Resampling.LANCZOS)
    return image.resize((max_size, int(h * max_size / w)), Image.Resampling.LANCZOS)


def load_video_frames(video_path: str, num_frames: int = 16,
                      target_resolution: tuple = (448, 448)) -> Optional[List[Image.Image]]:
    """Sample num_frames uniformly from a video file using decord."""
    try:
        from decord import VideoReader, cpu as dcpu
        vr = VideoReader(video_path, ctx=dcpu())
        indices = np.linspace(0, len(vr) - 1, num_frames, dtype=int)
        return [
            resize_image(Image.fromarray(f), max(target_resolution))
            for f in vr.get_batch(indices).asnumpy()
        ]
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to load video {video_path}: {e}")
        return None


def crop_center_and_resize(x, ratio: float = 0.8):
    """
    Center-crop by `ratio` and resize back to original dimensions.
    Supports PIL.Image and torch.Tensor (C,H,W) or (N,C,H,W).
    """
    if not (0 < ratio <= 1.0):
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")

    if isinstance(x, Image.Image):
        w, h = x.size
        crop_w, crop_h = max(1, int(round(w * ratio))), max(1, int(round(h * ratio)))
        left, top = (w - crop_w) // 2, (h - crop_h) // 2
        return x.crop((left, top, left + crop_w, top + crop_h)).resize((w, h), Image.Resampling.BICUBIC)

    if not torch.is_tensor(x):
        raise TypeError(f"Expected PIL.Image or torch.Tensor, got {type(x)}")

    if x.ndim == 3:
        C, H, W = x.shape
        ch, cw = max(1, int(round(H * ratio))), max(1, int(round(W * ratio)))
        t, l = (H - ch) // 2, (W - cw) // 2
        return F.interpolate(x[:, t:t+ch, l:l+cw].unsqueeze(0), (H, W),
                             mode="bilinear", align_corners=False).squeeze(0)

    if x.ndim == 4:
        N, C, H, W = x.shape
        ch, cw = max(1, int(round(H * ratio))), max(1, int(round(W * ratio)))
        t, l = (H - ch) // 2, (W - cw) // 2
        return F.interpolate(x[:, :, t:t+ch, l:l+cw], (H, W),
                             mode="bilinear", align_corners=False)

    raise ValueError(f"Tensor must be (C,H,W) or (N,C,H,W), got {tuple(x.shape)}")


def process_and_pad_images(
    frame_batches: List[List[Image.Image]],
    extra_type: str,
) -> Tuple[List[List[Image.Image]], List[torch.Tensor]]:
    """
    Preprocess frame batches for Qwen2.5-VL + geometry encoder.

    Returns:
        batches_364: List[List[PIL]] — 364×364 square PIL images (video input)
        batches_420: List[Tensor]   — (N,3,420,420) tensors (geometry encoder input)
    """
    if not frame_batches or not frame_batches[0]:
        raise ValueError("frame_batches cannot be empty")

    def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
        """Convert PIL image to float32 (C,H,W) tensor with values in [0,1]."""
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)
    target_364 = 364
    target_420 = 364 if extra_type == "vggt" else 420

    def ensure_rgb(img: Image.Image) -> Image.Image:
        if img.mode == "RGBA":
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img)
        return img.convert("RGB")

    def compute_dims(w, h, target):
        if w >= h:
            nw, nh = target, round(h * (target / w) / 14) * 14
        else:
            nh, nw = target, round(w * (target / h) / 14) * 14
        return max(14, nw), max(14, nh)

    def pad_to_square_pil(img: Image.Image, size: int) -> Image.Image:
        bg = Image.new("RGB", (size, size), (255, 255, 255))
        w, h = img.size
        if w > size or h > size:
            l, t = max(0, (w - size) // 2), max(0, (h - size) // 2)
            img = img.crop((l, t, l + min(size, w), t + min(size, h)))
            w, h = img.size
        bg.paste(img, ((size - w) // 2, (size - h) // 2))
        return bg

    def to_padded_tensor(img: Image.Image, size: int) -> torch.Tensor:
        nw, nh = compute_dims(*img.size, size)
        t = _pil_to_tensor(img.resize((nw, nh), Image.Resampling.BICUBIC))
        hp, wp = size - t.shape[1], size - t.shape[2]
        if hp < 0 or wp < 0:
            top = max(0, (t.shape[1] - size) // 2)
            left = max(0, (t.shape[2] - size) // 2)
            t = t[:, top:top+size, left:left+size]
            hp, wp = 0, 0
        if hp > 0 or wp > 0:
            pt, pb = hp // 2, hp - hp // 2
            pl, pr = wp // 2, wp - wp // 2
            t = F.pad(t, (pl, pr, pt, pb), value=1.0)
        return t

    batches_364, batches_420 = [], []
    for frames in frame_batches:
        pils_364, tensors_420 = [], []
        for img in frames:
            img = ensure_rgb(img)
            nw, nh = compute_dims(*img.size, target_364)
            img_r = img.resize((nw, nh), Image.Resampling.BICUBIC)
            p364 = crop_center_and_resize(pad_to_square_pil(img_r, target_364), ratio=0.7)
            t420 = crop_center_and_resize(to_padded_tensor(img, target_420), ratio=0.7)
            pils_364.append(p364)
            tensors_420.append(t420)
        batches_364.append(pils_364)
        batches_420.append(torch.stack(tensors_420))
    return batches_364, batches_420


def load_frames_by_id(
    raw_id: Union[str, int],
    frame_num: int,
    root: str,
    exts: tuple = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"),
) -> Tuple[str, List[Path], List[Image.Image]]:
    """
    Load frames from a zero-padded (6-digit) subdirectory under `root`.
    Uniformly samples frame_num frames; returns (padded_id, paths, images).
    """
    padded_id = str(raw_id).zfill(6)
    dir_path = Path(root) / padded_id

    if not dir_path.is_dir():
        raise FileNotFoundError(f"Frame directory not found: {dir_path}")

    def _natural_key(p: Path):
        parts = re.split(r"(\d+)", p.stem)
        return [int(t) if t.isdigit() else t.lower() for t in parts] + [p.suffix.lower()]

    frame_paths = sorted(
        [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in exts],
        key=_natural_key,
    )
    if not frame_paths:
        raise FileNotFoundError(f"No image files in: {dir_path}")

    n = len(frame_paths)
    if frame_num is None or frame_num >= n:
        indices = list(range(n))
    elif frame_num == 1:
        indices = [n // 2]
    else:
        step = (n - 1) / (frame_num - 1)
        seen, dedup = set(), []
        for i in range(frame_num):
            idx = round(i * step)
            if idx not in seen:
                dedup.append(idx)
                seen.add(idx)
        i = 0
        while len(dedup) < frame_num:
            cand = min(n - 1, max(0, round(i * step)))
            if cand not in seen:
                dedup.append(cand)
                seen.add(cand)
            i += 1
        indices = sorted(dedup)

    sampled = [frame_paths[i] for i in indices]
    images = []
    for fp in sampled:
        with Image.open(fp) as im:
            images.append(im.copy())
    return padded_id, sampled, images


def merge_results(world_size: int, output_dir: str, final_filename: str) -> None:
    """Merge per-rank JSONL files into a single final file and remove the shards."""
    final_path = os.path.join(output_dir, final_filename)
    with open(final_path, "w") as out:
        for rank in range(world_size):
            shard = os.path.join(output_dir, f"results_rank_{rank}.jsonl")
            if os.path.exists(shard):
                with open(shard) as f:
                    out.write(f.read())
                os.remove(shard)
    print(f"Merged results → {final_path}")


# ---------------------------------------------------------------------------
# Model loading helper
# ---------------------------------------------------------------------------

def _load_model(model_name: str, gpu_ids: str, world_size: int, cpu_only: bool):
    """Load Qwen2.5-VL model and processor with appropriate settings."""
    _import_inference_deps()
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    processor.tokenizer.padding_side = "left"

    attn_impl = "eager" if cpu_only else "flash_attention_2"
    dtype = torch.float32 if cpu_only else torch.bfloat16
    multi_gpu_single_proc = (world_size == 1 and len(gpu_ids.split(",")) > 1)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        device_map="auto" if multi_gpu_single_proc else None,
    )
    return processor, model, multi_gpu_single_proc


def _run_inference(model, processor, batch_messages, batch_geom, device,
                   cpu_only: bool, no_geometry_encoder: bool):
    """Run batched inference and return decoded string predictions."""
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in batch_messages
    ]
    _, video_inputs = process_vision_info(batch_messages)

    proc_kwargs = dict(text=texts, videos=video_inputs, padding=True, return_tensors="pt")
    if not no_geometry_encoder:
        try:
            inputs = processor(**proc_kwargs, geometry_encoder_inputs=batch_geom).to(device)
        except TypeError:
            inputs = processor(**proc_kwargs).to(device)
    else:
        inputs = processor(**proc_kwargs).to(device)

    with torch.no_grad():
        if not cpu_only:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                gen_ids = model.generate(**inputs, use_cache=True, max_new_tokens=512, do_sample=False)
        else:
            gen_ids = model.generate(**inputs, use_cache=True, max_new_tokens=512, do_sample=False)

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
    return texts, processor.batch_decode(trimmed, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Benchmark evaluators
# ---------------------------------------------------------------------------

def evaluate_vsibench(rank, world_size, dataset_path, video_dir, model_name,
                      output_dir, log_file, gpu_ids, num_frames, target_resolution,
                      debug, batch_size, debug_size, params_dict, cpu_only, no_geometry_encoder):
    """Evaluate on VSI-Bench (parquet dataset with video files)."""
    logger = setup_logger(rank, log_file, params_dict)
    t0 = time.time()

    allocate_gpu(rank, gpu_ids, world_size)
    accelerator = Accelerator()
    device = accelerator.device
    logger.info(f"Rank {rank} device: {device}")

    processor, model, multi_gpu = _load_model(model_name, gpu_ids, world_size, cpu_only)
    if not multi_gpu:
        model.to(device)
    model = accelerator.prepare(model)
    model.eval()

    if isinstance(dataset_path, list):
        df = pd.concat([pd.read_parquet(p) for p in dataset_path], ignore_index=True)
    else:
        df = pd.read_parquet(dataset_path)
    if debug:
        df = df.head(debug_size)
    df_shard = np.array_split(df, world_size)[rank] if world_size > 1 else df
    logger.info(f"Rank {rank} shard: {len(df_shard)} samples")

    total = len(df_shard)
    if total == 0:
        return os.path.join(output_dir, f"results_rank_{rank}.jsonl")

    results = []
    for start in tqdm(range(0, total, batch_size), desc=f"Rank {rank}"):
        batch_df = df_shard.iloc[start:min(start + batch_size, total)]
        batch_messages, batch_meta, batch_geom = [], [], []

        for _, row in batch_df.iterrows():
            video_path = os.path.join(video_dir, row["dataset"], f"{row['scene_name']}.mp4")
            if not os.path.exists(video_path):
                logger.warning(f"Video not found: {video_path}")
                continue
            frames = load_video_frames(video_path, num_frames, target_resolution)
            if frames is None:
                continue

            frames_364, geom_tensors = process_and_pad_images([frames], "dinov3")
            frames_input = frames_364[0]
            geom_tensor = geom_tensors[0]

            qt = row["question_type"]
            question = row["question"]
            if qt in _NA_QT:
                prompt = "\n" + QUESTION_TEMPLATE + TYPE_TEMPLATE["regression"].format(Question=question)
            elif qt in _MCA_QT:
                opts = " ".join(row.get("options", []))
                prompt = "\n" + QUESTION_TEMPLATE + TYPE_TEMPLATE["multiple choice"].format(
                    Question=question, Option=opts)
            else:
                prompt = "\n" + QUESTION_TEMPLATE + "Question: \n" + question

            batch_messages.append([{"role": "user", "content": [
                {"type": "video", "video": frames_input},
                {"type": "text", "text": prompt},
            ]}])
            batch_meta.append(row)
            batch_geom.append(geom_tensor)

        if not batch_messages:
            continue

        texts, predictions = _run_inference(model, processor, batch_messages, batch_geom,
                                            device, cpu_only, no_geometry_encoder)
        for i, (row, pred) in enumerate(zip(batch_meta, predictions)):
            results.append({
                "id": row["id"],
                "dataset": row["dataset"],
                "scene_name": row["scene_name"],
                "question": row["question"],
                "ground_truth": row["ground_truth"],
                "predicted_answer": pred.strip(),
                "question_type": row["question_type"],
                "prompt": texts[i],
            })

    out_file = os.path.join(output_dir, f"results_rank_{rank}.jsonl")
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    logger.info(f"Rank {rank} done in {format_time(time.time() - t0)}. → {out_file}")
    return out_file


def evaluate_mmsibench(rank, world_size, dataset_path, video_dir, model_name,
                       output_dir, log_file, gpu_ids, num_frames, target_resolution,
                       debug, batch_size, debug_size, params_dict, cpu_only, no_geometry_encoder):
    """Evaluate on MMSI-Bench (images stored as bytes in parquet)."""
    logger = setup_logger(rank, log_file, params_dict)
    t0 = time.time()

    allocate_gpu(rank, gpu_ids, world_size)
    accelerator = Accelerator()
    device = accelerator.device

    processor, model, multi_gpu = _load_model(model_name, gpu_ids, world_size, cpu_only)
    if not multi_gpu:
        model.to(device)
    model = accelerator.prepare(model)
    model.eval()

    df = pd.read_parquet(dataset_path)
    if debug:
        df = df.head(debug_size)
    df_shard = np.array_split(df, world_size)[rank] if world_size > 1 else df
    total = len(df_shard)
    if total == 0:
        return os.path.join(output_dir, f"results_rank_{rank}.jsonl")

    results = []
    for start in tqdm(range(0, total, batch_size), desc=f"Rank {rank}"):
        batch_df = df_shard.iloc[start:min(start + batch_size, total)]
        batch_messages, batch_meta, batch_geom = [], [], []

        for _, row in batch_df.iterrows():
            frames = [
                resize_image(Image.open(io.BytesIO(b)), max(target_resolution))
                for b in row["images"]
            ]
            if not frames:
                continue

            frames_364, geom_tensors = process_and_pad_images([frames], "dinov3")
            frames_input = frames_364[0]
            geom_tensor = geom_tensors[0]
            # Double frames to meet even-frame requirement
            frames_input = [x for x in frames_input for _ in range(2)]
            geom_tensor = geom_tensor.repeat_interleave(2, dim=0)

            qfull = row["question"]
            try:
                question, options_text = qfull.split("\nOptions: ")
                options_text = options_text.strip()
            except ValueError:
                question, options_text = qfull, row.get("options", "")

            prompt = "\n" + QUESTION_TEMPLATE + TYPE_TEMPLATE["multiple choice"].format(
                Question=question, Option=options_text)
            batch_messages.append([{"role": "user", "content": [
                {"type": "video", "video": frames_input},
                {"type": "text", "text": prompt},
            ]}])
            batch_meta.append(row)
            batch_geom.append(geom_tensor)

        if not batch_messages:
            continue

        texts, predictions = _run_inference(model, processor, batch_messages, batch_geom,
                                            device, cpu_only, no_geometry_encoder)
        for i, (row, pred) in enumerate(zip(batch_meta, predictions)):
            results.append({
                "id": row["id"],
                "dataset": "mmsi-bench",
                "question": row["question"],
                "ground_truth": row["answer"],
                "predicted_answer": pred.strip(),
                "question_type": row["question_type"],
                "prompt": texts[i],
            })

    out_file = os.path.join(output_dir, f"results_rank_{rank}.jsonl")
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    logger.info(f"Rank {rank} done in {format_time(time.time() - t0)}. → {out_file}")
    return out_file


def evaluate_spacevista(rank, world_size, dataset_path, frames_root_dir, model_name,
                        output_dir, log_file, gpu_ids, num_frames, target_resolution,
                        debug, batch_size, debug_size, params_dict, cpu_only, no_geometry_encoder):
    """Evaluate on SpaceVista-Bench (JSONL metadata + frame directories)."""
    logger = setup_logger(rank, log_file, params_dict)
    t0 = time.time()

    allocate_gpu(rank, gpu_ids, world_size)
    accelerator = Accelerator()
    device = accelerator.device

    processor, model, multi_gpu = _load_model(model_name, gpu_ids, world_size, cpu_only)
    if not multi_gpu:
        model.to(device)
    model = accelerator.prepare(model)
    model.eval()

    with open(dataset_path) as f:
        df = pd.DataFrame([json.loads(l) for l in f])
    if debug:
        df = df.head(debug_size)
    df_shard = np.array_split(df, world_size)[rank] if world_size > 1 else df
    total = len(df_shard)
    if total == 0:
        return os.path.join(output_dir, f"results_rank_{rank}.jsonl")

    results = []
    for _, row in tqdm(df_shard.iterrows(), total=total, desc=f"Rank {rank}"):
        frames_dir = os.path.join(frames_root_dir, row["Frames Path"])
        if not os.path.isdir(frames_dir):
            logger.warning(f"Frames dir not found: {frames_dir}")
            continue

        image_files = natsorted(os.listdir(frames_dir))
        if not image_files:
            continue

        indices = np.linspace(0, len(image_files) - 1, num_frames, dtype=int)
        frames = []
        for i in indices:
            try:
                frames.append(resize_image(
                    Image.open(os.path.join(frames_dir, image_files[i])).convert("RGB"),
                    max(target_resolution),
                ))
            except Exception as e:
                logger.error(f"Error loading frame: {e}")
        if not frames:
            continue

        frames_364, geom_tensors = process_and_pad_images([frames], "dinov3")
        frames_input = frames_364[0]
        geom_tensor = geom_tensors[0]

        question = row["Question"]
        qt = row["Question Type"]
        if qt == "regression":
            prompt = "\n" + QUESTION_TEMPLATE + TYPE_TEMPLATE["regression"].format(Question=question)
        elif qt == "multiple choice":
            opts = " ".join(row["options"])
            prompt = "\n" + QUESTION_TEMPLATE + TYPE_TEMPLATE["multiple choice"].format(
                Question=question, Option=opts)
        else:
            prompt = "\n" + QUESTION_TEMPLATE + "Question: \n" + question

        messages = [{"role": "user", "content": [
            {"type": "video", "video": frames_input},
            {"type": "text", "text": prompt},
        ]}]
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        _, video_inputs = process_vision_info([messages])

        proc_kwargs = dict(
            text=[text_prompt], videos=video_inputs, padding=True, return_tensors="pt"
        )
        if not no_geometry_encoder:
            try:
                inputs = processor(**proc_kwargs, geometry_encoder_inputs=[geom_tensor]).to(device)
            except TypeError:
                inputs = processor(**proc_kwargs).to(device)
        else:
            inputs = processor(**proc_kwargs).to(device)

        with torch.no_grad():
            if not cpu_only:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out_ids = model.generate(**inputs, use_cache=True, max_new_tokens=512, do_sample=False)
            else:
                out_ids = model.generate(**inputs, use_cache=True, max_new_tokens=512, do_sample=False)

        gen = out_ids[0][inputs.input_ids.shape[1]:]
        prediction = processor.decode(gen, skip_special_tokens=True).strip()

        results.append({
            "Question Number": row["Question Number"],
            "Scene Source": row["Scene Source"],
            "Frames Path": row["Frames Path"],
            "question": row["Question"],
            "ground_truth": row["Answer"],
            "predicted_answer": prediction,
            # question_type ← TaskType (fine-grained category e.g. EXISTENCE, COUNTING)
            "question_type": row["TaskType"],
            # task_type ← Question Type (multiple choice / regression) for grouping
            "task_type": row["Question Type"],
            "prompt": text_prompt,
        })

    out_file = os.path.join(output_dir, f"results_rank_{rank}.jsonl")
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    logger.info(f"Rank {rank} done in {format_time(time.time() - t0)}. → {out_file}")
    return out_file


def evaluate_sparbench(rank, world_size, dataset_path, frames_root_dir, model_name,
                       output_dir, log_file, gpu_ids, num_frames, target_resolution,
                       debug, batch_size, debug_size, params_dict, cpu_only, no_geometry_encoder):
    """Evaluate on SPAR-Bench (images stored as bytes in parquet)."""
    logger = setup_logger(rank, log_file, params_dict)
    t0 = time.time()

    allocate_gpu(rank, gpu_ids, world_size)
    accelerator = Accelerator()
    device = accelerator.device

    processor, model, multi_gpu = _load_model(model_name, gpu_ids, world_size, cpu_only)
    if not multi_gpu:
        model.to(device)
    model = accelerator.prepare(model)
    model.eval()

    if isinstance(dataset_path, list):
        df = pd.concat([pd.read_parquet(p) for p in dataset_path], ignore_index=True)
    else:
        df = pd.read_parquet(dataset_path)
    if debug:
        df = df.head(debug_size)
    df_shard = np.array_split(df, world_size)[rank] if world_size > 1 else df
    total = len(df_shard)
    if total == 0:
        return os.path.join(output_dir, f"results_rank_{rank}.jsonl")

    results = []
    for start in tqdm(range(0, total, batch_size), desc=f"Rank {rank}"):
        batch_df = df_shard.iloc[start:min(start + batch_size, total)]
        batch_messages, batch_meta, batch_geom = [], [], []

        for _, row in batch_df.iterrows():
            frames = [
                resize_image(Image.open(io.BytesIO(b["bytes"])), max(target_resolution))
                for b in row["image"]
            ]
            if not frames:
                continue

            frames_364, geom_tensors = process_and_pad_images([frames], "dinov3")
            frames_input = frames_364[0]
            geom_tensor = geom_tensors[0]

            # Normalize format_type field
            fmt = row["format_type"]
            if fmt == "select":
                fmt = "multiple choice"
            elif fmt == "fill":
                fmt = "regression"

            question = row["question"]
            if row["task"] in {"position_matching", "camera_motion_infer"}:
                question += "\nThe values represent the bounding box coordinates normalized to a 0-1000 scale, with the top-left corner as the origin of the image."

            if fmt == "multiple choice":
                prompt = "\n" + QUESTION_TEMPLATE + TYPE_TEMPLATE["multiple choice"].format(
                    Question=question, Option=question)  # SPAR embeds options in question text
            else:
                prompt = "\n" + QUESTION_TEMPLATE + TYPE_TEMPLATE["regression"].format(Question=question)

            # Ensure even frame count (required by Qwen video processor)
            if len(frames_input) <= 3:
                frames_input = [x for x in frames_input for _ in range(2)]
                geom_tensor = geom_tensor.repeat_interleave(2, dim=0)
            elif len(frames_input) % 2:
                frames_input.append(frames_input[-1])
                geom_tensor = torch.cat([geom_tensor, geom_tensor[-1:]], dim=0)

            batch_messages.append([{"role": "user", "content": [
                {"type": "video", "video": frames_input},
                {"type": "text", "text": prompt},
            ]}])
            batch_meta.append((row, fmt))
            batch_geom.append(geom_tensor)

        if not batch_messages:
            continue

        texts, predictions = _run_inference(model, processor, batch_messages, batch_geom,
                                            device, cpu_only, no_geometry_encoder)
        for i, ((row, fmt), pred) in enumerate(zip(batch_meta, predictions)):
            results.append({
                "id": row["id"],
                "dataset": "spar-bench",
                "question": row["question"],
                "ground_truth": row["answer"],
                "predicted_answer": pred.strip(),
                "question_type": fmt,
                "task": row["task"],
                "image_type": row["img_type"],
                "prompt": texts[i],
            })

    out_file = os.path.join(output_dir, f"results_rank_{rank}.jsonl")
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    logger.info(f"Rank {rank} done in {format_time(time.time() - t0)}. → {out_file}")
    return out_file


def evaluate_stibench(rank, world_size, dataset_path, frames_root_dir, model_name,
                      output_dir, log_file, gpu_ids, num_frames, target_resolution,
                      debug, batch_size, debug_size, params_dict, cpu_only, no_geometry_encoder):
    """Evaluate on STI-Bench (parquet QA + frame directories indexed by ID)."""
    logger = setup_logger(rank, log_file, params_dict)
    t0 = time.time()

    allocate_gpu(rank, gpu_ids, world_size)
    accelerator = Accelerator()
    device = accelerator.device

    processor, model, multi_gpu = _load_model(model_name, gpu_ids, world_size, cpu_only)
    if not multi_gpu:
        model.to(device)
    model = accelerator.prepare(model)
    model.eval()

    df = pd.read_parquet(dataset_path)
    if debug:
        df = df.head(debug_size)
    df_shard = np.array_split(df, world_size)[rank] if world_size > 1 else df
    total = len(df_shard)
    if total == 0:
        return os.path.join(output_dir, f"results_rank_{rank}.jsonl")

    results = []
    for start in tqdm(range(0, total, batch_size), desc=f"Rank {rank}"):
        batch_df = df_shard.iloc[start:min(start + batch_size, total)]
        batch_messages, batch_meta, batch_geom = [], [], []

        for _, row in batch_df.iterrows():
            try:
                _, _, frames = load_frames_by_id(row["ID"], num_frames, frames_root_dir)
            except FileNotFoundError:
                logger.warning(f"Frames not found for ID {row['ID']}")
                continue
            if not frames:
                continue
            if len(frames) % 2:
                frames.append(frames[-1])

            frames_364, geom_tensors = process_and_pad_images([frames], "dinov3")
            frames_input = frames_364[0]
            geom_tensor = geom_tensors[0]

            question = row["Question"]
            options_text = row["Candidates"]
            prompt = "\n" + QUESTION_TEMPLATE + TYPE_TEMPLATE["multiple choice"].format(
                Question=question, Option=options_text)

            batch_messages.append([{"role": "user", "content": [
                {"type": "video", "video": frames_input},
                {"type": "text", "text": prompt},
            ]}])
            batch_meta.append(row)
            batch_geom.append(geom_tensor)

        if not batch_messages:
            continue

        texts, predictions = _run_inference(model, processor, batch_messages, batch_geom,
                                            device, cpu_only, no_geometry_encoder)
        for i, (row, pred) in enumerate(zip(batch_meta, predictions)):
            results.append({
                "id": row["ID"],
                "dataset": "sti-bench",
                "question": row["Question"],
                "ground_truth": row["Answer"],
                "predicted_answer": pred.strip(),
                "question_type": "multiple choice",
                "task": row["Task"],
                "prompt": texts[i],
            })

    out_file = os.path.join(output_dir, f"results_rank_{rank}.jsonl")
    with open(out_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    logger.info(f"Rank {rank} done in {format_time(time.time() - t0)}. → {out_file}")
    return out_file


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate spatial reasoning video benchmarks with Qwen2.5-VL."
    )
    # Core
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model or checkpoint directory.")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["vsibench", "mmsibench", "spacevista", "sparbench", "stibench"])
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3")
    parser.add_argument("--num_processes", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_size", type=int, default=20)
    # CPU / degraded mode
    parser.add_argument("--cpu_only", action="store_true",
                        help="Use float32 + eager attention for CPU-only runs.")
    parser.add_argument("--no_geometry_encoder", action="store_true",
                        help="Skip geometry_encoder_inputs (for standard HF processors).")
    # Skip inference and score an existing JSONL
    parser.add_argument("--metrics_only", type=str, default=None, metavar="JSONL",
                        help="Score a pre-existing prediction JSONL; skip inference.")
    # Dataset paths (env-var fallback, then hardcoded defaults)
    parser.add_argument("--vsibench_dataset", type=str,
                        default=os.environ.get("VSIBENCH_DATASET",
                            "/kpfs-intern/sunpeiwen/dataset/dataset/vsi-bench/test-00000-of-00001.parquet"))
    parser.add_argument("--vsibench_video_dir", type=str,
                        default=os.environ.get("VSIBENCH_VIDEO_DIR",
                            "/kpfs-intern/sunpeiwen/dataset/dataset/vsi-bench"))
    parser.add_argument("--mmsibench_dataset", type=str,
                        default=os.environ.get("MMSIBENCH_DATASET",
                            "/kpfs-intern/sunpeiwen/dataset/vsi-bench/MMSI_Bench.parquet"))
    parser.add_argument("--spacevista_dataset", type=str,
                        default=os.environ.get("SPACEVISTA_DATASET",
                            "/kpfs-intern/sunpeiwen/dataset/our_outdoor/self_collect_data/from_pc/unified_qa.jsonl"))
    parser.add_argument("--spacevista_frames_dir", type=str,
                        default=os.environ.get("SPACEVISTA_FRAMES_DIR",
                            "/kpfs-intern/sunpeiwen/dataset/our_outdoor/self_collect_data/frames/all"))
    parser.add_argument("--sparbench_dataset", type=str, nargs="+",
                        default=os.environ.get("SPARBENCH_DATASET", "").split(",") if os.environ.get("SPARBENCH_DATASET") else [
                            "/kpfs-intern/sunpeiwen/dataset/SPAR-Bench/data/test-00000-of-00004.parquet",
                            "/kpfs-intern/sunpeiwen/dataset/SPAR-Bench/data/test-00001-of-00004.parquet",
                            "/kpfs-intern/sunpeiwen/dataset/SPAR-Bench/data/test-00002-of-00004.parquet",
                            "/kpfs-intern/sunpeiwen/dataset/SPAR-Bench/data/test-00003-of-00004.parquet",
                        ])
    parser.add_argument("--stibench_dataset", type=str,
                        default=os.environ.get("STIBENCH_DATASET",
                            "/kpfs-intern/sunpeiwen/dataset/dataset/sti-bench/qa.parquet"))
    parser.add_argument("--stibench_frames_dir", type=str,
                        default=os.environ.get("STIBENCH_FRAMES_DIR",
                            "/kpfs-intern/sunpeiwen/dataset/dataset/sti-bench/frames"))

    args = parser.parse_args()

    DATASET_CONFIGS = {
        "vsibench": {
            "dataset_path": args.vsibench_dataset,
            "video_dir": args.vsibench_video_dir,
            "evaluation_fn": evaluate_vsibench,
            "metric_fn": calculate_vsi_metrics,
        },
        "mmsibench": {
            "dataset_path": args.mmsibench_dataset,
            "video_dir": "",
            "evaluation_fn": evaluate_mmsibench,
            "metric_fn": calculate_mmsi_metrics,
        },
        "spacevista": {
            "dataset_path": args.spacevista_dataset,
            "video_dir": args.spacevista_frames_dir,
            "evaluation_fn": evaluate_spacevista,
            "metric_fn": calculate_spacevista_metrics,
        },
        "sparbench": {
            "dataset_path": args.sparbench_dataset,
            "video_dir": "",
            "evaluation_fn": evaluate_sparbench,
            "metric_fn": calculate_spar_metrics,
        },
        "stibench": {
            "dataset_path": args.stibench_dataset,
            "video_dir": args.stibench_frames_dir,
            "evaluation_fn": evaluate_stibench,
            "metric_fn": calculate_sti_metrics,
        },
    }

    config = DATASET_CONFIGS[args.dataset]

    # --metrics_only: score a pre-existing JSONL without running inference
    if args.metrics_only:
        print(f"Metrics-only mode: scoring {args.metrics_only}")
        result = config["metric_fn"](args.metrics_only)
        print("\n--- Evaluation Results ---")
        print(json.dumps(result, indent=2))
        print("--------------------------")
        return

    # Resolve model path (handle checkpoint dirs)
    model_path = args.model_path
    if "checkpoint-" in os.path.basename(model_path):
        model_name = model_path
    else:
        ckpts = glob.glob(os.path.join(model_path, "checkpoint-*"))
        if ckpts:
            model_name = max(ckpts, key=lambda x: int(x.split("-")[-1]))
            print(f"Using latest checkpoint: {os.path.basename(model_name)}")
        else:
            model_name = model_path
            print(f"Using model path directly: {model_name}")

    # Setup output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = os.path.basename(os.path.normpath(model_name))
    output_dir = os.path.join(args.output_dir, args.dataset, model_id, ts)
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
        "num_processes": args.num_processes,
        "cpu_only": args.cpu_only,
        "no_geometry_encoder": args.no_geometry_encoder,
    }

    set_start_method("spawn", force=True)
    t0 = time.time()

    eval_fn = config["evaluation_fn"]
    eval_args = (
        config["dataset_path"], config["video_dir"], model_name,
        output_dir, log_file, args.gpu_ids, args.num_frames, (448, 448),
        args.debug, args.batch_size, args.debug_size, params_to_log,
        args.cpu_only, args.no_geometry_encoder,
    )

    if args.num_processes > 1:
        with mp.Pool(processes=args.num_processes) as pool:
            pool.starmap(eval_fn, [
                (rank, args.num_processes) + eval_args
                for rank in range(args.num_processes)
            ])
        merge_results(args.num_processes, output_dir, final_results_file)
    else:
        out_file = eval_fn(0, 1, *eval_args)
        os.rename(out_file, os.path.join(output_dir, final_results_file))

    print(f"\nEvaluation done. Total time: {format_time(time.time() - t0)}")

    final_path = os.path.join(output_dir, final_results_file)
    if os.path.exists(final_path):
        print(f"\nCalculating metrics from: {final_path}")
        metrics = config["metric_fn"](final_path)
        print("\n--- Evaluation Results ---")
        print(json.dumps(metrics, indent=2))
        print("--------------------------")
        with open(log_file, "a") as f:
            f.write(f"\nFinal metrics:\n{json.dumps(metrics, indent=2)}\n")
    else:
        print("Could not find final results file.")


if __name__ == "__main__":
    main()
