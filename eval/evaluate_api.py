"""
SpaceVista-Bench API Evaluation Script.

Evaluates bench_all_clean_corrected.json using VLM APIs (e.g. Qwen2.5-VL-72B via OpenRouter).
No local model loading or GPU required — images are sent as base64 via OpenAI-compatible API.

Usage:
    python evaluate_api.py                           # full run with defaults
    python evaluate_api.py --debug --debug_size 5    # quick test on 5 entries
    python evaluate_api.py --category Indoor          # evaluate one scale only
    python evaluate_api.py --resume results.jsonl     # resume interrupted run
    python evaluate_api.py --metrics_only results.jsonl  # recompute metrics
"""

import argparse
import base64
import io
import json
import logging
import os
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from openai import OpenAI
from tqdm import tqdm

from metrics.spacevista_bench import compute_metrics

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ===================================================================
# A. Answer type classification
# ===================================================================

def classify_answer_type(entry: dict) -> str:
    """Classify answer as 'mc', 'numerical', or 'text'."""
    answer = str(entry["answer"]).strip()
    if answer in ("A", "B", "C", "D", "E"):
        return "mc"
    try:
        float(answer)
        return "numerical"
    except (ValueError, TypeError):
        return "text"


# ===================================================================
# B. Prompt construction
# ===================================================================

def build_prompt(entry: dict) -> str:
    """Build evaluation prompt with appropriate answer format instructions."""
    answer_type = classify_answer_type(entry)
    question = entry["question"]

    if answer_type == "mc":
        suffix = (
            "Please provide the thinking process within the <think> </think> tags.\n"
            "Please provide only the single option letter (e.g., A, B, C, D, etc.) "
            "within the <answer> </answer> tags."
        )
    elif answer_type == "numerical":
        suffix = (
            "Please provide the thinking process within the <think> </think> tags.\n"
            "Please provide the numerical value (e.g., 42 or 3.14) "
            "within the <answer> </answer> tags."
        )
    else:  # text
        suffix = (
            "Please provide the thinking process within the <think> </think> tags.\n"
            "Please provide the answer within the <answer> </answer> tags."
        )

    return f"{question}\n\n{suffix}"


# ===================================================================
# C. Image utilities
# ===================================================================

def resize_image(image: Image.Image, max_size: int = 448) -> Image.Image:
    """Resize image maintaining aspect ratio with longest side as max_size."""
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


def load_and_sample_frames(
    frame_paths: List[str], num_frames: int, max_size: int = 448
) -> Tuple[List[Image.Image], List[int]]:
    """Uniformly sample frames, resize, return PIL images and sampled indices."""
    n = len(frame_paths)
    if n == 0:
        return [], []
    if n <= num_frames:
        indices = list(range(n))
    else:
        indices = np.linspace(0, n - 1, num_frames, dtype=int).tolist()

    frames = []
    for i in indices:
        try:
            img = Image.open(frame_paths[i]).convert("RGB")
            img = resize_image(img, max_size)
            frames.append(img)
        except Exception as e:
            logger.warning(f"Failed to load frame {frame_paths[i]}: {e}")
    return frames, indices


def encode_image(image: Image.Image) -> str:
    """Encode PIL image to base64 JPEG string for API calls."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode()


# ===================================================================
# D. Annotation rendering (ported from serve.py)
# ===================================================================

COLORS_ORDER = ["red", "blue", "green", "yellow"]
COLORS_RGB = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 128, 255),
    "yellow": (255, 215, 0),
}


def overlay_mask_rgba(
    pil_img: Image.Image, mask: np.ndarray,
    color=(255, 0, 0), alpha=0.35,
) -> Image.Image:
    """Overlay a boolean mask on an image with semi-transparent color."""
    if mask is None:
        return pil_img
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if not mask.any():
        return pil_img
    base = pil_img.convert("RGBA")
    color_img = Image.new("RGBA", base.size, color + (int(255 * alpha),))
    mh, mw = mask.shape[:2]
    if (mw, mh) != (base.width, base.height):
        mask_pil = Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (base.width, base.height), resample=Image.NEAREST
        )
    else:
        mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
    return Image.composite(color_img, base, mask_pil).convert("RGB")


def load_mask_array(mask_path: str) -> Optional[np.ndarray]:
    """Load mask array from .pkl/.npy file."""
    if not os.path.isfile(mask_path):
        return None
    try:
        return np.load(mask_path, allow_pickle=True)
    except Exception:
        return None


def select_mask_from_array(
    mask_arr: np.ndarray, object_ids
) -> Optional[np.ndarray]:
    """Select relevant mask channels from a loaded mask array."""
    if mask_arr is None:
        return None
    arr = mask_arr
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2:
        if arr.dtype == bool:
            if object_ids is None:
                return arr
            if isinstance(object_ids, list) and any(bool(x) for x in object_ids):
                return arr
            return np.zeros_like(arr, dtype=bool)
        else:
            if object_ids is None:
                return arr != 0
            ids = []
            for x in (object_ids if isinstance(object_ids, list) else [object_ids]):
                try:
                    ids.append(int(x))
                except Exception:
                    pass
            if not ids:
                return arr != 0
            mask = np.zeros_like(arr, dtype=bool)
            for k in ids:
                mask |= arr == k
            return mask
    if arr.ndim == 3:
        if (arr.shape[0] not in (arr.shape[1],) and arr.shape[0] < 16
                and arr.shape[0] != arr.shape[1]):
            arr = np.moveaxis(arr, 0, 2)
        if arr.dtype != bool:
            arr = arr != 0
        H, W, K = arr.shape
        if object_ids is None:
            return arr.any(axis=2)
        if (isinstance(object_ids, list) and len(object_ids) == K
                and all(isinstance(x, (bool, np.bool_)) for x in object_ids)):
            sel = [i for i, v in enumerate(object_ids) if v]
        else:
            tmp = []
            for x in (object_ids if isinstance(object_ids, list) else [object_ids]):
                if isinstance(x, (bool, np.bool_)):
                    if x:
                        return arr.any(axis=2)
                else:
                    try:
                        tmp.append(int(x))
                    except Exception:
                        pass
            sel = tmp
        if not sel:
            return arr.any(axis=2)
        sel = [i for i in sel if 0 <= i < K]
        if not sel:
            return arr.any(axis=2)
        return arr[:, :, sel].any(axis=2)
    return None


def _scale_point(point, width, height):
    """Scale point from [y, x] format, returns [x, y] or None."""
    if (isinstance(point, (list, tuple)) and len(point) == 1
            and isinstance(point[0], (list, tuple, np.ndarray))):
        point = point[0]
    if not (isinstance(point, (list, tuple, np.ndarray)) and len(point) == 2):
        return None
    y, x = float(point[0]), float(point[1])
    if not (0 <= x < width and 0 <= y < height):
        print(f"[WARN] point ({y},{x}) out of bounds for {width}x{height}")
        return None
    return np.array([x, y], dtype=float)


def _scale_bbox(bbox, width, height):
    """Scale bbox from [x1,y1,x2,y2] array or dict format, returns clipped array."""
    if isinstance(bbox, dict):
        if all(k in bbox for k in ("x1", "y1", "x2", "y2")):
            x1, y1 = float(bbox["x1"]), float(bbox["y1"])
            x2, y2 = float(bbox["x2"]), float(bbox["y2"])
        elif all(k in bbox for k in ("x", "y", "w", "h")):
            x, y, w, h = float(bbox["x"]), float(bbox["y"]), float(bbox["w"]), float(bbox["h"])
            x1, y1, x2, y2 = x, y, x + w, y + h
        else:
            return None
    elif isinstance(bbox, (list, tuple, np.ndarray)):
        arr_b = np.array(bbox, dtype=float).flatten()
        if arr_b.size != 4:
            return None
        x1, y1, x2, y2 = arr_b.tolist()
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
    else:
        return None
    x1 = max(0.0, min(x1, width - 1))
    y1 = max(0.0, min(y1, height - 1))
    x2 = max(0.0, min(x2, width - 1))
    y2 = max(0.0, min(y2, height - 1))
    return np.array([x1, y1, x2, y2], dtype=float)


def _find_frame_index(image_path: str, frame_paths: List[str]) -> int:
    """Find the index of image_path in frame_paths, with fallbacks."""
    if not image_path or not frame_paths:
        return 0
    # Exact match (relative or absolute)
    if image_path in frame_paths:
        return frame_paths.index(image_path)
    # Absolute path match
    abs_path = os.path.abspath(image_path)
    for i, fp in enumerate(frame_paths):
        if os.path.abspath(fp) == abs_path:
            return i
    # images_8 <-> images substitution
    for src, dst in [("/images_8/", "/images/"), ("/images/", "/images_8/")]:
        if src in image_path:
            alt = image_path.replace(src, dst)
            if alt in frame_paths:
                return frame_paths.index(alt)
            alt_abs = os.path.abspath(alt)
            for i, fp in enumerate(frame_paths):
                if os.path.abspath(fp) == alt_abs:
                    return i
    # Basename fallback
    base = os.path.basename(image_path)
    matches = [i for i, fp in enumerate(frame_paths) if os.path.basename(fp) == base]
    if len(matches) >= 1:
        return matches[0]
    return 0


def build_annotation_list(item: Dict[str, Any],
                          frame_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Convert new-format 'annotations' dict into a flat list of drawing
    instructions, each with {type, frame_idx, color, data}.

    Only renders annotations matching the entry's input_type to avoid
    drawing spurious overlays (e.g. masks on bbox-only entries).
    """
    annotations = item.get("annotations") or {}
    out: List[Dict[str, Any]] = []
    input_type = (item.get("input_type") or "").lower()

    # --- Points (only for point-type entries) ---
    if "point" in input_type:
        points = annotations.get("points") or []
        for i, pt in enumerate(points):
            color = pt.get("color") or COLORS_ORDER[i % len(COLORS_ORDER)]
            out.append({
                "type": "point",
                "frame_idx": pt.get("image_index", 0),
                "color": color,
                "data": pt["coords"],  # [y, x]
            })

    # --- Bboxes (only for bbox-type entries) ---
    if "bbox" in input_type:
        bboxes = annotations.get("bboxes") or []
        for i, bb in enumerate(bboxes):
            color = bb.get("color") or COLORS_ORDER[i % len(COLORS_ORDER)]
            out.append({
                "type": "bbox",
                "frame_idx": bb.get("image_index", 0),
                "color": color,
                "data": bb["coords"],  # [x1, y1, x2, y2]
            })

    # --- Masks (only for mask-type entries) ---
    if "mask" in input_type:
        masks = annotations.get("masks") or []
        for i, mask_info in enumerate(masks):
            mask_path = mask_info.get("mask_path")
            object_ids = mask_info.get("object_ids")
            image_path = mask_info.get("image_path")
            frame_idx = _find_frame_index(image_path, frame_paths)

            if mask_path:
                arr = load_mask_array(mask_path)
                if arr is not None:
                    sel = select_mask_from_array(arr, object_ids)
                    if sel is not None:
                        out.append({
                            "type": "mask",
                            "frame_idx": frame_idx,
                            "color": COLORS_ORDER[i % len(COLORS_ORDER)],
                            "data": sel,
                        })

    # --- Dual BBox (only for bbox-type entries) ---
    dual_bbox = annotations.get("dual_bbox")
    if dual_bbox and "bbox" in input_type:
        src_img = dual_bbox.get("source_image")
        src_bbox = dual_bbox.get("source_bbox")
        tgt_img = dual_bbox.get("target_image")
        tgt_bboxes = dual_bbox.get("target_bboxes") or []

        src_idx = _find_frame_index(src_img, frame_paths)
        tgt_idx = _find_frame_index(tgt_img, frame_paths)

        if src_bbox:
            out.append({
                "type": "bbox",
                "frame_idx": src_idx,
                "color": "red",
                "data": src_bbox,
            })
        for j, tb in enumerate(tgt_bboxes):
            out.append({
                "type": "bbox",
                "frame_idx": tgt_idx,
                "color": COLORS_ORDER[(j + 1) % len(COLORS_ORDER)],
                "data": tb,
            })

    return out


def draw_annotations_on_pil(
    img: Image.Image, annotation_list: List[Dict[str, Any]], current_img_idx: int
) -> Image.Image:
    """Draw all annotations that belong to current_img_idx on the image."""
    if not annotation_list:
        return img

    draw = ImageDraw.Draw(img)
    W, H = img.width, img.height
    stroke = max(2, int(0.004 * min(W, H)))

    for ann in annotation_list:
        if ann["frame_idx"] != current_img_idx:
            continue
        rgb = COLORS_RGB.get(ann["color"], (255, 0, 0))

        if ann["type"] == "point":
            xy = _scale_point(ann["data"], W, H)
            if xy is not None:
                x, y = float(xy[0]), float(xy[1])
                r = max(8, int(0.014 * min(W, H)))
                draw.ellipse(
                    (x - r, y - r, x + r, y + r),
                    fill=rgb, outline=rgb, width=2,
                )

        elif ann["type"] == "bbox":
            result = _scale_bbox(ann["data"], W, H)
            if result is not None:
                x1, y1, x2, y2 = result
                if x1 < W and y1 < H and x2 > 0 and y2 > 0:
                    draw.rectangle((x1, y1, x2, y2), outline=rgb, width=stroke)

    # Masks overlay last
    for ann in annotation_list:
        if ann["frame_idx"] != current_img_idx:
            continue
        if ann["type"] == "mask":
            rgb = COLORS_RGB.get(ann["color"], (255, 0, 0))
            img = overlay_mask_rgba(img, ann["data"], color=rgb, alpha=0.35)

    return img


def render_annotations(
    entry: dict,
    frames: List[Image.Image],
    sampled_indices: List[int],
    all_frame_paths: List[str],
) -> List[Image.Image]:
    """Draw annotations on the appropriate sampled frames."""
    input_type = entry.get("input_type", "")
    if input_type == "image_text":
        return frames

    # Build annotation list using original (full) frame paths
    ann_list = build_annotation_list(entry, all_frame_paths)
    if not ann_list:
        return frames

    # Map original frame indices to sampled frame indices
    idx_map = {orig: sampled for sampled, orig in enumerate(sampled_indices)}

    # Re-map annotation frame_idx to sampled indices (nearest match)
    for ann in ann_list:
        orig_idx = ann["frame_idx"]
        if orig_idx in idx_map:
            ann["frame_idx"] = idx_map[orig_idx]
        else:
            # Find nearest sampled index
            nearest = min(sampled_indices, key=lambda x: abs(x - orig_idx))
            ann["frame_idx"] = idx_map[nearest]

    # Copy frames and draw annotations
    frames = [f.copy() for f in frames]
    for i in range(len(frames)):
        frames[i] = draw_annotations_on_pil(frames[i], ann_list, i)

    return frames


# ===================================================================
# E. API call
# ===================================================================

def call_api(
    client: OpenAI,
    model: str,
    frames: List[Image.Image],
    prompt: str,
    max_retries: int = 3,
    max_tokens: int = 4096,
) -> str:
    """Call OpenRouter-compatible API with images and text prompt."""
    content = [{"type": "text", "text": prompt}]
    for frame in frames:
        b64 = encode_image(frame)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
                temperature=0,
            )
            # Defensive: some models return empty choices or None content
            if not response or not response.choices:
                raise ValueError("API returned empty choices")
            text = response.choices[0].message.content
            if text is None:
                raise ValueError("API returned None content")
            return text.strip()
        except Exception as e:
            wait = min(2 ** attempt * 2, 60)
            logger.warning(
                f"API call failed (attempt {attempt + 1}/{max_retries}): {e}. "
                f"Retrying in {wait}s..."
            )
            time.sleep(wait)
    return "ERROR"


# ===================================================================
# F. Single entry processing
# ===================================================================

def process_single_entry(
    entry: dict, client: OpenAI, args: argparse.Namespace
) -> dict:
    """Process one benchmark entry: load frames, render annotations, call API.

    Annotations use pixel coordinates from the original image, so we must
    draw them on full-resolution frames first, then resize for the API
    (matching the preview_script/serve.py rendering logic).
    """
    frame_paths = entry["videos"][0]

    # 1. Determine sampling indices
    n = len(frame_paths)
    if n == 0:
        logger.warning(f"No frames for entry {entry['id']}, skipping.")
        return _make_result(entry, "ERROR: no frames", 0)
    if n <= args.num_frames:
        sampled_indices = list(range(n))
    else:
        sampled_indices = np.linspace(0, n - 1, args.num_frames, dtype=int).tolist()

    # 2. Load frames at full resolution (pixel-coord annotations need original size)
    frames = []
    valid_indices = []
    for i in sampled_indices:
        try:
            frames.append(Image.open(frame_paths[i]).convert("RGB"))
            valid_indices.append(i)
        except Exception as e:
            logger.warning(f"Failed to load frame {frame_paths[i]}: {e}")
    if not frames:
        logger.warning(f"No frames loaded for entry {entry['id']}, skipping.")
        return _make_result(entry, "ERROR: no frames", 0)

    # 3. Render annotations on full-res frames (coords match original pixels)
    frames = render_annotations(entry, frames, valid_indices, frame_paths)

    # 4. Resize after annotation rendering for API submission
    frames = [resize_image(f, args.max_image_size) for f in frames]

    prompt = build_prompt(entry)
    prediction = call_api(client, args.model, frames, prompt, args.max_retries, args.max_tokens)

    return _make_result(entry, prediction, len(frames))


def _make_result(entry: dict, prediction: str, num_frames: int) -> dict:
    return {
        "id": entry["id"],
        "input_type": entry.get("input_type", ""),
        "category": entry["metadata"]["category"],
        "scale": entry["metadata"]["scale"],
        "original_type": entry["metadata"]["original_type"],
        "question": entry["question"],
        "ground_truth": entry["answer"],
        "predicted_answer": prediction,
        "answer_type": classify_answer_type(entry),
        "num_frames_sent": num_frames,
    }


# ===================================================================
# G. Resume support
# ===================================================================

def load_completed_ids(jsonl_path: str) -> set:
    """Load IDs of already-completed entries from a JSONL file."""
    completed = set()
    if not os.path.exists(jsonl_path):
        return completed
    with open(jsonl_path, "r") as f:
        for line in f:
            try:
                doc = json.loads(line.strip())
                completed.add(doc["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return completed


# ===================================================================
# H. Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SpaceVista-Bench using VLM API (OpenRouter)."
    )
    parser.add_argument(
        "--dataset_path", type=str,
        default="bench_all_clean_corrected.json",
        help="Path to bench_all_clean_corrected.json",
    )
    parser.add_argument(
        "--model", type=str, default="qwen/qwen2.5-vl-72b-instruct",
        help="Model name (e.g. qwen/qwen2.5-vl-72b-instruct, google/gemini-2.5-flash)",
    )
    parser.add_argument(
        "--api_key", type=str,
        default=os.environ.get("API_KEY"),
        help="API key (defaults to $API_KEY env var)",
    )
    parser.add_argument(
        "--base_url", type=str, default="https://openrouter.ai/api/v1",
        help="API base URL (e.g. https://openrouter.ai/api/v1, https://api.openai.com/v1)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./eval_results_api",
        help="Output directory",
    )
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--max_image_size", type=int, default=448)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--max_tokens", type=int, default=8196,
                        help="Max output tokens per API call (default: 4096)")
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to existing JSONL to resume from",
    )
    parser.add_argument(
        "--category", type=str, default=None,
        choices=["Indoor", "Outdoor", "TinyTabletop", "Tabletop"],
        help="Evaluate only one scale category",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_size", type=int, default=20)
    parser.add_argument(
        "--metrics_only", type=str, default=None,
        help="Path to existing JSONL — just recompute metrics, no API calls",
    )
    args = parser.parse_args()

    # --- Metrics-only mode ---
    if args.metrics_only:
        print(f"Computing metrics from: {args.metrics_only}")
        results = compute_metrics(args.metrics_only)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    # --- Load dataset ---
    print(f"Loading dataset from {args.dataset_path} ...")
    with open(args.dataset_path, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries")

    if args.category:
        data = [d for d in data if d["metadata"]["category"] == args.category]
        print(f"Filtered to category={args.category}: {len(data)} entries")

    if args.debug:
        data = data[: args.debug_size]
        print(f"Debug mode: using first {len(data)} entries")

    # --- Setup output ---
    model_slug = args.model.replace("/", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, model_slug, ts)
    os.makedirs(out_dir, exist_ok=True)

    output_file = os.path.join(out_dir, f"results_{model_slug}.jsonl")

    # --- Resume ---
    completed_ids = set()
    if args.resume:
        output_file = args.resume
        completed_ids = load_completed_ids(output_file)
        print(f"Resuming: {len(completed_ids)} entries already completed")

    pending = [d for d in data if d["id"] not in completed_ids]
    print(f"Entries to process: {len(pending)}")

    if not pending:
        print("Nothing to process.")
        print(f"Computing metrics from: {output_file}")
        results = compute_metrics(output_file)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    # --- Save run config ---
    config_path = os.path.join(os.path.dirname(output_file), "run_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    # --- Initialize API client ---
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    # --- Process ---
    print(f"Output file: {output_file}")
    print(f"Model: {args.model}")
    print(f"Frames per entry: {args.num_frames}, max image size: {args.max_image_size}")

    if args.max_workers <= 1:
        # Sequential processing
        with open(output_file, "a") as out_f:
            for entry in tqdm(pending, desc="Evaluating"):
                result = process_single_entry(entry, client, args)
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
    else:
        # Parallel processing
        lock = threading.Lock()
        with open(output_file, "a") as out_f:
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = {
                    executor.submit(process_single_entry, e, client, args): e
                    for e in pending
                }
                for future in tqdm(
                    as_completed(futures), total=len(futures), desc="Evaluating"
                ):
                    try:
                        result = future.result()
                    except Exception as e:
                        entry = futures[future]
                        logger.error(f"Entry {entry['id']} failed: {e}")
                        result = _make_result(entry, f"ERROR: {e}", 0)
                    with lock:
                        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        out_f.flush()

    # --- Compute metrics ---
    print(f"\nEvaluation complete. Computing metrics from: {output_file}")
    results = compute_metrics(output_file)
    print("\n" + json.dumps(results, indent=2, ensure_ascii=False))

    # Save metrics
    metrics_path = output_file.replace(".jsonl", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
