import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import shutil
import numpy as np
from PIL import Image, ImageDraw
import imageio.v3 as iio
import random

# ============= Utilities for mask coloring =============
def overlay_mask_rgba(pil_img: Image.Image, mask: np.ndarray, color=(255, 0, 0), alpha=0.35) -> Image.Image:
    """
    Overlay a boolean or 0/1 mask onto a PIL image with given color and alpha.
    Auto-resizes the mask to image size with nearest neighbor.
    """
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
        mask_pil = Image.fromarray((mask.astype(np.uint8) * 255)).resize((base.width, base.height), resample=Image.NEAREST)
    else:
        mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)

    out = Image.composite(color_img, base, mask_pil)
    return out.convert("RGB")


def load_mask_array(mask_path: str) -> Optional[np.ndarray]:
    """
    Load a segmentation mask numpy file. It can be:
    - boolean HxW
    - int HxW with object ids
    - one-hot HxWxK or KxHxW
    """
    if not os.path.isfile(mask_path):
        print(f"[WARN] Mask path not found: {mask_path}")
        return None
    try:
        arr = np.load(mask_path, allow_pickle=True)
        return arr
    except Exception as e:
        print(f"[WARN] Failed to load mask {mask_path}: {e}")
        return None


def select_mask_from_array(mask_arr: np.ndarray, object_ids: Union[List[bool], List[int], List[str], None]) -> Optional[np.ndarray]:
    """
    Turn different mask encodings into a single boolean mask HxW.
    Heuristics:
    - If mask_arr is 2D boolean: return as is (or, if object_ids True-any -> return; else zeros)
    - If mask_arr is 2D int: ids represent classes/instances. object_ids must be list of ints -> mask = OR over ids.
      If no ids, foreground = arr != 0
    - If mask_arr is 3D one-hot (H,W,K) or (K,H,W):
        * If object_ids is bool list length K: OR selected channels
        * If object_ids is int list: OR channels by indices
        * If object_ids is None: OR all channels
    """
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
                except:
                    pass
            if not ids:
                return arr != 0
            mask = np.zeros_like(arr, dtype=bool)
            for k in ids:
                mask |= (arr == k)
            return mask

    if arr.ndim == 3:
        if arr.shape[0] not in (arr.shape[1],) and arr.shape[0] < 16 and arr.shape[0] != arr.shape[1]:
            arr = np.moveaxis(arr, 0, 2)  # (H,W,K)
        if arr.dtype != bool:
            arr = arr != 0
        H, W, K = arr.shape
        if object_ids is None:
            return arr.any(axis=2)
        if isinstance(object_ids, list) and len(object_ids) == K and all(isinstance(x, (bool, np.bool_)) for x in object_ids):
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
                    except:
                        pass
            sel = tmp
        if not sel:
            return arr.any(axis=2)
        sel = [i for i in sel if 0 <= i < K]
        if not sel:
            return arr.any(axis=2)
        return arr[:, :, sel].any(axis=2)

    print(f"[WARN] Unsupported mask array shape: {arr.shape}")
    return None


# ============= Visualizer (drawing) =============
class Visualizer:
    def __init__(self):
        pass

    def _safe_to_pil(self, img) -> Image.Image:
        if isinstance(img, Image.Image):
            return img
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                return Image.fromarray(img, mode="L")
            if img.ndim == 3:
                if img.shape[2] == 3:
                    if img.flags["C_CONTIGUOUS"]:
                        return Image.fromarray(img[:, :, ::-1])  # BGR->RGB
                    else:
                        return Image.fromarray(np.ascontiguousarray(img)[:, :, ::-1])
                if img.shape[2] == 4:
                    rgba = img[:, :, [2, 1, 0, 3]]  # BGRA->RGBA
                    return Image.fromarray(rgba, mode="RGBA")
        try:
            return Image.fromarray(np.array(img))
        except Exception:
            raise TypeError(f"Unsupported image type for conversion to PIL: {type(img)}")

    def _pil_to_numpy_bgr(self, img: Image.Image) -> np.ndarray:
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        arr = np.array(img)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return arr[:, :, ::-1].copy()  # RGB->BGR

    def _scale_point(self, point, width, height):
        """
        Prefer (x, y). If that is out-of-bounds but (y, x) is valid,
        auto-swap once and WARN.
        """
        if isinstance(point, (list, tuple)) and len(point) == 1 and isinstance(point[0], (list, tuple, np.ndarray)):
            point = point[0]
        if not (isinstance(point, (list, tuple, np.ndarray)) and len(point) == 2):
            raise ValueError(f"Unsupported point format: {point}")

        y, x = float(point[0]), float(point[1])

        if 0 <= x < width and 0 <= y < height:
            return np.array([x, y], dtype=float)

        sx, sy = y, x
        if 0 <= sx < width and 0 <= sy < height:
            print(f"[WARN] Input point seems to be (y,x); auto-swapped to (x,y) for {point}")
            return np.array([sx, sy], dtype=float)

        return np.array([x, y], dtype=float)

    def _scale_bbox(self, bbox, width, height):
        """
        Normalize bbox to [x1, y1, x2, y2].
        Accept:
          - dict with x1,y1,x2,y2
          - dict with x,y,w,h
          - list/tuple/ndarray of 4 numbers treated as xyxy by default; if reversed, we sort.
          - single-element containers like [dict] are flattened.
        """
        if isinstance(bbox, (list, tuple)) and len(bbox) == 1 and isinstance(bbox[0], dict):
            b = bbox[0]
        elif isinstance(bbox, dict):
            b = bbox
        elif isinstance(bbox, (list, tuple, np.ndarray)):
            b = np.array(bbox, dtype=float).flatten()
        else:
            raise ValueError(f"Unsupported bbox format: {bbox}")

        if isinstance(b, dict):
            if all(k in b for k in ("x1", "y1", "x2", "y2")):
                x1 = float(b["x1"]); y1 = float(b["y1"]); x2 = float(b["x2"]); y2 = float(b["y2"])
            elif all(k in b for k in ("x", "y", "w", "h")):
                x = float(b["x"]); y = float(b["y"]); w = float(b["w"]); h = float(b["h"])
                x1, y1, x2, y2 = x, y, x + w, y + h
            else:
                vals = [b.get(k) for k in ("x1","y1","x2","y2")]
                if all(v is not None for v in vals):
                    x1, y1, x2, y2 = map(float, vals)
                else:
                    raise ValueError(f"bbox dict must contain x1,y1,x2,y2 or x,y,w,h: {b}")
        else:
            arr = np.array(b, dtype=float).flatten()
            if arr.size != 4:
                raise ValueError(f"bbox must have 4 elements, got {bbox}")
            x1, y1, x2, y2 = arr.tolist()
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1

        x1 = max(0.0, min(x1, width - 1))
        y1 = max(0.0, min(y1, height - 1))
        x2 = max(0.0, min(x2, width - 1))
        y2 = max(0.0, min(y2, height - 1))
        return np.array([x1, y1, x2, y2], dtype=float)

    def _draw_annotations_on_pil(self, img: Image.Image, extra_info: Dict[str, Any], current_img_idx: int) -> Image.Image:
        draw = ImageDraw.Draw(img)
        W, H = img.width, img.height
        colors_rgb = {
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 128, 255),
            "yellow": (255, 215, 0),
        }
        order = ["red", "blue", "green", "yellow"]

        def _normalize_indices(val):
            if val is None:
                return None
            if isinstance(val, list) and len(val) == 1 and isinstance(val[0], list):
                return val[0]
            return val

        point_img_idx = _normalize_indices(extra_info.get("point_img_idx"))
        bbox_img_idx = _normalize_indices(extra_info.get("bbox_img_idx"))
        mask_img_idx = _normalize_indices(extra_info.get("mask_img_idx"))

        stroke = max(2, int(0.004 * min(W, H)))

        # ---------- helpers to collect shapes ----------
        def _get_points_list(val):
            if val is None:
                return []
            if isinstance(val, (list, tuple)):
                # 允许 [pt,...] 或 [[pt], ...]
                out = []
                for el in val:
                    if isinstance(el, (list, tuple, np.ndarray)):
                        out.append(el)
                    elif isinstance(el, (list, tuple)) and len(el) == 1 and isinstance(el[0], (list, tuple, np.ndarray)):
                        out.append(el[0])
                if not out and len(val) == 2 and all(isinstance(x, (int, float)) for x in val):
                    return [val]
                return out
            if isinstance(val, (list, tuple, np.ndarray)):
                return [val]
            return []

        def _get_boxes_list(val):
            if val is None:
                return []
            if isinstance(val, dict):
                return [val]
            if isinstance(val, (list, tuple)):
                out = []
                for el in val:
                    if isinstance(el, dict):
                        out.append(el)
                    elif isinstance(el, (list, tuple)) and len(el) == 1 and isinstance(el[0], dict):
                        out.append(el[0])
                return out
            return []

        def _iter_mask_infos_for_color(color_key: str):
            info = extra_info.get(f"{color_key}_mask_info", None)
            outs = []
            if info is None:
                return outs
            if isinstance(info, dict):
                arr = info.get("mask_array")
                if isinstance(arr, np.ndarray):
                    outs.append(arr)
            elif isinstance(info, list):
                for el in info:
                    if isinstance(el, dict) and isinstance(el.get("mask_array"), np.ndarray):
                        outs.append(el["mask_array"])
            return outs

        # ---------- collect per-color data ----------
        red_points = _get_points_list(extra_info.get("red_point"))
        blue_points = _get_points_list(extra_info.get("blue_point"))
        green_points = _get_points_list(extra_info.get("green_point"))
        yellow_points = _get_points_list(extra_info.get("yellow_point"))
        per_color_points = [red_points, blue_points, green_points, yellow_points]

        red_boxes = _get_boxes_list(extra_info.get("red_bbox"))
        blue_boxes = _get_boxes_list(extra_info.get("blue_bbox"))
        green_boxes = _get_boxes_list(extra_info.get("green_bbox"))
        yellow_boxes = _get_boxes_list(extra_info.get("yellow_bbox"))

        # print(extra_info)

        red_masks = _iter_mask_infos_for_color("red")
        blue_masks = _iter_mask_infos_for_color("blue")
        green_masks = _iter_mask_infos_for_color("green")
        yellow_masks = _iter_mask_infos_for_color("yellow")

        # ---------- unified ref/cand index decision (strictly aligned) ----------
        idx_ref_from_global = None
        idx_cand_from_global = None

        # 先看 bbox 再看 mask 再看 point；后者如满足条件则覆盖前者（与你给出的代码片段一致）
        if bbox_img_idx is not None and isinstance(bbox_img_idx, list) and len(bbox_img_idx) >= 4:
            idx_ref_from_global = bbox_img_idx[0]
            for k in [1, 2, 3, 4]:
                if k < len(bbox_img_idx) and bbox_img_idx[k] is not None:
                    idx_cand_from_global = bbox_img_idx[k]
                    break
        if mask_img_idx is not None and isinstance(mask_img_idx, list) and len(mask_img_idx) >= 4:
            idx_ref_from_global = mask_img_idx[0]
            for k in [1, 2, 3, 4]:
                if k < len(mask_img_idx) and mask_img_idx[k] is not None:
                    idx_cand_from_global = mask_img_idx[k]
                    break
        if point_img_idx is not None and isinstance(point_img_idx, list) and len(point_img_idx) >= 4:
            idx_ref_from_global = point_img_idx[0]
            for k in [1, 2, 3, 4]:
                if k < len(point_img_idx) and point_img_idx[k] is not None:
                    idx_cand_from_global = point_img_idx[k]
                    break

        on_ref = (idx_ref_from_global is not None and current_img_idx == idx_ref_from_global)
        on_cand = (idx_cand_from_global is not None and current_img_idx == idx_cand_from_global)

        # ---------- draw helpers ----------
        def _draw_point(pt, color):
            try:
                xy = self._scale_point(pt, W, H)
                x, y = float(xy[0]), float(xy[1])
                if not (0 <= x < W and 0 <= y < H):
                    return
                r = max(8, int(0.014 * min(W, H)))
                draw.ellipse((x - r, y - r, x + r, y + r), fill=colors_rgb[color], outline=colors_rgb[color], width=2)
            except Exception as e:
                print(f"[WARN] bad point {pt}: {e}")

        def _overlay_mask(mk, color):
            try:
                img_with = overlay_mask_rgba(
                    img,
                    mk,
                    color={"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 128, 255), "yellow": (255, 215, 0)}[color],
                    alpha=0.35,
                )
                img.paste(img_with)
            except Exception as e:
                print(f"[WARN] bad mask for color {color}: {e}")

        # ---------- POINTS (aligned with bbox logic) ----------
        if on_ref:
            # Reference frame: show first red and first blue point
            if point_img_idx and point_img_idx[0]==0:
                for pt in red_points[:1]:
                    _draw_point(pt, "red")
            if point_img_idx and point_img_idx[1]==0:
                for pt in blue_points[:1]:
                    _draw_point(pt, "blue")
        elif on_cand:
            # 候选帧：red(除第一个) + green + blue + yellow，循环重着色
            cand_pool = []
            if len(red_points) > 1:
                cand_pool.extend(red_points[1:])
            cand_pool.extend(green_points)
            cand_pool.extend(blue_points)
            cand_pool.extend(yellow_points)
            recolor = ["red", "blue", "green", "yellow"]
            for j, pt in enumerate(cand_pool):
                _draw_point(pt, recolor[j % 4])
        else:
            # 非配对/legacy：遵循 point_img_idx gating
            for i, color in enumerate(order):
                allowed = True
                if point_img_idx is not None:
                    if isinstance(point_img_idx, list):
                        if i >= len(point_img_idx) or point_img_idx[i] is None or point_img_idx[i] != current_img_idx:
                            allowed = False
                    else:
                        allowed = (point_img_idx == current_img_idx)
                if not allowed:
                    continue
                for pt in per_color_points[i]:
                    _draw_point(pt, color)

        # ---------- BBOXES (retain your existing aligned logic) ----------
        if on_ref:
            if bbox_img_idx and bbox_img_idx[0]==0:
                for bb in red_boxes[:1]:
                    try:
                        x1, y1, x2, y2 = self._scale_bbox(bb, W, H)
                        if x1 < W and y1 < H and x2 > 0 and y2 > 0:
                            draw.rectangle((x1, y1, x2, y2), outline=colors_rgb["red"], width=stroke)
                    except Exception as e:
                        print(f"[WARN] bad ref bbox: {bb} ({e})")
            if bbox_img_idx and bbox_img_idx[1]==0:
                for bb in blue_boxes[:1]:
                    try:
                        x1, y1, x2, y2 = self._scale_bbox(bb, W, H)
                        if x1 < W and y1 < H and x2 > 0 and y2 > 0:
                            draw.rectangle((x1, y1, x2, y2), outline=colors_rgb["blue"], width=stroke)
                    except Exception as e:
                        print(f"[WARN] bad ref bbox: {bb} ({e})")
        elif on_cand:
            cand_pool = []
            if len(red_boxes) > 1:
                cand_pool.extend(red_boxes[1:])
            if green_boxes:
                cand_pool.extend(green_boxes)
            if blue_boxes:
                cand_pool.extend(blue_boxes)
            if yellow_boxes:
                cand_pool.extend(yellow_boxes)
            recolor = ["red", "blue", "green", "yellow"]
            for j, bb in enumerate(cand_pool):
                try:
                    x1, y1, x2, y2 = self._scale_bbox(bb, W, H)
                    if x1 < W and y1 < H and x2 > 0 and y2 > 0:
                        draw.rectangle((x1, y1, x2, y2), outline=colors_rgb[recolor[j % 4]], width=stroke)
                except Exception as e:
                    print(f"[WARN] bad cand bbox: {bb} ({e})")
        else:
            per_color_boxes = [red_boxes, blue_boxes, green_boxes, yellow_boxes]
            for i, color in enumerate(order):
                if bbox_img_idx is not None:
                    if isinstance(bbox_img_idx, list):
                        if i >= len(bbox_img_idx) or bbox_img_idx[i] is None or bbox_img_idx[i] != current_img_idx:
                            continue
                    else:
                        if bbox_img_idx != current_img_idx:
                            continue
                boxes = per_color_boxes[i]
                for bb in boxes:
                    try:
                        x1, y1, x2, y2 = self._scale_bbox(bb, W, H)
                        if x1 < W and y1 < H and x2 > 0 and y2 > 0:
                            draw.rectangle((x1, y1, x2, y2), outline=colors_rgb[color], width=stroke)
                    except Exception as e:
                        print(f"[WARN] bad bbox on frame {current_img_idx}: {bb} ({e})")

        # ---------- MASKS (aligned with bbox logic) ----------
        if on_ref:
            if red_masks and mask_img_idx[0]==0:
                _overlay_mask(red_masks[0], "red")
            if blue_masks and mask_img_idx[1]==0:
                _overlay_mask(blue_masks[0], "blue")
        elif on_cand:
            cand_pool = []
            if len(red_masks) > 1:
                cand_pool.extend(red_masks[1:])
            cand_pool.extend(green_masks)
            cand_pool.extend(blue_masks)
            cand_pool.extend(yellow_masks)
            palette = ["red", "blue", "green", "yellow"]
            for j, mk in enumerate(cand_pool):
                _overlay_mask(mk, palette[j % 4])
        else:
            per_color_masks = [red_masks, blue_masks, green_masks, yellow_masks]
            for i, color in enumerate(order):
                allowed = True
                if mask_img_idx is not None:
                    if isinstance(mask_img_idx, list):
                        if i < len(mask_img_idx) and mask_img_idx[i] is not None:
                            allowed = (mask_img_idx[i] == current_img_idx)
                        else:
                            # 如果提供的是“集合含义”的列表，允许 current 在该列表中
                            allowed = current_img_idx in [m for m in mask_img_idx if m is not None] if any(m is not None for m in mask_img_idx) else True
                    else:
                        allowed = (mask_img_idx == current_img_idx)
                if not allowed:
                    continue
                for mk in per_color_masks[i]:
                    _overlay_mask(mk, color)

        return img
    def _maybe_annotate_frames(
        self,
        processed_frames: List[Any],
        sampled_indices: List[int],
        extra_info: Optional[Dict[str, Any]],
        debug: bool = False,
        debug_dir: str = "debug_video_anns",
        video_tag: str = "vid",
    ) -> List[Any]:
        if not extra_info:
            return processed_frames

        os.makedirs(debug_dir, exist_ok=True)

        annotated_frames = []
        for j, frame in enumerate(processed_frames):
            pil = self._safe_to_pil(frame)
            current_idx = int(sampled_indices[j]) if sampled_indices is not None and j < len(sampled_indices) else j
            annotated = self._draw_annotations_on_pil(pil.copy(), extra_info, current_img_idx=current_idx)

            if isinstance(frame, Image.Image):
                out = annotated
            else:
                out = self._pil_to_numpy_bgr(annotated)

            annotated_frames.append(out)

            if debug:
                debug_name = f"{video_tag}_frame{j:04d}_orig{current_idx:04d}.jpg"
                try:
                    annotated.save(os.path.join(debug_dir, debug_name))
                except Exception:
                    pass

        return annotated_frames

    def load_frames(self, frame_paths: List[str]) -> List[Image.Image]:
        frames = []
        for p in frame_paths:
            if not os.path.isfile(p):
                print(f"[WARN] Missing frame: {p}")
                continue
            try:
                img = Image.open(p).convert("RGB")
            except Exception as e:
                print(f"[WARN] Failed to load {p}: {e}")
                continue
            frames.append(img)
        return frames

    def build_video(
        self,
        frames: List[Image.Image],
        out_path: str,
        fps: int = 2,
        extra_info: Optional[Dict[str, Any]] = None,
        debug: bool = False,
        tag: str = "vid",
    ):
        if not frames:
            raise ValueError("No frames to write.")
        processed = [self._pil_to_numpy_bgr(self._safe_to_pil(f)) for f in frames]
        sampled_indices = list(range(len(frames)))
        annotated = self._maybe_annotate_frames(processed, sampled_indices, extra_info, debug=debug, video_tag=tag)
        annotated_rgb = [img[:, :, ::-1] for img in annotated]  # BGR -> RGB
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        iio.imwrite(out_path, annotated_rgb, fps=fps, codec="h264", quality=8)
        print(f"[OK] Wrote video: {out_path} ({len(annotated_rgb)} frames @ {fps} fps)")


# ============= Helpers to adapt JSON into Visualizer.extra_info schema =============
def extract_all_frame_paths(item: Dict[str, Any]) -> List[str]:
    videos = item.get("videos", [])
    all_paths: List[str] = []
    for group in videos:
        if isinstance(group, list):
            all_paths.extend(group)
    return [p for p in all_paths if isinstance(p, str) and len(p) > 0]


def index_of_image_in_frames(image_path: str, frames: List[str]) -> Optional[int]:
    """
    Robustly find index of image_path within frames list:
    - exact absolute path match
    - try /images/ <-> /images_8/ swap
    - basename unique match
    """
    if not image_path:
        return None
    apath = os.path.abspath(image_path)

    for i, f in enumerate(frames):
        if os.path.abspath(f) == apath:
            return i

    cand = []
    if "/images_8/" in apath:
        cand.append(apath.replace("/images_8/", "/images/"))
    if "/images/" in apath:
        cand.append(apath.replace("/images/", "/images_8/"))

    for c in cand:
        ac = os.path.abspath(c)
        for i, f in enumerate(frames):
            if os.path.abspath(f) == ac:
                return i

    base = os.path.basename(apath)
    matches = [i for i, f in enumerate(frames) if os.path.basename(f) == base]
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        for i in matches:
            af = os.path.abspath(frames[i])
            if "/images_8/" in af or "/images/" in af:
                return i
        return matches[0]

    print(f"[WARN] Could not locate reference image in frames: {image_path}")
    return None


def _json_maybe_parse(value):
    """
    If value is a JSON-encoded string, parse it. Otherwise, return as-is.
    Accepts None, list, dict, bool, numbers.
    """
    if value is None:
        return None
    if isinstance(value, (list, dict, bool, int, float)):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s == "":
            return None
        try:
            return json.loads(s)
        except Exception:
            try:
                s2 = s.replace("True", "true").replace("False", "false")
                return json.loads(s2)
            except Exception:
                return value
    return value


def build_extra_info_for_item(item: Dict[str, Any], frame_paths: List[str]) -> Dict[str, Any]:
    """
    将 bbox/point/mask 的 extra_info 构造成 Visualizer 需要的键，并在检测到“配对模式”时
    将 point 与 mask 的 ref/cand 索引与候选分配与 bbox 对齐。对单侧输入的 point/mask，
    仍生成严格对齐的 idx 列表，如 [idx_ref, None, None, None, None]。
    """
    ein = item.get("extra_info", {}) or {}
    out: Dict[str, Any] = {}
    colors = ["red", "blue", "green", "yellow"]

    def _as_list(x):
        if x is None:
            return []
        if isinstance(x, list):
            return x
        return [x]

    def _normalize_points_list(val):
        if val is None:
            return []
        if isinstance(val, (list, tuple)):
            if len(val) == 2 and all(isinstance(a, (int, float)) for a in val):
                return [val]
            outp = []
            for el in val:
                if isinstance(el, (list, tuple)) and len(el) == 2:
                    outp.append(el)
                elif isinstance(el, (list, tuple)) and len(el) == 1 and isinstance(el[0], (list, tuple)) and len(el[0]) == 2:
                    outp.append(el[0])
            return outp
        return [val] if isinstance(val, (tuple, list)) else []

    def _add_points_into_out(points_list, color_key):
        if not points_list:
            return
        if out.get(f"{color_key}_point") is None:
            out[f"{color_key}_point"] = list(points_list)
        else:
            if isinstance(out[f"{color_key}_point"], list):
                out[f"{color_key}_point"].extend(list(points_list))
            else:
                out[f"{color_key}_point"] = [out[f"{color_key}_point"], *list(points_list)]

    def _load_and_select_masks(mask_paths, ids_list):
        res = []
        for j, mp in enumerate(mask_paths):
            arr = load_mask_array(mp) if mp else None
            sel = select_mask_from_array(arr, ids_list[j]) if arr is not None else None
            if sel is not None:
                res.append(sel)
        return res

    # 先尝试从 bbox 中建立 ref/cand
    bbox_img_path = ein.get("bbox_image_path")
    bbox_bboxes = _json_maybe_parse(ein.get("bbox_bboxes"))
    ib_old = ein.get("input_bbox")

    idx_ref = None
    idx_cand = None
    handled_pair_bbox = False

    if isinstance(ib_old, dict) and ("image_path_1" in ib_old) and ("bbox_1" in ib_old) and ("image_path_2" in ib_old) and ("bboxes_2" in ib_old):
        ref_img = ib_old.get("image_path_1")
        ref_bbox = ib_old.get("bbox_1")
        cand_img = ib_old.get("image_path_2")
        cand_bboxes = ib_old.get("bboxes_2") or []

        idx_ref = index_of_image_in_frames(ref_img, frame_paths)
        idx_cand = index_of_image_in_frames(cand_img, frame_paths)

        if idx_ref is None:
            idx_ref = 0
            print(f"[WARN] bbox ref frame not found; fallback idx=0 ({os.path.basename(ref_img) if ref_img else 'N/A'})")
        if idx_cand is None:
            idx_cand = len(frame_paths) - 1
            print(f"[WARN] bbox candidate frame not found; fallback to last idx={idx_cand} ({os.path.basename(cand_img) if cand_img else 'N/A'})")

        if isinstance(ref_bbox, dict) and all(k in ref_bbox for k in ("x1", "y1", "x2", "y2")):
            out["red_bbox"] = [ref_bbox]
        else:
            print("[WARN] bbox_1 missing or invalid dict with x1,y1,x2,y2")

        for j, b in enumerate(cand_bboxes):
            if isinstance(b, dict) and all(k in b for k in ("x1", "y1", "x2", "y2")):
                color = colors[j % 4]
                out.setdefault(f"{color}_bbox", []).append(b)
            else:
                print(f"[WARN] candidate bbox idx {j} invalid")

        out["bbox_img_idx"] = [idx_ref, idx_cand, idx_cand, idx_cand, idx_cand]
        handled_pair_bbox = True

    if not handled_pair_bbox:
        if (bbox_img_path and bbox_bboxes) or (isinstance(ib_old, dict) and ib_old.get("image_path") and ib_old.get("bboxes")):
            if not (bbox_img_path and bbox_bboxes):
                bbox_img_path = ib_old.get("image_path")
                bbox_bboxes = ib_old.get("bboxes")
            idx = index_of_image_in_frames(bbox_img_path, frame_paths)
            if idx is None:
                idx = 0
                print(f"[WARN] bbox ref frame not found; fallback idx=0 ({os.path.basename(bbox_img_path) if bbox_img_path else 'N/A'})")
            if isinstance(bbox_bboxes, list) and len(bbox_bboxes) > 0:
                for j, b in enumerate(bbox_bboxes):
                    if isinstance(b, dict) and all(k in b for k in ("x1", "y1", "x2", "y2")):
                        color = colors[j % 4]
                        out.setdefault(f"{color}_bbox", []).append(b)
                    else:
                        print(f"[WARN] bbox idx {j} invalid")
                out["bbox_img_idx"] = [idx if out.get(f"{c}_bbox") else None for c in colors] + [None]

    # POINT（支持单侧）
    ip_old = ein.get("input_point")
    point_img_path = ein.get("point_image_path")
    point_points = _json_maybe_parse(ein.get("point_points"))

    if isinstance(ip_old, dict) and (ip_old.get("image_path") and ip_old.get("points")):
        p_img = ip_old.get("image_path")
        pts = _normalize_points_list(ip_old.get("points"))
        idxp = index_of_image_in_frames(p_img, frame_paths)
        if idxp is None:
            idxp = 0
            print(f"[WARN] point ref frame not found; fallback idx=0 ({os.path.basename(p_img) if p_img else 'N/A'})")
        for j, pt in enumerate(pts):
            color = colors[j % 4]
            _add_points_into_out([pt], color)
        # 单侧：与对齐策略一致 -> [idx_ref, None, None, None, None]
        out["point_img_idx"] = [idxp if out.get(f"{c}_point") else None for c in colors] + [None]

    elif (point_img_path and point_points):
        idxp = index_of_image_in_frames(point_img_path, frame_paths)
        if idxp is None:
            idxp = 0
            print(f"[WARN] point ref frame not found; fallback idx=0 ({os.path.basename(point_img_path) if point_img_path else 'N/A'})")
        pts = _normalize_points_list(point_points)
        for j, pt in enumerate(pts):
            color = colors[j % 4]
            _add_points_into_out([pt], color)
        out["point_img_idx"] = [idxp if out.get(f"{c}_point") else None for c in colors] + [None]

    # MASK（支持单侧 input_mask 格式：一个 mask_path + 多个 object_ids）
    im_old = ein.get("input_mask")
    mask_img_path = ein.get("mask_image_path")
    mask_mask_path = ein.get("mask_mask_path")
    mask_object_ids = _json_maybe_parse(ein.get("mask_object_ids"))

    def _assign_masks_to_colors(masks_list):
        # 将多个mask轮转分配到 red/blue/green/yellow
        for j, m in enumerate(masks_list):
            color = colors[j % 4]
            out.setdefault(f"{color}_mask_info", []).append({"mask_array": m})

    if isinstance(im_old, dict) and im_old.get("image_path") and im_old.get("mask_path"):
        m_img = im_old.get("image_path")
        m_path = im_old.get("mask_path")
        obj_ids = im_old.get("object_ids", None)
        idxm = index_of_image_in_frames(m_img, frame_paths)
        if idxm is None:
            idxm = 0
            print(f"[WARN] mask ref frame not found; fallback idx=0 ({os.path.basename(m_img) if m_img else 'N/A'})")

        # 单 .npy 下的多个 object_id
        mask_paths = [m_path]
        ids_list = _as_list(obj_ids) if isinstance(obj_ids, list) else [obj_ids]
        # 对于单路径多 id，我们需要逐个选择并追加
        masks_ref = []
        if mask_paths and len(ids_list) >= 1:
            arr = load_mask_array(mask_paths[0])
            if arr is not None:
                # 将每个 id 选择出一个布尔掩码
                for oid in ids_list:
                    sel = select_mask_from_array(arr, oid)
                    if sel is not None:
                        masks_ref.append(sel)

        if masks_ref:
            _assign_masks_to_colors(masks_ref)
            # 单侧：idx 对齐 -> [idx_ref, None, None, None, None]
            out["mask_img_idx"] = [idxm if out.get(f"{c}_mask_info") else None for c in colors]

    elif (mask_img_path and mask_mask_path):
        idxm = index_of_image_in_frames(mask_img_path, frame_paths)
        if idxm is None:
            idxm = 0
            print(f"[WARN] mask ref frame not found; fallback idx=0 ({os.path.basename(mask_img_path) if mask_img_path else 'N/A'})")
        mask_paths = _as_list(mask_mask_path)
        obj_ids_list = _as_list(mask_object_ids) if isinstance(mask_object_ids, list) and len(mask_paths) == len(_as_list(mask_object_ids)) else [mask_object_ids] * len(mask_paths)

        masks_selected = []
        for j, mp in enumerate(mask_paths):
            arr = load_mask_array(mp) if mp else None
            sel = select_mask_from_array(arr, obj_ids_list[j]) if arr is not None else None
            if sel is not None:
                masks_selected.append(sel)

        if masks_selected:
            _assign_masks_to_colors(masks_selected)
            out["mask_img_idx"] = [idxm if out.get(f"{c}_mask_info") else None for c in colors]

    # 如果存在 bbox 的成对 idx_ref/idx_cand，统一覆盖对齐（point/mask 也对齐）
    if idx_ref is not None and idx_cand is not None:
        if any(out.get(f"{c}_point") for c in colors):
            out["point_img_idx"] = [idx_ref, idx_cand, idx_cand, idx_cand, idx_cand]
        if any(out.get(f"{c}_mask_info") for c in colors):
            out["mask_img_idx"] = [idx_ref, idx_cand, idx_cand, idx_cand, idx_cand]

    return out if out else {}

def main():
    parser = argparse.ArgumentParser(description="Annotate frames from JSON and export MP4 videos at 0.5s per frame.")
    parser.add_argument("--json", required=True, help="Path to input JSON.")
    parser.add_argument("--outdir", default="output_videos_vllm", help="Output directory for MP4 videos.")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second (2 => 0.5s per frame).")
    parser.add_argument("--debug", action="store_true", help="Save per-frame annotated JPEGs.")
    args = parser.parse_args()

    if os.path.isdir(args.outdir):
        shutil.rmtree(args.outdir)

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    args.seed = 47
    random.seed(args.seed)
    num_samples=10
    total = len(data)
    n = min(num_samples, total)
    if n < total:
        indices = random.sample(range(total), n)
        data = [data[i] for i in indices]
        print(f"[INFO] Randomly selected {n} of {total} items with seed={args.seed}. Indices: {sorted(indices)}")
    else:
        data = data
        print(f"[INFO] JSON has only {total} items; processing all.")


    if not isinstance(data, list):
        raise ValueError("Top-level JSON must be a list of items.")

    vis = Visualizer()
    os.makedirs(args.outdir, exist_ok=True)

    for idx, item in enumerate(data):
        problem_id = item.get("problem_id", f"item{idx:04d}")
        frame_paths = extract_all_frame_paths(item)
        if not frame_paths:
            print(f"[INFO] No frames for {problem_id}, skipping.")
            continue
        # 如需限定输入类型，可按需开启；你示例中是 Image+bbox+text

        # if item.get("extra_info", {}).get("input_type") not in {"Image+bbox+text"} or  item.get("extra_info", {}).get("task_type") not in {"COMPARISON_WIDTH"}:
        # if item.get("extra_info", {}).get("input_type") not in {"Image+point+text"}:
        #     continue

        frames = vis.load_frames(frame_paths)
        if not frames:
            print(f"[INFO] Failed to load any frames for {problem_id}, skipping.")
            continue

        drawing_info = build_extra_info_for_item(item, frame_paths)

        if drawing_info:
            pidx = drawing_info.get("point_img_idx")
            bidx = drawing_info.get("bbox_img_idx")
            midx = drawing_info.get("mask_img_idx")
            print(f"[DBG] {problem_id} point_idx={pidx} bbox_idx={bidx} mask_idx={midx}")

        out_path = os.path.join(args.outdir, f"output_{problem_id}.mp4")
        vis.build_video(frames, out_path, fps=args.fps, extra_info=drawing_info, debug=args.debug, tag=str(problem_id))


if __name__ == "__main__":
    main()