import json
import os
from argparse import Namespace
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


def infer_dataset_type(source_path: str) -> str:
    if os.path.exists(os.path.join(source_path, "transforms_train.json")):
        return "blender"
    if os.path.exists(os.path.join(source_path, "sparse")):
        return "colmap"
    return "unknown"


def load_cfg_args(model_path: str) -> Dict:
    cfg_path = os.path.join(model_path, "cfg_args")
    if not os.path.exists(cfg_path):
        return {}

    with open(cfg_path, encoding="utf-8") as cfg_file:
        cfg_text = cfg_file.read().strip()

    if not cfg_text:
        return {}

    try:
        namespace = eval(cfg_text, {"Namespace": Namespace})
    except Exception:
        return {"_raw": cfg_text}

    return vars(namespace)


def find_point_cloud_iteration(model_path: str, iteration: int) -> Optional[int]:
    if iteration != -1:
        point_cloud_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
        return iteration if os.path.exists(point_cloud_path) else None

    point_cloud_root = os.path.join(model_path, "point_cloud")
    if not os.path.isdir(point_cloud_root):
        return None

    candidates = []
    for name in os.listdir(point_cloud_root):
        if not name.startswith("iteration_"):
            continue
        try:
            candidates.append(int(name.split("_")[-1]))
        except ValueError:
            continue
    return max(candidates) if candidates else None


def _channel_delta(rgb_array: np.ndarray) -> float:
    rg = np.abs(rgb_array[..., 0] - rgb_array[..., 1]).mean()
    rb = np.abs(rgb_array[..., 0] - rgb_array[..., 2]).mean()
    gb = np.abs(rgb_array[..., 1] - rgb_array[..., 2]).mean()
    return float((rg + rb + gb) / 3.0)


def analyze_rgb_array(rgb_array: np.ndarray, gray_threshold: float = 0.01) -> Dict:
    rgb_array = np.asarray(rgb_array).astype(np.float32)
    if rgb_array.max() > 1.0:
        rgb_array = rgb_array / 255.0

    mean_rgb = rgb_array.mean(axis=(0, 1))
    channel_std = rgb_array.std(axis=(0, 1))
    channel_delta = _channel_delta(rgb_array)
    return {
        "mean_rgb": [float(v) for v in mean_rgb],
        "std_rgb": [float(v) for v in channel_std],
        "mean_channel_delta": channel_delta,
        "grayscale_like": bool(channel_delta < gray_threshold),
    }


def analyze_color_vectors(color_vectors: np.ndarray, gray_threshold: float = 0.01) -> Dict:
    color_vectors = np.asarray(color_vectors).astype(np.float32)
    if color_vectors.ndim != 2 or color_vectors.shape[1] != 3:
        raise ValueError(f"Expected Nx3 color vectors, got shape={color_vectors.shape}")

    if color_vectors.size == 0:
        return {
            "count": 0,
            "mean_rgb": None,
            "std_rgb": None,
            "mean_channel_delta": None,
            "grayscale_like": None,
        }

    if color_vectors.max() > 1.0:
        color_vectors = color_vectors / 255.0

    mean_rgb = color_vectors.mean(axis=0)
    std_rgb = color_vectors.std(axis=0)
    channel_delta = _channel_delta(color_vectors[None, ...].reshape(1, color_vectors.shape[0], 3))
    return {
        "count": int(color_vectors.shape[0]),
        "mean_rgb": [float(v) for v in mean_rgb],
        "std_rgb": [float(v) for v in std_rgb],
        "mean_channel_delta": float(channel_delta),
        "grayscale_like": bool(channel_delta < gray_threshold),
    }


def analyze_image_path(image_path: str) -> Dict:
    with Image.open(image_path) as image:
        image_np = np.array(image)
        rgb = image_np[..., :3] if image_np.ndim == 3 else np.repeat(image_np[..., None], 3, axis=2)
        stats = analyze_rgb_array(rgb)
        stats.update(
            {
                "path": image_path,
                "mode": image.mode,
                "size": [int(image.size[0]), int(image.size[1])],
                "has_alpha": bool(image_np.ndim == 3 and image_np.shape[-1] == 4),
            }
        )
        return stats


def summarize_image_stats(stats_list: Iterable[Dict]) -> Dict:
    stats_list = list(stats_list)
    if not stats_list:
        return {"count": 0, "grayscale_like_count": 0, "grayscale_like_ratio": None}

    channel_deltas = [item["mean_channel_delta"] for item in stats_list]
    grayscale_like_count = sum(1 for item in stats_list if item["grayscale_like"])
    return {
        "count": len(stats_list),
        "grayscale_like_count": grayscale_like_count,
        "grayscale_like_ratio": float(grayscale_like_count / len(stats_list)),
        "mean_channel_delta": float(np.mean(channel_deltas)),
        "min_channel_delta": float(np.min(channel_deltas)),
        "max_channel_delta": float(np.max(channel_deltas)),
    }


def sample_blender_image_paths(source_path: str, split: str = "train", limit: int = 8) -> List[str]:
    transforms_path = os.path.join(source_path, f"transforms_{split}.json")
    if not os.path.exists(transforms_path):
        return []

    with open(transforms_path, encoding="utf-8") as json_file:
        contents = json.load(json_file)

    image_paths = []
    for frame in contents.get("frames", []):
        frame_path = frame.get("file_path")
        if not frame_path:
            continue

        candidate = os.path.join(source_path, frame_path + ".png")
        if os.path.exists(candidate):
            image_paths.append(candidate)
        if len(image_paths) >= limit:
            break
    return image_paths


def analyze_image_directory(image_dir: str, limit: int = 8) -> Tuple[Dict, List[Dict]]:
    if not os.path.isdir(image_dir):
        return {"exists": False, "count": 0, "grayscale_like_count": 0, "grayscale_like_ratio": None}, []

    image_names = [
        name
        for name in sorted(os.listdir(image_dir))
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ][:limit]
    stats_list = [analyze_image_path(os.path.join(image_dir, name)) for name in image_names]
    summary = summarize_image_stats(stats_list)
    summary["exists"] = True
    return summary, stats_list
