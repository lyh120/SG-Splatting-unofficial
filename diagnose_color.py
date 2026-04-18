import json
import os
from argparse import ArgumentParser

from utils.diagnostics import (
    analyze_image_directory,
    analyze_image_path,
    find_point_cloud_iteration,
    infer_dataset_type,
    load_cfg_args,
    sample_blender_image_paths,
    summarize_image_stats,
)


def build_source_report(source_path: str, dataset_type: str, sample_limit: int):
    if dataset_type != "blender":
        return {
            "dataset_type": dataset_type,
            "message": "Source color sampling is currently implemented for Blender/NeRF-synthetic layouts only.",
        }

    sample_paths = sample_blender_image_paths(source_path, split="train", limit=sample_limit)
    sample_stats = [analyze_image_path(path) for path in sample_paths]
    return {
        "dataset_type": dataset_type,
        "sample_count": len(sample_stats),
        "summary": summarize_image_stats(sample_stats),
        "samples": sample_stats,
    }


def build_render_report(model_path: str, iteration: int, split: str, sample_limit: int):
    split_root = os.path.join(model_path, split, f"ours_{iteration}")
    render_summary, render_samples = analyze_image_directory(os.path.join(split_root, "renders"), limit=sample_limit)
    gt_summary, gt_samples = analyze_image_directory(os.path.join(split_root, "gt"), limit=sample_limit)
    return {
        "split": split,
        "iteration": iteration,
        "paths": {
            "root": split_root,
            "renders": os.path.join(split_root, "renders"),
            "gt": os.path.join(split_root, "gt"),
        },
        "renders": {"summary": render_summary, "samples": render_samples},
        "gt": {"summary": gt_summary, "samples": gt_samples},
    }


def compare_expectations(cfg_args, cli_white_background):
    report = {
        "train_cfg_white_background": cfg_args.get("white_background"),
        "cli_white_background": cli_white_background,
        "consistent": None,
        "warning": None,
    }
    if "white_background" in cfg_args:
        report["consistent"] = bool(cfg_args["white_background"]) == bool(cli_white_background)
        if not report["consistent"]:
            report["warning"] = (
                "Training cfg_args and current CLI white_background do not match. "
                "For Blender/RGBA data, training and rendering should use the same background setting."
            )
    return report


if __name__ == "__main__":
    parser = ArgumentParser(description="Diagnose color/grayscale issues for Blender/RGBA SG-Splatting results")
    parser.add_argument("-s", "--source_path", required=True, type=str)
    parser.add_argument("-m", "--model_path", required=True, type=str)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--sample_limit", default=8, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    args = parser.parse_args()

    source_path = os.path.abspath(args.source_path)
    model_path = os.path.abspath(args.model_path)
    dataset_type = infer_dataset_type(source_path)
    cfg_args = load_cfg_args(model_path)
    resolved_iteration = find_point_cloud_iteration(model_path, args.iteration)

    report = {
        "source_path": source_path,
        "model_path": model_path,
        "dataset_type": dataset_type,
        "requested_iteration": args.iteration,
        "resolved_iteration": resolved_iteration,
        "cfg_args": cfg_args,
        "white_background_check": compare_expectations(cfg_args, args.white_background),
        "source_report": build_source_report(source_path, dataset_type, args.sample_limit),
        "render_reports": {},
        "warnings": [],
    }

    if resolved_iteration is None:
        report["warnings"].append("Could not resolve a trained point_cloud iteration under model_path/point_cloud.")
    else:
        if not args.skip_train:
            report["render_reports"]["train"] = build_render_report(model_path, resolved_iteration, "train", args.sample_limit)
        if not args.skip_test:
            report["render_reports"]["test"] = build_render_report(model_path, resolved_iteration, "test", args.sample_limit)

    if report["white_background_check"]["warning"] is not None:
        report["warnings"].append(report["white_background_check"]["warning"])

    source_summary = report["source_report"].get("summary")
    if source_summary and source_summary.get("grayscale_like_ratio") == 1.0:
        report["warnings"].append("All sampled source images look grayscale-like. The Blender export itself may lack color.")

    for split_name, split_report in report["render_reports"].items():
        render_summary = split_report["renders"]["summary"]
        gt_summary = split_report["gt"]["summary"]
        if gt_summary.get("exists") and gt_summary.get("grayscale_like_ratio") == 1.0:
            report["warnings"].append(f"{split_name} GT images look grayscale-like, so the source supervision may already be grayscale.")
        if render_summary.get("exists") and render_summary.get("grayscale_like_ratio") == 1.0:
            report["warnings"].append(f"{split_name} rendered images look grayscale-like.")

    diagnostics_dir = os.path.join(model_path, "diagnostics")
    os.makedirs(diagnostics_dir, exist_ok=True)
    suffix = resolved_iteration if resolved_iteration is not None else "missing"
    out_path = os.path.join(diagnostics_dir, f"color_diagnosis_iter_{suffix}.json")
    with open(out_path, "w", encoding="utf-8") as out_file:
        json.dump(report, out_file, indent=2, ensure_ascii=False)

    print(f"Dataset type: {dataset_type}")
    print(f"Resolved iteration: {resolved_iteration}")
    print(f"Training cfg_args white_background: {cfg_args.get('white_background')}")
    print(f"Current CLI white_background: {args.white_background}")
    for warning in report["warnings"]:
        print(f"[WARN] {warning}")
    print(f"Saved diagnosis to: {out_path}")
