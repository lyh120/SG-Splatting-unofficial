import json
import os
from argparse import ArgumentParser

import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.diagnostics import analyze_color_vectors, analyze_rgb_array, summarize_image_stats
from utils.general_utils import safe_state
from utils.repro_utils import apply_paper_defaults, paper_repro_summary


def _format_stats(label, stats):
    mean_delta = stats["mean_channel_delta"]
    mean_delta_str = "None" if mean_delta is None else f"{mean_delta:.6f}"
    return (
        f"{label}: grayscale_like={stats['grayscale_like']} "
        f"mean_channel_delta={mean_delta_str} mean_rgb={stats['mean_rgb']}"
    )


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, dataset_args, debug_color_stats=False):
    render_path = os.path.join(model_path, name, f"ours_{iteration}", "renders")
    gt_path = os.path.join(model_path, name, f"ours_{iteration}", "gt")
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gt_path, exist_ok=True)
    render_stats = []
    gt_stats = []

    for idx, view in enumerate(tqdm(views, desc=f"Rendering {name}")):
        render_pkg = render(
            view,
            gaussians,
            pipeline,
            background,
            sg_weight=1.0,
            use_adaptive_low_sh=dataset_args.use_adaptive_low_sh,
            adaptive_sh_max_degree=dataset_args.adaptive_sh_max_degree,
            adaptive_sh_size_metric=dataset_args.adaptive_sh_size_metric,
            sh_small_radius_threshold=dataset_args.sh_small_radius_threshold,
            sh_medium_radius_threshold=dataset_args.sh_medium_radius_threshold,
        )
        rendering = render_pkg["render"]
        gt = view.original_image[0:3, :, :]
        render_stats.append(analyze_rgb_array(rendering.detach().permute(1, 2, 0).cpu().numpy()))
        gt_stats.append(analyze_rgb_array(gt.detach().permute(1, 2, 0).cpu().numpy()))

        if debug_color_stats and idx == 0:
            diffuse_stats = analyze_color_vectors(gaussians.get_diffuse.detach().cpu().numpy())
            sg_stats = analyze_color_vectors(
                gaussians.get_sg_amplitude.detach().reshape(-1, 3).cpu().numpy()
            )
            precomp_stats = analyze_color_vectors(render_pkg["colors_precomp"].detach().cpu().numpy())
            render_image_stats = render_stats[-1]
            gt_image_stats = gt_stats[-1]
            print(f"[COLOR DEBUG][{name}] image={view.image_name}")
            print("[COLOR DEBUG] " + _format_stats("gaussians.get_diffuse", diffuse_stats))
            print("[COLOR DEBUG] " + _format_stats("gaussians.get_sg_amplitude", sg_stats))
            print("[COLOR DEBUG] " + _format_stats("render_pkg.colors_precomp", precomp_stats))
            print("[COLOR DEBUG] " + _format_stats("render image", render_image_stats))
            print("[COLOR DEBUG] " + _format_stats("gt image", gt_image_stats))

        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{idx:05d}.png"))
        torchvision.utils.save_image(gt, os.path.join(gt_path, f"{idx:05d}.png"))

    return {
        "split": name,
        "num_views": len(views),
        "render_summary": summarize_image_stats(render_stats),
        "gt_summary": summarize_image_stats(gt_stats),
    }


if __name__ == "__main__":
    parser = ArgumentParser(description="SG-Splatting rendering script")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--debug_color_stats", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--paper_mode", action="store_true")
    args = get_combined_args(parser)
    if args.paper_mode:
        args = apply_paper_defaults(args)
        print(paper_repro_summary(args))

    safe_state(args.quiet, seed=args.seed, deterministic=args.deterministic)
    dataset = model.extract(args)
    gaussians = GaussianModel(
        sh_degree=dataset.sh_degree,
        num_sg=dataset.num_sg,
        sg_init_sharpness=dataset.sg_init_sharpness,
        diffuse_bias=dataset.sg_diffuse_bias,
        sg_axis_mode=dataset.sg_axis_mode,
        adaptive_sh_max_degree=dataset.adaptive_sh_max_degree,
        adaptive_sh_size_metric=dataset.adaptive_sh_size_metric,
    )
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )

    render_report = {
        "iteration": scene.loaded_iter,
        "model_path": dataset.model_path,
        "source_path": dataset.source_path,
        "white_background": bool(dataset.white_background),
        "background_rgb": [1, 1, 1] if dataset.white_background else [0, 0, 0],
        "splits": {},
    }

    if not args.skip_train:
        render_report["splits"]["train"] = render_set(
            dataset.model_path,
            "train",
            scene.loaded_iter,
            scene.getTrainCameras(),
            gaussians,
            pipeline.extract(args),
            background,
            dataset,
            debug_color_stats=args.debug_color_stats,
        )
    if not args.skip_test:
        render_report["splits"]["test"] = render_set(
            dataset.model_path,
            "test",
            scene.loaded_iter,
            scene.getTestCameras(),
            gaussians,
            pipeline.extract(args),
            background,
            dataset,
            debug_color_stats=args.debug_color_stats,
        )

    report_path = os.path.join(dataset.model_path, f"render_report_iter_{scene.loaded_iter}.json")
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(render_report, report_file, indent=2, ensure_ascii=False)
    print(f"Saved render report to: {report_path}")
