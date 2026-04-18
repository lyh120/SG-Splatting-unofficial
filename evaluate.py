import json
import os
from argparse import ArgumentParser
from datetime import datetime

import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import ssim
from utils.repro_utils import apply_paper_defaults, paper_repro_summary

try:
    import lpips

    LPIPS_FOUND = True
except ImportError:
    LPIPS_FOUND = False


def evaluate_split(views, gaussians, pipeline, background, dataset, lpips_net=None):
    l1_total = 0.0
    psnr_total = 0.0
    ssim_total = 0.0
    lpips_total = 0.0
    per_view = []

    with torch.no_grad():
        for view in tqdm(views, desc="Evaluating"):
            render_pkg = render(
                view,
                gaussians,
                pipeline,
                background,
                sg_weight=1.0,
                use_adaptive_low_sh=dataset.use_adaptive_low_sh,
                adaptive_sh_max_degree=dataset.adaptive_sh_max_degree,
                adaptive_sh_size_metric=dataset.adaptive_sh_size_metric,
                sh_small_radius_threshold=dataset.sh_small_radius_threshold,
                sh_medium_radius_threshold=dataset.sh_medium_radius_threshold,
            )
            pred = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt = view.original_image.cuda()

            l1_val = torch.abs(pred - gt).mean().item()
            psnr_val = psnr(pred, gt).mean().item()
            ssim_val = ssim(pred, gt).item()

            lpips_val = None
            if lpips_net is not None:
                pred_lpips = pred[None, ...] * 2.0 - 1.0
                gt_lpips = gt[None, ...] * 2.0 - 1.0
                lpips_val = lpips_net(pred_lpips, gt_lpips).item()
                lpips_total += lpips_val

            l1_total += l1_val
            psnr_total += psnr_val
            ssim_total += ssim_val
            per_view.append(
                {
                    "image_name": view.image_name,
                    "l1": l1_val,
                    "psnr": psnr_val,
                    "ssim": ssim_val,
                    "lpips": lpips_val,
                }
            )

    count = len(views)
    avg = {
        "l1": l1_total / count,
        "psnr": psnr_total / count,
        "ssim": ssim_total / count,
        "lpips": (lpips_total / count) if lpips_net is not None else None,
    }
    return avg, per_view


if __name__ == "__main__":
    parser = ArgumentParser(description="SG-Splatting evaluation script")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--split", default="test", choices=["train", "test", "both"])
    parser.add_argument("--lpips_net", default="alex", choices=["alex", "vgg"])
    parser.add_argument("--skip_lpips", action="store_true")
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
    pipe = pipeline.extract(args)
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )

    lpips_net = None
    if not args.skip_lpips:
        if LPIPS_FOUND:
            lpips_net = lpips.LPIPS(net=args.lpips_net).cuda().eval()
        else:
            print("LPIPS package not found, falling back to PSNR/SSIM/L1 only.")

    split_map = {
        "train": scene.getTrainCameras(),
        "test": scene.getTestCameras(),
    }
    target_splits = ["train", "test"] if args.split == "both" else [args.split]

    report = {
        "timestamp": datetime.now().isoformat(),
        "iteration": scene.loaded_iter,
        "model_path": dataset.model_path,
        "source_path": dataset.source_path,
        "splits": {},
    }

    for split_name in target_splits:
        views = split_map[split_name]
        if not views:
            print(f"Skip split={split_name}: no cameras.")
            continue
        avg_metrics, per_view = evaluate_split(views, gaussians, pipe, background, dataset, lpips_net=lpips_net)
        report["splits"][split_name] = {
            "count": len(views),
            "avg": avg_metrics,
            "per_view": per_view,
        }
        print(
            f"[{split_name}] count={len(views)} "
            f"L1={avg_metrics['l1']:.6f} PSNR={avg_metrics['psnr']:.3f} SSIM={avg_metrics['ssim']:.4f}"
            + (f" LPIPS={avg_metrics['lpips']:.4f}" if avg_metrics["lpips"] is not None else "")
        )

    eval_dir = os.path.join(dataset.model_path, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    out_path = os.path.join(eval_dir, f"metrics_iter_{scene.loaded_iter}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved metrics to: {out_path}")
