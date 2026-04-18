import os
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint

import torch
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from utils.repro_utils import apply_paper_defaults, paper_repro_summary

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def prepare_output_and_logger(args):
    dataset_name = os.path.basename(args.source_path.rstrip("/\\"))
    if not args.model_path:
        unique_str = str(uuid.uuid4())[:8]
        args.model_path = os.path.join("./output/sg_splatting", f"{dataset_name}_{unique_str}")

    args.model_path = os.path.abspath(args.model_path)
    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w", encoding="utf-8") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    if TENSORBOARD_FOUND:
        return SummaryWriter(args.model_path)
    print("Tensorboard not available: progress will only be printed to stdout")
    return None


def get_sg_weight(iteration: int, start_iter: int, warmup_iters: int) -> float:
    if iteration < start_iter:
        return 0.0
    if warmup_iters <= 0:
        return 1.0
    return min(1.0, (iteration - start_iter + 1) / float(warmup_iters))


def evaluate(scene, gaussians, pipe, background):
    test_cameras = scene.getTestCameras()
    if not test_cameras:
        return None

    l1_total = 0.0
    psnr_total = 0.0
    ssim_total = 0.0
    with torch.no_grad():
        for viewpoint in test_cameras:
            render_pkg = render(
                viewpoint,
                gaussians,
                pipe,
                background,
                sg_weight=1.0,
                use_adaptive_low_sh=getattr(scene.dataset_args, "use_adaptive_low_sh", False),
                adaptive_sh_max_degree=getattr(scene.dataset_args, "adaptive_sh_max_degree", 2),
                adaptive_sh_size_metric=getattr(scene.dataset_args, "adaptive_sh_size_metric", "approx"),
                sh_small_radius_threshold=getattr(scene.dataset_args, "sh_small_radius_threshold", 1.5),
                sh_medium_radius_threshold=getattr(scene.dataset_args, "sh_medium_radius_threshold", 6.0),
            )
            pred = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt = viewpoint.original_image.cuda()
            l1_total += l1_loss(pred, gt).item()
            psnr_total += psnr(pred, gt).mean().item()
            ssim_total += ssim(pred, gt).item()

    num = len(test_cameras)
    return {"l1": l1_total / num, "psnr": psnr_total / num, "ssim": ssim_total / num}


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(
        sh_degree=dataset.sh_degree,
        num_sg=dataset.num_sg,
        sg_init_sharpness=dataset.sg_init_sharpness,
        diffuse_bias=dataset.sg_diffuse_bias,
        sg_axis_mode=dataset.sg_axis_mode,
        adaptive_sh_max_degree=dataset.adaptive_sh_max_degree,
        adaptive_sh_size_metric=dataset.adaptive_sh_size_metric,
    )
    scene = Scene(dataset, gaussians, resolution_scales=[1.0])
    gaussians.training_setup(opt)

    if checkpoint:
        model_params, first_iter = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = scene.getTrainCameras(scale=1.0).copy()
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        gaussians.update_learning_rate(iteration)
        sg_phase_enabled = iteration >= dataset.sg_start_iter
        gaussians.set_sg_phase(sg_phase_enabled, strict_lrs=opt.paper_strict_sg_lrs)

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras(scale=1.0).copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        sg_weight = get_sg_weight(iteration, dataset.sg_start_iter, dataset.sg_warmup_iters)

        bg = background if dataset.white_background else torch.rand((3), device="cuda")
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            bg,
            sg_weight=sg_weight,
            use_adaptive_low_sh=dataset.use_adaptive_low_sh and sg_phase_enabled,
            adaptive_sh_max_degree=dataset.adaptive_sh_max_degree,
            adaptive_sh_size_metric=dataset.adaptive_sh_size_metric,
            sh_small_radius_threshold=dataset.sh_small_radius_threshold,
            sh_medium_radius_threshold=dataset.sh_medium_radius_threshold,
        )

        image = render_pkg["render"]
        gt_image = viewpoint_cam.original_image.cuda()
        if viewpoint_cam.gt_alpha_mask is not None:
            gt_alpha = viewpoint_cam.gt_alpha_mask.cuda()
            gt_image = gt_image * gt_alpha + (1.0 - gt_alpha) * bg[:, None, None]

        Ll1 = l1_loss(image, gt_image)
        recon_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        reg_loss = opt.lambda_sg_reg * gaussians.sg_regularization() if sg_phase_enabled else torch.zeros((), device="cuda")
        loss = recon_loss + reg_loss
        loss.backward()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            visibility_filter = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]
            viewspace_point_tensor = render_pkg["viewspace_points"]

            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "loss": f"{ema_loss_for_log:.5f}",
                        "points": f"{len(gaussians.get_xyz)}",
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.opacity_cull,
                        scene.cameras_extent,
                        size_threshold,
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), os.path.join(scene.model_path, f"chkpnt{iteration}.pth"))

            if tb_writer is not None:
                tb_writer.add_scalar("train/loss_total", loss.item(), iteration)
                tb_writer.add_scalar("train/loss_recon", recon_loss.item(), iteration)
                tb_writer.add_scalar("train/loss_sg_reg", reg_loss.item(), iteration)
                tb_writer.add_scalar("train/num_points", gaussians.get_xyz.shape[0], iteration)
                tb_writer.add_scalar("train/sg_weight", sg_weight, iteration)
                tb_writer.add_scalar("train/sg_phase_enabled", float(sg_phase_enabled), iteration)

            if iteration in testing_iterations:
                metrics = evaluate(scene, gaussians, pipe, background)
                if metrics is not None:
                    print(
                        f"\n[ITER {iteration}] Test L1 {metrics['l1']:.6f} "
                        f"PSNR {metrics['psnr']:.3f} SSIM {metrics['ssim']:.4f}"
                    )
                    if tb_writer is not None:
                        tb_writer.add_scalar("test/l1", metrics["l1"], iteration)
                        tb_writer.add_scalar("test/psnr", metrics["psnr"], iteration)
                        tb_writer.add_scalar("test/ssim", metrics["ssim"], iteration)


if __name__ == "__main__":
    parser = ArgumentParser(description="SG-Splatting training script")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations",
        nargs="+",
        type=int,
        default=[1_000, 5_000, 10_000, 15_000, 20_000, 30_000],
    )
    parser.add_argument(
        "--save_iterations",
        nargs="+",
        type=int,
        default=[1_000, 5_000, 10_000, 15_000, 20_000, 30_000],
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--paper_mode", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    if args.paper_mode:
        args = apply_paper_defaults(args)
        print(paper_repro_summary(args))
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    safe_state(args.quiet, seed=args.seed, deterministic=args.deterministic)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
    )
    print("\nTraining complete.")
