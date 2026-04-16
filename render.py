import os
from argparse import ArgumentParser

import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.repro_utils import apply_paper_defaults, paper_repro_summary


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, dataset_args):
    render_path = os.path.join(model_path, name, f"ours_{iteration}", "renders")
    gt_path = os.path.join(model_path, name, f"ours_{iteration}", "gt")
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gt_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc=f"Rendering {name}")):
        rendering = render(
            view,
            gaussians,
            pipeline,
            background,
            sg_weight=1.0,
            use_adaptive_low_sh=dataset_args.use_adaptive_low_sh,
            adaptive_sh_max_degree=dataset_args.adaptive_sh_max_degree,
            sh_small_radius_threshold=dataset_args.sh_small_radius_threshold,
            sh_medium_radius_threshold=dataset_args.sh_medium_radius_threshold,
        )["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{idx:05d}.png"))
        torchvision.utils.save_image(gt, os.path.join(gt_path, f"{idx:05d}.png"))


if __name__ == "__main__":
    parser = ArgumentParser(description="SG-Splatting rendering script")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
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
    )
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )

    if not args.skip_train:
        render_set(
            dataset.model_path,
            "train",
            scene.loaded_iter,
            scene.getTrainCameras(),
            gaussians,
            pipeline.extract(args),
            background,
            dataset,
        )
    if not args.skip_test:
        render_set(
            dataset.model_path,
            "test",
            scene.loaded_iter,
            scene.getTestCameras(),
            gaussians,
            pipeline.extract(args),
            background,
            dataset,
        )
