from argparse import Namespace


def apply_paper_defaults(args: Namespace) -> Namespace:
    args.eval = True
    args.num_sg = 3
    args.sg_axis_mode = "orthogonal"
    args.sg_start_iter = 2000
    args.sg_warmup_iters = 500
    args.use_adaptive_low_sh = True
    args.adaptive_sh_max_degree = 2
    args.sh_small_radius_threshold = 1.5
    args.sh_medium_radius_threshold = 6.0
    return args


def paper_repro_summary(args: Namespace) -> str:
    return (
        "paper_mode=ON | "
        f"eval={args.eval}, num_sg={args.num_sg}, sg_axis_mode={args.sg_axis_mode}, "
        f"sg_start_iter={args.sg_start_iter}, sg_warmup_iters={args.sg_warmup_iters}, "
        f"use_adaptive_low_sh={args.use_adaptive_low_sh}, adaptive_sh_max_degree={args.adaptive_sh_max_degree}, "
        f"sh_small_radius_threshold={args.sh_small_radius_threshold}, "
        f"sh_medium_radius_threshold={args.sh_medium_radius_threshold}"
    )
