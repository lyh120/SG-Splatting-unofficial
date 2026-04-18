from argparse import ArgumentParser, Namespace
import os
import sys


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none: bool = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]

            arg_type = type(value)
            value = value if not fill_none else None
            names = ["--" + key]
            if shorthand:
                names.append("-" + key[0:1])

            if arg_type == bool:
                group.add_argument(*names, default=value, action="store_true")
            elif arg_type == list:
                group.add_argument(*names, default=value, nargs="+")
            else:
                group.add_argument(*names, default=value, type=arg_type)

    def extract(self, args):
        group = GroupParams()
        for arg_name, arg_value in vars(args).items():
            if arg_name in vars(self) or ("_" + arg_name) in vars(self):
                setattr(group, arg_name, arg_value)
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel: bool = False):
        self.sh_degree = 0
        self.num_sg = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.sg_init_sharpness = 8.0
        self.sg_diffuse_bias = 0.5
        self.sg_axis_mode = "orthogonal_learned"
        self.sg_start_iter = 2000
        self.sg_warmup_iters = 500
        self.use_adaptive_low_sh = False
        self.adaptive_sh_max_degree = 2
        self.adaptive_sh_size_metric = "approx"
        self.sh_small_radius_threshold = 1.5
        self.sh_medium_radius_threshold = 6.0
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        group = super().extract(args)
        group.source_path = os.path.abspath(group.source_path)
        group.model_path = os.path.abspath(group.model_path) if group.model_path else group.model_path
        return group


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.diffuse_lr = 0.0025
        self.sg_color_lr = 0.0025
        self.sg_axis_lr = 0.001
        self.sg_basis_lr = 0.001
        self.sg_sharpness_lr = 0.001
        self.sh_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_sg_reg = 0.0001
        self.opacity_cull = 0.005
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.paper_strict_sg_lrs = False
        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdline_args = parser.parse_args(sys.argv[1:])
    cfgfile_string = "Namespace()"

    try:
        cfgfilepath = os.path.join(cmdline_args.model_path, "cfg_args")
        with open(cfgfilepath, encoding="utf-8") as cfg_file:
            cfgfile_string = cfg_file.read()
            print(f"Config file found: {cfgfilepath}")
    except Exception:
        pass

    args_cfgfile = eval(cfgfile_string)
    merged_dict = vars(args_cfgfile).copy()
    for key, value in vars(cmdline_args).items():
        if value is not None:
            merged_dict[key] = value
    return Namespace(**merged_dict)
