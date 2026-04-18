import argparse
import os
import shlex
import subprocess
import sys


def run_cmd(cmd):
    print(f"\n[RUN] {cmd}")
    process = subprocess.run(cmd, shell=True)
    if process.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="One-click SG-Splatting paper-mode runner (staged SG, projected-cov adaptive SH, fixed orthogonal axes)"
    )
    parser.add_argument("-s", "--source_path", required=True, type=str)
    parser.add_argument("-m", "--model_path", required=True, type=str)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--skip_render", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--skip_diagnose", action="store_true")
    parser.add_argument("--extra_train_args", type=str, default="")
    parser.add_argument("--extra_render_args", type=str, default="")
    parser.add_argument("--extra_eval_args", type=str, default="")
    parser.add_argument("--extra_diagnose_args", type=str, default="")
    args = parser.parse_args()

    python = shlex.quote(sys.executable)
    source_path = shlex.quote(os.path.abspath(args.source_path))
    model_path = shlex.quote(os.path.abspath(args.model_path))
    deterministic_flag = "--deterministic" if args.deterministic else ""
    wb_flag = "--white_background" if args.white_background else ""

    train_cmd = (
        f"{python} train.py -s {source_path} -m {model_path} "
        f"--iterations {args.iterations} --paper_mode --seed {args.seed} "
        f"{deterministic_flag} {wb_flag} {args.extra_train_args}"
    )
    run_cmd(train_cmd)

    if not args.skip_render:
        render_cmd = (
            f"{python} render.py -s {source_path} -m {model_path} --iteration -1 "
            f"--paper_mode --seed {args.seed} {deterministic_flag} {wb_flag} {args.extra_render_args}"
        )
        run_cmd(render_cmd)

    if not args.skip_diagnose:
        diagnose_cmd = (
            f"{python} diagnose_color.py -s {source_path} -m {model_path} --iteration -1 "
            f"{wb_flag} {args.extra_diagnose_args}"
        )
        run_cmd(diagnose_cmd)

    if not args.skip_eval:
        eval_cmd = (
            f"{python} evaluate.py -s {source_path} -m {model_path} --iteration -1 --split test "
            f"--paper_mode --seed {args.seed} {deterministic_flag} {wb_flag} {args.extra_eval_args}"
        )
        run_cmd(eval_cmd)

    print("\nPaper-mode reproduction workflow finished.")
