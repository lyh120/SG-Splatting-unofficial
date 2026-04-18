# SG-Splatting

SG-Splatting 的工程化复现仓库，基于 3D Gaussian Splatting，并补充了论文思路中的球面高斯（Spherical Gaussian, SG）颜色建模、自适应低阶 SH、训练/渲染/评测脚本，以及颜色诊断工具。

本仓库当前目标是：

- 尽量贴近论文思路完成可运行复现
- 使用官方 3 通道 rasterizer 接口
- 提供从训练到渲染、评测、诊断的一站式流程

## Features

- 支持 `orthogonal_fixed` / `orthogonal_learned` / `free` 三种 SG 轴模式
- 支持 `sg_start_iter` + `sg_warmup_iters` 的论文式分阶段 SG 引入策略
- 支持自适应低阶 SH（0/1/2 阶）与 `approx` / `projected_cov` 两种尺寸判据
- 支持 COLMAP 与 Blender / NeRF-Synthetic 数据格式
- 提供一键复现脚本 `repro_paper.py`
- 提供颜色诊断脚本 `diagnose_color.py`

## Repository Structure

```text
SG-Splatting/
├── arguments/                  # 参数定义
├── gaussian_renderer/          # 渲染主逻辑
├── scene/                      # 数据读取、相机、GaussianModel
├── scripts/                    # 辅助脚本
├── submodules/                 # CUDA 扩展源码
│   ├── diff-gaussian-rasterization/
│   └── simple-knn/
├── utils/                      # 工具函数、诊断工具
├── train.py                    # 训练
├── render.py                   # 渲染
├── evaluate.py                 # 评测
├── repro_paper.py              # 一键复现流程
└── diagnose_color.py           # 彩色/灰度问题排查
```

## Environment

推荐环境：

- Python 3.10
- CUDA Toolkit 11.8
- PyTorch cu118

示例：

```bash
conda create -n sg_splatting python=3.10 -y
conda activate sg_splatting

pip install -U pip setuptools wheel ninja
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Build CUDA Extensions

本仓库依赖以下两个本地扩展：

- `submodules/diff-gaussian-rasterization`
- `submodules/simple-knn`

它们不是冗余目录，必须保留并正确编译。

推荐使用：

```bash
bash scripts/switch_official_rasterizer.sh
```

该脚本会：

1. 对齐官方 rasterizer 仓库
2. 拉取必要子模块（包括 `third_party/glm`）
3. 重新编译并安装 `diff_gaussian_rasterization`
4. 重新编译并安装 `simple_knn`

如果单独安装：

```bash
python -m pip install -e ./submodules/simple-knn --no-build-isolation
python -m pip install -e ./submodules/diff-gaussian-rasterization --no-build-isolation
```

## Supported Data Layouts

### 1. COLMAP

```text
<scene_root>/
├── images/
└── sparse/0/
```

### 2. Blender / NeRF-Synthetic

```text
<scene_root>/
├── transforms_train.json
├── transforms_test.json
└── train|test/*.png
```

对于 Blender / RGBA 场景，通常建议训练和渲染都带上 `--white_background`，以保持监督口径一致。

## Quick Start

```bash
cd ~/ll_further/SG-Splatting
DATA=~/ll_further/SG-Splatting/Chocolate
OUT=~/ll_further/SG-Splatting/output/chocolate_run
```

### 1. 训练

```bash
python train.py -s "$DATA" -m "$OUT" \
  --paper_mode --white_background --seed 0 \
  --iterations 5000 \
  --test_iterations 500 1000 2000 3000 4000 5000 \
  --save_iterations 500 1000 2000 3000 4000 5000
```

### 2. 渲染

```bash
python render.py -s "$DATA" -m "$OUT" \
  --iteration 5000 --paper_mode --white_background --seed 0
```

输出目录：

- `output/.../train/ours_5000/renders`
- `output/.../train/ours_5000/gt`
- `output/.../test/ours_5000/renders`
- `output/.../test/ours_5000/gt`

### 3. 评测

```bash
python evaluate.py -s "$DATA" -m "$OUT" \
  --iteration 5000 --split test --paper_mode --white_background --seed 0
```

输出文件：

- `output/.../eval/metrics_iter_5000.json`

## One-Click Reproduction

```bash
python repro_paper.py \
  -s ~/ll_further/SG-Splatting/Chocolate \
  -m ~/ll_further/SG-Splatting/output/chocolate_sg \
  --seed 0 \
  --white_background
```

默认流程：

1. 训练
2. 渲染
3. 颜色诊断
4. 评测

可选参数：

- `--iterations`
- `--skip_render`
- `--skip_diagnose`
- `--skip_eval`
- `--deterministic`
- `--extra_train_args`
- `--extra_render_args`
- `--extra_eval_args`
- `--extra_diagnose_args`

### Script Usage

#### `repro_paper.py`

用途：
- 一键执行训练、渲染、颜色诊断、评测全流程

必须外部传入：
- `-s` / `--source_path`：数据集根目录
- `-m` / `--model_path`：输出目录

常用可选参数：
- `--iterations`：训练迭代数，默认 `30000`
- `--white_background`：Blender / RGBA 数据通常建议开启
- `--seed`：默认 `0`
- `--deterministic`
- `--skip_render`
- `--skip_diagnose`
- `--skip_eval`
- `--extra_train_args`
- `--extra_render_args`
- `--extra_eval_args`
- `--extra_diagnose_args`

说明：
- `repro_paper.py` 会在内部调用 `train.py`、`render.py`、`evaluate.py` 时自动追加 `--paper_mode`
- 因此下面这条命令是走论文主线对齐模式的：

```bash
python repro_paper.py \
  -s "$DATA" \
  -m "$OUT" \
  --white_background \
  --iterations 5000
```

对应你的路径，完整示例为：

```bash
cd ~/ll_further/SG-Splatting_fen

OUT=~/ll_further/SG-Splatting_fen/output/chocolate_smoke
DATA=~/ll_further/SG-Splatting_fen/Chocolate

python repro_paper.py \
  -s "$DATA" \
  -m "$OUT" \
  --white_background \
  --iterations 5000
```

#### `train.py`

用途：
- 只执行训练

必须外部传入：
- `-s` / `--source_path`
- `-m` / `--model_path`

常用可选参数：
- `--paper_mode`
- `--white_background`
- `--iterations`
- `--seed`
- `--deterministic`
- `--test_iterations`
- `--save_iterations`
- `--checkpoint_iterations`
- `--start_checkpoint`

如果不使用 `--paper_mode`，还可以手动控制：
- `--num_sg`
- `--sg_axis_mode`：`orthogonal_fixed` / `orthogonal_learned` / `free`
- `--sg_start_iter`
- `--sg_warmup_iters`
- `--use_adaptive_low_sh`
- `--adaptive_sh_max_degree`
- `--adaptive_sh_size_metric`：`approx` / `projected_cov`
- `--paper_strict_sg_lrs`

示例：

```bash
cd ~/ll_further/SG-Splatting_fen

OUT=~/ll_further/SG-Splatting_fen/output/chocolate_smoke
DATA=~/ll_further/SG-Splatting_fen/Chocolate

python train.py \
  -s "$DATA" \
  -m "$OUT" \
  --paper_mode \
  --white_background \
  --iterations 5000 \
  --test_iterations 500 1000 2000 3000 4000 5000 \
  --save_iterations 500 1000 2000 3000 4000 5000
```

#### `render.py`

用途：
- 从已有模型目录加载指定迭代结果并导出渲染图

必须外部传入：
- `-s` / `--source_path`
- `-m` / `--model_path`

常用可选参数：
- `--iteration`：指定渲染哪个迭代；`-1` 表示自动加载最新迭代
- `--paper_mode`
- `--white_background`
- `--skip_train`
- `--skip_test`
- `--debug_color_stats`
- `--seed`
- `--deterministic`

示例：

```bash
cd ~/ll_further/SG-Splatting_fen

OUT=~/ll_further/SG-Splatting_fen/output/chocolate_smoke
DATA=~/ll_further/SG-Splatting_fen/Chocolate

python render.py \
  -s "$DATA" \
  -m "$OUT" \
  --iteration 5000 \
  --paper_mode \
  --white_background
```

如果想自动加载最新迭代：

```bash
python render.py \
  -s "$DATA" \
  -m "$OUT" \
  --iteration -1 \
  --paper_mode \
  --white_background
```

#### `evaluate.py`

用途：
- 对已有模型进行指标评测，输出 PSNR / SSIM / LPIPS 等结果

必须外部传入：
- `-s` / `--source_path`
- `-m` / `--model_path`

常用可选参数：
- `--iteration`：`-1` 表示自动加载最新迭代
- `--split`：`train` / `test` / `both`
- `--paper_mode`
- `--white_background`
- `--skip_lpips`
- `--lpips_net`：`alex` / `vgg`
- `--seed`
- `--deterministic`

示例：

```bash
cd ~/ll_further/SG-Splatting_fen

OUT=~/ll_further/SG-Splatting_fen/output/chocolate_smoke
DATA=~/ll_further/SG-Splatting_fen/Chocolate

python evaluate.py \
  -s "$DATA" \
  -m "$OUT" \
  --iteration 5000 \
  --split test \
  --paper_mode \
  --white_background
```

## Paper Mode

`--paper_mode` 会自动启用一组论文主线对齐参数：

- `eval=True`
- `num_sg=3`
- `sg_axis_mode=orthogonal_fixed`
- `sg_start_iter=2000`
- `sg_warmup_iters=500`
- `use_adaptive_low_sh=True`
- `adaptive_sh_max_degree=2`
- `adaptive_sh_size_metric=projected_cov`
- `sh_small_radius_threshold=1.5`
- `sh_medium_radius_threshold=6.0`

在该模式下：

- 前 `sg_start_iter` 次迭代冻结 SG 与 low-degree SH 参数
- `sg_warmup_iters` 只用于 SG 分支的平滑混合，不再代替 staged training
- 自适应 SH 使用论文式 2D footprint 尺寸估计
- 正交 SG 使用固定三轴 `(1,0,0)`, `(0,1,0)`, `(0,0,1)`

如果你想做严格对比实验，建议固定：

- 数据划分
- 随机种子
- 训练迭代数
- 是否 `white_background`
- 评测口径

## Render Debugging

如果需要检查颜色到底在什么阶段出了问题，可以使用：

```bash
python render.py -s "$DATA" -m "$OUT" \
  --iteration 5000 --paper_mode --white_background --seed 0 \
  --debug_color_stats
```

它会额外打印：

- `gaussians.get_diffuse`
- `gaussians.get_sg_amplitude`
- `render_pkg.colors_precomp`
- `render image`
- `gt image`

适合定位：

- 模型颜色参数是否本身已经塌成灰度
- SG 颜色分支是否没学起来
- 颜色是否在进入 rasterizer 之前就被压坏

## Color Diagnosis

如果 Blender / RGBA 场景训练后渲染结果发灰、偏白或接近黑白，优先运行：

```bash
python diagnose_color.py -s "$DATA" -m "$OUT" --iteration -1 --white_background
```

该脚本会自动检查：

- 数据集类型是否正确识别
- `cfg_args` 中的训练参数是否与当前渲染参数一致
- `point_cloud/iteration_xxx` 是否与目标迭代匹配
- Blender 源图是否本身接近灰度
- `train/test/ours_xxx/gt` 是否本身接近灰度
- `train/test/ours_xxx/renders` 是否接近灰度

输出文件：

- `output/.../diagnostics/color_diagnosis_iter_xxx.json`
- `output/.../render_report_iter_xxx.json`

## Important Color Fix

本仓库已修复一个与自适应低阶 SH 相关的颜色 bug：

- 问题位置：`gaussian_renderer/__init__.py`
- 问题表现：Blender 彩色 GT 正常，但渲染输出接近黑白或灰白
- 根因：`eval_sh(...)` 的结果处于 SH 域，之前没有通过 `SH2RGB(...)` 转回 RGB 域，导致颜色在 clamp 前被错误压成接近 0

修复后：

- `colors_precomp` 会正确回到 RGB 域
- 渲染输出不再被硬性压成单通道/黑白

注意：

如果你的旧 checkpoint 是在该 bug 修复前训练得到的，即使修复代码后重新渲染，颜色也可能仍然偏灰。此时建议从头重训。

## Recommended Re-Training After Color Fix

如果你此前遇到过：

- 渲染结果黑白
- 渲染结果灰白
- `render_pkg.colors_precomp` 接近 0

建议在修复代码后重新训练：

```bash
python train.py -s "$DATA" -m "${OUT}_fixed" \
  --paper_mode --white_background --seed 0 \
  --iterations 5000
```

然后重新渲染：

```bash
python render.py -s "$DATA" -m "${OUT}_fixed" \
  --iteration 5000 --paper_mode --white_background --seed 0 --debug_color_stats
```

## Common Issues

### 1. `ModuleNotFoundError: diff_gaussian_rasterization`

扩展未正确编译或当前环境未安装。优先执行：

```bash
bash scripts/switch_official_rasterizer.sh
```

### 2. `ModuleNotFoundError: simple_knn`

重新安装：

```bash
python -m pip uninstall -y simple_knn
python -m pip install -e ./submodules/simple-knn --no-build-isolation
```

### 3. `glm/glm.hpp: No such file or directory`

说明 rasterizer 子模块不完整：

```bash
cd submodules/diff-gaussian-rasterization
git submodule update --init --recursive
```

### 4. Blender / RGBA 不加 `--white_background` 指标异常

这是常见现象。对于 Blender / RGBA 数据，训练和渲染通常都应使用 `--white_background`。

### 5. 渲染结果是黑白或灰白

请按下面顺序排查：

1. 先运行 `diagnose_color.py`
2. 再运行 `render.py --debug_color_stats`
3. 确认源图与 GT 是否彩色
4. 确认训练和渲染的 `--white_background` 是否一致
5. 如果是修复前训练得到的旧 checkpoint，直接重训

## Outputs

训练过程中的常见输出：

- `cfg_args`
- `point_cloud/iteration_xxx/point_cloud.ply`
- `render_report_iter_xxx.json`
- `diagnostics/color_diagnosis_iter_xxx.json`
- `eval/metrics_iter_xxx.json`

## Status

当前仓库适合：

- SG-Splatting 思路复现
- 训练/渲染/评测实验
- Blender / COLMAP 数据测试
- 彩色结果问题排查

如果你要做更严格的论文级数值复现，建议额外固定：

- 数据版本
- 环境版本
- CUDA / PyTorch 版本
- 子模块 commit
- 随机种子
- 多次运行后的统计均值和方差

## License

请同时遵循本仓库以及所依赖子模块的原始许可证，尤其是：

- `submodules/diff-gaussian-rasterization`
- `submodules/simple-knn`
- 相关上游 3DGS / SH 实现
