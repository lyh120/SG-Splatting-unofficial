# SG-Splatting (论文复现版)

本仓库在 3D Gaussian Splatting 基础上实现 SG-Splatting 的核心思想，并提供从环境配置到测评结果导出的完整流程。

## 1. 与论文复现要求对齐检查

已完成（核心一致）：
- SG 外观建模（Diffuse + 多个 SG lobes，视角相关颜色）。
- 正交多轴 SG（`--sg_axis_mode orthogonal`，每个高斯学习一个 SG 基旋转）。
- 分阶段训练（默认 `--sg_start_iter 2000`，可 warmup）。
- 自适应低阶 SH 融合（按投影尺寸动态选 0/1/2 阶）。
- 3DGS 稠密化/裁剪/不透明度重置训练策略。
- 训练、渲染、checkpoint、PLY 参数保存与恢复。
- 标准评测脚本（PSNR/SSIM/LPIPS），自动输出 JSON 结果。

仍属于工程近似（不是逐行官方代码）：
- SG 着色在 Python 中计算后传入 rasterizer（未改 CUDA 内核）。
- 自适应 SH 的投影尺寸使用高斯尺度近似估计（工程可复现，可能与论文实现细节有常数差异）。

## 1.1 项目原理（简述）

- 几何：沿用 3DGS 的高斯位置/尺度/旋转/不透明度优化，以及 densify/prune。
- 外观：每个高斯包含 `diffuse + SG lobes`，SG 用于建模视角相关高光与方向性辐射。
- 论文关键优化：
  - 正交多轴 SG（`orthogonal`）：每个高斯学习一个局部正交基，再派生 SG 方向，降低自由度并提升稳定性。
  - 分阶段引入 SG：前期先学几何与基础颜色，后期再逐步打开 SG（默认 2000 iter 后 warmup）。
  - 自适应低阶 SH：按投影尺寸选择 0/1/2 阶，平衡细节与稳定性。

## 2. 项目结构

- `train.py`：训练入口。
- `render.py`：渲染 train/test 视角。
- `evaluate.py`：评测 PSNR/SSIM/LPIPS，保存 `eval/metrics_iter_xxx.json`。
- `scene/gaussian_model.py`：高斯参数、SG/SH 参数、densify/prune、PLY IO。
- `gaussian_renderer/__init__.py`：SG + 自适应低阶 SH 颜色计算与 rasterizer 对接。

已删除与当前复现流程无关的旧模块（扩散/深度/额外可视化/Replica loader 等），避免依赖污染与复现歧义。

## 3. 环境配置（从零开始）

### 3.1 建议环境
- OS: Linux / Windows
- Python: 3.9 或 3.10
- CUDA: 与本机 PyTorch 对应版本一致
- GPU: NVIDIA（建议 >= 12GB 显存用于较大场景）

### 3.2 安装依赖
```bash
pip install -r requirements.txt
```

### 3.3 编译 CUDA 扩展
```bash
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

若此步未成功，`train.py` / `render.py` / `evaluate.py` 会因 `diff_gaussian_rasterization` 缺失而无法运行。

## 4. 数据准备

支持两类数据布局：

1. COLMAP 数据：
- `scene_root/sparse/0/...`
- `scene_root/images/...`

2. Blender/NeRF-synthetic：
- `scene_root/transforms_train.json`
- `scene_root/transforms_test.json`
- 图像文件

## 5. 训练（论文配置）

推荐直接使用 `--paper_mode`，它会自动锁定论文复现关键参数：
- `eval=True`
- `num_sg=3`
- `sg_axis_mode=orthogonal`
- `sg_start_iter=2000`
- `sg_warmup_iters=500`
- `use_adaptive_low_sh=True`
- `adaptive_sh_max_degree=2`
- `sh_small_radius_threshold=1.5`
- `sh_medium_radius_threshold=6.0`

### 5.1 COLMAP 场景（推荐）
```bash
python train.py -s /path/to/scene -m ./output/scene_sg \
  --paper_mode
```

### 5.2 Blender 场景
```bash
python train.py -s /path/to/lego -m ./output/lego_sg --white_background \
  --paper_mode
```

训练日志会输出测试集 `L1 / PSNR / SSIM`，并在 `--save_iterations` 对应迭代保存点云参数。

如需严格可复现，可附加：
```bash
--seed 0 --deterministic
```

## 6. 渲染结果

```bash
python render.py -s /path/to/scene -m ./output/scene_sg --iteration -1 --paper_mode
```

输出目录：
- `output/scene_sg/test/ours_<iter>/renders`
- `output/scene_sg/test/ours_<iter>/gt`

## 7. 测评结果（论文常用指标）

### 7.1 评测 test split
```bash
python evaluate.py -s /path/to/scene -m ./output/scene_sg --iteration -1 --split test --paper_mode
```

### 7.2 同时评测 train + test
```bash
python evaluate.py -s /path/to/scene -m ./output/scene_sg --iteration -1 --split both --paper_mode
```

### 7.3 不计算 LPIPS（加速）
```bash
python evaluate.py -s /path/to/scene -m ./output/scene_sg --iteration -1 --split test --skip_lpips --paper_mode
```

评测完成后会写入：
- `output/scene_sg/eval/metrics_iter_<iter>.json`

JSON 包含：
- 每个 split 的平均 `L1 / PSNR / SSIM / LPIPS`
- 每张图的逐帧指标（`per_view`）

## 8. 论文级复现实验建议

为降低偶然性，建议：
- 使用固定数据划分。
- 每个场景至少跑 3 次（不同 seed），报告均值与标准差。
- 汇总多个场景的平均 PSNR/SSIM/LPIPS，并和论文表格同口径对齐。

结果表模板（示例）：

| Scene | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| scene_a | xx.xx | 0.xxxx | 0.xxxx |
| scene_b | xx.xx | 0.xxxx | 0.xxxx |
| mean | xx.xx | 0.xxxx | 0.xxxx |

## 8.1 一键论文复现（推荐）

```bash
python repro_paper.py -s /path/to/scene -m ./output/scene_sg
```

这条命令会自动串行执行：
1. `train.py --paper_mode`
2. `render.py --paper_mode`
3. `evaluate.py --paper_mode`

可选参数：
- `--seed 0 --deterministic`：更强可复现
- `--skip_render` / `--skip_eval`：跳过阶段

## 9. 常见问题

- `ModuleNotFoundError: diff_gaussian_rasterization`  
  说明 CUDA 扩展未编译成功，重新执行第 3.3 节命令。

- LPIPS 导入失败  
  执行 `pip install lpips`，或评测时加 `--skip_lpips`。

- 显存不足  
  可降低输入分辨率、减少迭代轮次、关闭部分测试频率后再逐步恢复。
