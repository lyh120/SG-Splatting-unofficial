# SG-Splatting（论文思路复现版）

本项目基于 3D Gaussian Splatting，实现了 SG-Splatting 的核心思路，并提供从环境部署、训练、渲染到评测的完整流程。

## 1. 当前实现状态（与你本次排查结论一致）

- 已对齐到 **官方 3 通道 rasterizer 接口**（不再依赖 11 通道兼容补丁）。
- 已支持论文关键训练策略：
  - `orthogonal` 多轴 SG
  - `sg_start_iter` 分阶段启用 SG
  - 自适应低阶 SH（0/1/2）
- 提供一键流程脚本：`repro_paper.py`（train → render → evaluate）。

## 2. `submodules` 是否冗余？

不是冗余，必须保留：

- `submodules/diff-gaussian-rasterization`：核心可微光栅化 CUDA 扩展
- `submodules/simple-knn`：初始化和几何处理依赖的 CUDA 扩展

这些模块需要本地源码编译，不能只靠纯 pip 依赖替代。  
可以删的是编译残留/缓存（例如旧 Python 版本 `.so`、`__pycache__`），本仓库已清理明显冗余残留。

## 3. 环境配置（Linux）

建议环境：
- Python 3.10
- CUDA Toolkit 11.8
- PyTorch cu118

```bash
conda create -n sg_splatting python=3.10 -y
conda activate sg_splatting

pip install -U pip setuptools wheel ninja
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 4. 切换并安装官方 rasterizer（推荐）

```bash
cd ~/ll_further/SG-Splatting
bash scripts/switch_official_rasterizer.sh
```

该脚本会：
1. 将 `diff-gaussian-rasterization` 对齐到官方仓库
2. 拉取子模块（含 `third_party/glm`）
3. 重新编译安装 `diff_gaussian_rasterization` 与 `simple_knn`
4. 自动导入验证

## 5. 数据格式要求

只支持以下两种场景根目录：

1. COLMAP
- `<scene_root>/sparse/0/...`
- `<scene_root>/images/...`

2. Blender / NeRF-Synthetic
- `<scene_root>/transforms_train.json`
- `<scene_root>/transforms_test.json`

## 6. 快速开始（你当前最建议流程）

```bash
cd ~/ll_further/SG-Splatting
DATA=~/ll_further/SG-Splatting/Chocolate
OUT=~/ll_further/SG-Splatting/output/chocolate_run
```

### 6.1 快速收敛验证（5k）

对于 Blender/RGBA 数据，推荐带 `--white_background`：

```bash
rm -rf "$OUT"
python train.py -s "$DATA" -m "$OUT" \
  --paper_mode --white_background --seed 0 \
  --iterations 5000 \
  --test_iterations 500 1000 2000 3000 4000 5000 \
  --save_iterations 500 1000 2000 3000 4000 5000
```

### 6.2 渲染输出图片

```bash
python render.py -s "$DATA" -m "$OUT" \
  --iteration 5000 --paper_mode --white_background --seed 0
```

输出目录：
- `output/.../train/ours_5000/renders`
- `output/.../train/ours_5000/gt`
- `output/.../test/ours_5000/renders`
- `output/.../test/ours_5000/gt`

### 6.3 评测（PSNR/SSIM/LPIPS）

```bash
python evaluate.py -s "$DATA" -m "$OUT" \
  --iteration 5000 --split test --paper_mode --white_background --seed 0
```

指标文件：
- `output/.../eval/metrics_iter_5000.json`

## 7. 一键全流程（30k）

```bash
python repro_paper.py \
  -s ~/ll_further/SG-Splatting/Chocolate \
  -m ~/ll_further/SG-Splatting/output/chocolate_sg \
  --seed 0 --white_background
```

## 8. 常见问题

### 8.1 `ModuleNotFoundError: diff_gaussian_rasterization`
- 扩展未编译或当前环境未安装，重跑：
  - `bash scripts/switch_official_rasterizer.sh`

### 8.2 `ModuleNotFoundError: simple_knn`
- 重新安装：
```bash
python -m pip uninstall -y simple_knn
python -m pip install -e ./submodules/simple-knn --no-build-isolation
```

### 8.3 `glm/glm.hpp: No such file or directory`
- 官方子模块未拉取完整，执行：
```bash
cd submodules/diff-gaussian-rasterization
git submodule update --init --recursive
```

### 8.4 不加 `--white_background` 指标异常或几乎不变
- 对 Blender/RGBA 场景，通常应使用 `--white_background` 保持监督口径一致。

## 9. 论文口径说明

本仓库当前目标是“论文思路 + 官方 3 通道接口”的工程复现。  
若需要更严格的数值复现，请固定：
- 相同数据划分
- 相同随机种子
- 相同训练迭代与评测口径
- 多次运行后统计 mean/std
