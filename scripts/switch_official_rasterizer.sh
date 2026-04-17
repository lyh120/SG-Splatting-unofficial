#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RASTER_DIR="${ROOT_DIR}/submodules/diff-gaussian-rasterization"

echo "[INFO] Root: ${ROOT_DIR}"
echo "[INFO] Rasterizer dir: ${RASTER_DIR}"

if [[ -d "${RASTER_DIR}/.git" ]]; then
  echo "[INFO] Existing git repo detected, resetting to official remote."
  git -C "${RASTER_DIR}" remote remove origin >/dev/null 2>&1 || true
  git -C "${RASTER_DIR}" remote add origin https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
  git -C "${RASTER_DIR}" fetch origin
  git -C "${RASTER_DIR}" checkout main
  git -C "${RASTER_DIR}" reset --hard origin/main
  git -C "${RASTER_DIR}" submodule sync --recursive
  git -C "${RASTER_DIR}" submodule update --init --recursive
else
  echo "[INFO] Replacing directory with official repo clone."
  rm -rf "${RASTER_DIR}"
  git clone --recursive https://github.com/graphdeco-inria/diff-gaussian-rasterization.git "${RASTER_DIR}"
fi

if [[ ! -f "${RASTER_DIR}/third_party/glm/glm/glm.hpp" ]]; then
  echo "[ERROR] GLM header missing: ${RASTER_DIR}/third_party/glm/glm/glm.hpp"
  echo "[ERROR] Please check GitHub/submodule network access and rerun this script."
  exit 2
fi

echo "[INFO] Reinstalling CUDA extensions (no-build-isolation)."
python -m pip uninstall -y diff_gaussian_rasterization || true
python -m pip install -e "${RASTER_DIR}" --no-build-isolation
python -m pip install -e "${ROOT_DIR}/submodules/simple-knn" --no-build-isolation

echo "[INFO] Verifying imports."
python - <<'PY'
import torch
import diff_gaussian_rasterization
import simple_knn._C
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("official rasterizer + simple_knn import: OK")
PY

echo "[DONE] Official rasterizer switch complete."
