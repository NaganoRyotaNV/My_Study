#!/usr/bin/env bash
set -euo pipefail

CUDA_VER="${1:-12.0}"
ENV_NAME="viewpoint-graph"
ENV_YML="environment.base.yml"

if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/home/mi24a002/anaconda3/etc/profile.d/conda.sh" ]; then
  source "/home/mi24a002/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/home/mi24a002/miniconda3/etc/profile.d/conda.sh" ]; then
  source "/home/mi24a002/miniconda3/etc/profile.d/conda.sh"
else
  echo "[ERROR] conda.sh not found. Please install Anaconda/Miniconda or set the path." >&2
  exit 1
fi

echo "[INFO] CUDA_VER=${CUDA_VER}"
echo "[INFO] ENV_NAME=${ENV_NAME}"

if [ ! -f "${ENV_YML}" ]; then
  echo "[ERROR] ${ENV_YML} not found in current directory: $(pwd)" >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[INFO] Removing existing environment: ${ENV_NAME}"
  conda env remove -n "${ENV_NAME}" -y
fi

echo "[INFO] Creating environment from ${ENV_YML}"
conda env create -f "${ENV_YML}"

echo "[INFO] Activating ${ENV_NAME}"
conda activate "${ENV_NAME}"

echo "[INFO] Python executable: $(which python)"
python -V

echo "[INFO] Verifying key imports"
python - <<'PY'
mods = ["cv2", "numpy", "scipy", "sklearn", "PIL", "yaml", "matplotlib", "torch", "torchvision", "skimage", "ot"]
for m in mods:
    __import__(m)
    print("[OK]", m)
PY

echo "[DONE] Environment ${ENV_NAME} is ready."