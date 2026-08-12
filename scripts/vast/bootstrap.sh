#!/usr/bin/env bash
set -euo pipefail

if [[ -f /venv/main/bin/activate ]]; then
  source /venv/main/bin/activate
fi

WORKSPACE="${WSSS_WORKSPACE:-/workspace}"
REPO_DIR="${WSSS_REPO_DIR:-${WORKSPACE}/native-wsss}"

mkdir -p "${WORKSPACE}"/{data,models,outputs,cache,logs}
export HF_HOME="${WORKSPACE}/cache/huggingface"
export TORCH_HOME="${WORKSPACE}/cache/torch"
export XDG_CACHE_HOME="${WORKSPACE}/cache"
export WANDB_DIR="${WORKSPACE}/outputs/wandb"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY

python -m pip install --upgrade pip setuptools wheel
python -m pip install "vastai>=0.2"
python -m pip install -e "${REPO_DIR}[train,sam,dev]"

SAM_BACKEND="$(python -c "import yaml; print(yaml.safe_load(open('${REPO_DIR}/configs/pipeline.yaml'))['sam']['backend'])")"
if [[ "${SAM_BACKEND}" == "sam_med2d" ]]; then
  SAM_CHECKPOINT="${WORKSPACE}/models/sam-med2d_b.pth"
  if [[ ! -s "${SAM_CHECKPOINT}" ]]; then
    SAM_DOWNLOAD="${SAM_CHECKPOINT}.official"
    gdown 1ARiB5RkSsWmAB_8mqWnwDF8ZKTtFwsjl -O "${SAM_DOWNLOAD}"
    python - "${SAM_DOWNLOAD}" "${SAM_CHECKPOINT}.part" <<'PY'
import sys

import torch

source, destination = sys.argv[1:]
payload = torch.load(source, map_location="cpu", weights_only=False)
if not isinstance(payload, dict) or "model" not in payload:
    raise RuntimeError("Official SAM-Med2D checkpoint is missing its model state")
torch.save(payload["model"], destination)
PY
    mv "${SAM_CHECKPOINT}.part" "${SAM_CHECKPOINT}"
    rm -f "${SAM_DOWNLOAD}"
  fi
elif [[ "${SAM_BACKEND}" == "sam_vit_b" ]]; then
  SAM_CHECKPOINT="${WORKSPACE}/models/sam_vit_b_01ec64.pth"
  if [[ ! -s "${SAM_CHECKPOINT}" ]]; then
    wget -q --show-progress -O "${SAM_CHECKPOINT}.part" \
      https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    mv "${SAM_CHECKPOINT}.part" "${SAM_CHECKPOINT}"
  fi
else
  echo "Unsupported SAM backend: ${SAM_BACKEND}" >&2
  exit 1
fi

nvidia-smi
btxrd-wsss --config "${REPO_DIR}/configs/pipeline.yaml" preflight --require-assets
btxrd-wsss --config "${REPO_DIR}/configs/pipeline.yaml" smoke-models
echo "Bootstrap complete: RTX 5090 environment and ${SAM_BACKEND} checkpoint verified."
