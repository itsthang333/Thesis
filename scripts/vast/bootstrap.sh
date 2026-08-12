#!/usr/bin/env bash
set -euo pipefail

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
python -m pip install -e "${REPO_DIR}[train,sam,dev]"

SAM_CHECKPOINT="${WORKSPACE}/models/sam_vit_b_01ec64.pth"
if [[ ! -s "${SAM_CHECKPOINT}" ]]; then
  wget -q --show-progress -O "${SAM_CHECKPOINT}.part" \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
  mv "${SAM_CHECKPOINT}.part" "${SAM_CHECKPOINT}"
fi

nvidia-smi
btxrd-wsss --config "${REPO_DIR}/configs/pipeline.yaml" preflight --require-assets
btxrd-wsss --config "${REPO_DIR}/configs/pipeline.yaml" smoke-models
echo "Bootstrap complete: RTX 5090 environment and SAM ViT-B checkpoint verified."
