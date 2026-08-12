#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WSSS_WORKSPACE:-/workspace}"
REPO_DIR="${WSSS_REPO_DIR:-${WORKSPACE}/native-wsss}"
CONFIG="${WSSS_CONFIG:-${REPO_DIR}/configs/pipeline.yaml}"
LOG_DIR="${WORKSPACE}/logs"
mkdir -p "${LOG_DIR}"
export HF_HOME="${WORKSPACE}/cache/huggingface"
export TORCH_HOME="${WORKSPACE}/cache/torch"
export XDG_CACHE_HOME="${WORKSPACE}/cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY
export PYTHONUNBUFFERED=1

cd "${REPO_DIR}"
btxrd-wsss --config "${CONFIG}" preflight --require-assets | tee "${LOG_DIR}/00_preflight.log"
btxrd-wsss --config "${CONFIG}" train-hrnet 2>&1 | tee "${LOG_DIR}/01_hrnet.log"
btxrd-wsss --config "${CONFIG}" source-maps 2>&1 | tee "${LOG_DIR}/02_sources.log"
btxrd-wsss --config "${CONFIG}" sam-gallery 2>&1 | tee "${LOG_DIR}/03_sam.log"
btxrd-wsss --config "${CONFIG}" rad-dino 2>&1 | tee "${LOG_DIR}/04_rad_dino.log"
btxrd-wsss --config "${CONFIG}" train-g1 2>&1 | tee "${LOG_DIR}/05_g1.log"
btxrd-wsss --config "${CONFIG}" select 2>&1 | tee "${LOG_DIR}/06_select.log"
btxrd-wsss --config "${CONFIG}" evaluate --splits val,test 2>&1 | tee "${LOG_DIR}/07_evaluate.log"
