#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WSSS_WORKSPACE:-/workspace}"
REPO_DIR="${WSSS_REPO_DIR:-${WORKSPACE}/native-wsss}"
CONFIG="${WSSS_CONFIG:-${REPO_DIR}/configs/pipeline.yaml}"

cd "${REPO_DIR}"
btxrd-wsss --config "${CONFIG}" source-maps --limit 100
btxrd-wsss --config "${CONFIG}" sam-gallery --limit 100
btxrd-wsss --config "${CONFIG}" rad-dino --limit 100
python scripts/vast/report_eta.py --output "${WORKSPACE}/outputs/hrnet_rich_gallery_5090" --images 3746
