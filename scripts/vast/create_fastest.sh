#!/usr/bin/env bash
set -euo pipefail

: "${OFFER_ID:?Set OFFER_ID from search_fastest.sh}"
: "${OFFER_DPH:?Set OFFER_DPH from the selected offer}"
DISK_GB="${DISK_GB:-400}"
IMAGE="${VAST_IMAGE:-vastai/pytorch:cuda-12.9.2-auto}"
MAX_DPH="${MAX_DPH:-0.60}"

python - "${OFFER_DPH}" "${MAX_DPH}" <<'PY'
import sys
price, maximum = map(float, sys.argv[1:])
if price > maximum:
    raise SystemExit(f"Refusing ${price:.3f}/h offer above ${maximum:.3f}/h ceiling")
for hours in (12, 24, 30):
    print(f"{hours:>2} hours: at most ${price * hours:.2f} compute")
PY

vastai create instance "${OFFER_ID}" \
  --image "${IMAGE}" \
  --disk "${DISK_GB}" \
  --ssh \
  --direct \
  --label btxrd-wsss-rtx5090 \
  --onstart-cmd 'mkdir -p /workspace/{data,models,outputs,cache,logs} && chmod -R 777 /workspace'
