#!/usr/bin/env bash
set -euo pipefail

MAX_DPH="${MAX_DPH:-0.60}"
DISK_GB="${DISK_GB:-400}"
FILTER="rentable=true verified=true num_gpus=1 reliability>=0.98 direct_port_count>=1 disk_space>=${DISK_GB} disk_bw>=1000 inet_down>=500 cuda_max_good>=12.8 dph_total<=${MAX_DPH} gpu_name=RTX_5090"

echo "Verified RTX 5090 offers at or below USD ${MAX_DPH}/hour"
vastai search offers "${FILTER}" -o 'dlperf-'
