#!/usr/bin/env bash
set -uo pipefail

if [[ -f /venv/main/bin/activate ]]; then
  source /venv/main/bin/activate
fi

WORKSPACE="${WSSS_WORKSPACE:-/workspace}"
REPO_DIR="${WSSS_REPO_DIR:-${WORKSPACE}/native-wsss}"
LOG_DIR="${WORKSPACE}/logs"
MAX_ATTEMPTS="${WSSS_MAX_ATTEMPTS:-3}"
MAIN_CONFIG="${WSSS_CONFIG:-${REPO_DIR}/configs/pipeline.yaml}"
MAIN_OUTPUT="${WORKSPACE}/outputs/hrnet_rich_gallery_5090"
BENCHMARK_OUTPUT="${WORKSPACE}/outputs/benchmark_100_epoch1"
BENCHMARK_CONFIG="${REPO_DIR}/configs/pipeline_benchmark.yaml"
mkdir -p "${LOG_DIR}"

stop_instance() {
  if [[ -n "${CONTAINER_ID:-}" && -n "${CONTAINER_API_KEY:-}" ]] \
    && command -v vastai >/dev/null; then
    printf '%s stopping Vast instance %s after exit code %d\n' "$(date --iso-8601=seconds)" \
      "${CONTAINER_ID}" "${1}" | tee -a "${LOG_DIR}/managed.log"
    vastai stop instance "${CONTAINER_ID}" --api-key "${CONTAINER_API_KEY}" \
      >> "${LOG_DIR}/managed.log" 2>&1 || true
  fi
}

# An interactive setup may already be running the one-epoch warm-up. Wait for it
# rather than starting a second trainer on the same GPU.
while pgrep -f "pipeline_epoch1.yaml train-hrnet" >/dev/null; do
  printf '%s waiting for epoch-1 warm-up\n' "$(date --iso-8601=seconds)" \
    | tee -a "${LOG_DIR}/managed.log"
  sleep 60
done

if [[ ! -s "${MAIN_OUTPUT}/checkpoints/hrnet_last.pt" ]]; then
  printf '%s epoch-1 checkpoint is missing after warm-up\n' "$(date --iso-8601=seconds)" \
    | tee -a "${LOG_DIR}/managed.log"
  stop_instance 1
  exit 1
fi

if [[ ! -f "${BENCHMARK_OUTPUT}/benchmark.complete" ]]; then
  python "${REPO_DIR}/scripts/vast/prepare_benchmark_config.py" \
    --source "${MAIN_CONFIG}" \
    --target "${BENCHMARK_CONFIG}" \
    --output-dir "${BENCHMARK_OUTPUT}" \
    --checkpoint-source "${MAIN_OUTPUT}/checkpoints"
  WSSS_CONFIG="${BENCHMARK_CONFIG}" WSSS_OUTPUT_DIR="${BENCHMARK_OUTPUT}" \
    bash "${REPO_DIR}/scripts/vast/benchmark_100.sh" 2>&1 \
    | tee -a "${LOG_DIR}/benchmark_100.log"
  benchmark_code=${PIPESTATUS[0]}
  if (( benchmark_code != 0 )); then
    printf '%s benchmark failed with exit code %d\n' "$(date --iso-8601=seconds)" \
      "${benchmark_code}" | tee -a "${LOG_DIR}/managed.log"
    stop_instance "${benchmark_code}"
    exit "${benchmark_code}"
  fi
  touch "${BENCHMARK_OUTPUT}/benchmark.complete"
fi

attempt=1
exit_code=1
while (( attempt <= MAX_ATTEMPTS )); do
  printf '%s managed pipeline attempt %d/%d\n' "$(date --iso-8601=seconds)" \
    "${attempt}" "${MAX_ATTEMPTS}" | tee -a "${LOG_DIR}/managed.log"
  bash "${REPO_DIR}/scripts/vast/run_5090.sh" 2>&1 | tee -a "${LOG_DIR}/managed.log"
  exit_code=${PIPESTATUS[0]}
  if (( exit_code == 0 )); then
    break
  fi
  printf '%s attempt %d failed with exit code %d\n' "$(date --iso-8601=seconds)" \
    "${attempt}" "${exit_code}" | tee -a "${LOG_DIR}/managed.log"
  attempt=$((attempt + 1))
  if (( attempt <= MAX_ATTEMPTS )); then
    sleep 60
  fi
done

stop_instance "${exit_code}"

exit "${exit_code}"
