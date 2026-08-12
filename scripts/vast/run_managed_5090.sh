#!/usr/bin/env bash
set -uo pipefail

if [[ -f /venv/main/bin/activate ]]; then
  source /venv/main/bin/activate
fi

WORKSPACE="${WSSS_WORKSPACE:-/workspace}"
REPO_DIR="${WSSS_REPO_DIR:-${WORKSPACE}/native-wsss}"
LOG_DIR="${WORKSPACE}/logs"
MAX_ATTEMPTS="${WSSS_MAX_ATTEMPTS:-3}"
mkdir -p "${LOG_DIR}"

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

if [[ -n "${CONTAINER_ID:-}" && -n "${CONTAINER_API_KEY:-}" ]] && command -v vastai >/dev/null; then
  printf '%s stopping Vast instance %s after exit code %d\n' "$(date --iso-8601=seconds)" \
    "${CONTAINER_ID}" "${exit_code}" | tee -a "${LOG_DIR}/managed.log"
  vastai stop instance "${CONTAINER_ID}" --api-key "${CONTAINER_API_KEY}" \
    >> "${LOG_DIR}/managed.log" 2>&1 || true
fi

exit "${exit_code}"
