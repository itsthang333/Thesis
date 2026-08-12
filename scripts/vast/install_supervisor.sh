#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WSSS_WORKSPACE:-/workspace}"
REPO_DIR="${WSSS_REPO_DIR:-${WORKSPACE}/native-wsss}"

chmod +x "${REPO_DIR}/scripts/vast/run_managed_5090.sh"
install -m 0644 "${REPO_DIR}/scripts/vast/btxrd-wsss-supervisor.conf" \
  /etc/supervisor/conf.d/btxrd-wsss.conf
supervisorctl reread
supervisorctl update
supervisorctl status btxrd-wsss
