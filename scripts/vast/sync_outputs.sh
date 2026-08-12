#!/usr/bin/env bash
set -euo pipefail

: "${SYNC_URI:?Set SYNC_URI, for example s3:bucket/native-wsss}"
WORKSPACE="${WSSS_WORKSPACE:-/workspace}"
rclone sync "${WORKSPACE}/outputs" "${SYNC_URI}/outputs" --fast-list --transfers 16 --checkers 32 --progress
