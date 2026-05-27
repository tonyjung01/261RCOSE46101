#!/usr/bin/env bash
# 512-token generation length, 4-GPU. See run_experiment_core.sh.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GEN_LENGTH=512 TAG=g512 MASTER_PORT_BASE=29700 bash "${SCRIPT_DIR}/run_experiment_core.sh"
