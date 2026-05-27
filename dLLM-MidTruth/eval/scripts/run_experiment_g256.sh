#!/usr/bin/env bash
# 256-token generation length, 4-GPU. See run_experiment_core.sh.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GEN_LENGTH=256 TAG=g256 MASTER_PORT_BASE=29600 bash "${SCRIPT_DIR}/run_experiment_core.sh"
