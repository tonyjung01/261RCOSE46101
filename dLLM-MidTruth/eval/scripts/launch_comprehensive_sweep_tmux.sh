#!/usr/bin/env bash
# Queue run_comprehensive_sweep.sh via tsp (serial) and monitor in tmux.
#
# Usage:
#   bash scripts/launch_comprehensive_sweep_tmux.sh
#
# All 113 runs execute sequentially, one 4-GPU job at a time.
# tsp -S 1 ensures no overlap with other queued tsp jobs.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EVAL_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$EVAL_DIR"

SESSION_NAME="${SESSION_NAME:-comprehensive_sweep}"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[error] tmux session already exists: $SESSION_NAME"
  echo "  Kill first: tmux kill-session -t $SESSION_NAME"
  exit 1
fi

tsp -S 1

job_id=$(tsp bash "$SCRIPT_DIR/run_comprehensive_sweep.sh" 0 1 2 3)

echo "tsp job $job_id queued  (tsp -S 1, 113 runs, all 4 GPUs per run)"

tmux new-session -d -s "$SESSION_NAME" \
  "echo '[monitor] tsp job $job_id — following output...'; tsp -f $job_id; echo ''; echo '[done] job $job_id finished.'; read"

echo ""
echo "======================================================"
echo "  tmux session : $SESSION_NAME"
echo "  tsp job id   : $job_id"
echo "======================================================"
echo "  Attach  : tmux attach -t $SESSION_NAME"
echo "  Job list: tsp"
echo "  Job tail: tsp -t"
echo "======================================================"
