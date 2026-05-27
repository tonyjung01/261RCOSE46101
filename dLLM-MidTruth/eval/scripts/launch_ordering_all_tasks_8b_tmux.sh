#!/usr/bin/env bash
# Queue run_ordering_all_tasks_8b.sh as a single tsp job (serial, all 4 GPUs)
# and monitor its output in a dedicated tmux session.
#
# Usage:
#   bash scripts/launch_ordering_all_tasks_8b_tmux.sh
#
# The sweep runs one experiment at a time using all 4 GPUs via torchrun.
# tsp -S 1 ensures no other tsp job runs concurrently.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EVAL_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$EVAL_DIR"

SESSION_NAME="${SESSION_NAME:-ordering_8b_all_tasks}"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[error] tmux session already exists: $SESSION_NAME"
  echo "  Kill it first: tmux kill-session -t $SESSION_NAME"
  exit 1
fi

# Set tsp to process one job at a time (so 4-GPU runs never overlap)
tsp -S 1

# Submit the sweep; tsp prints the job id to stdout
job_id=$(tsp bash "$SCRIPT_DIR/run_ordering_all_tasks_8b.sh" 0 1 2 3)

echo "tsp job $job_id queued  (tsp -S 1, all 4 GPUs per run)"
echo "Starting tmux session: $SESSION_NAME"

# Open a tmux window that follows the job output from the start
tmux new-session -d -s "$SESSION_NAME" \
  "echo '[monitor] tsp job $job_id — following output...'; tsp -f $job_id; echo ''; echo '[done] job $job_id finished. Press Enter to close.'; read"

echo ""
echo "======================================================"
echo "  tmux session : $SESSION_NAME"
echo "  tsp job id   : $job_id"
echo "======================================================"
echo "  Attach  : tmux attach -t $SESSION_NAME"
echo "  Job list: tsp"
echo "  Job tail: tsp -t"
echo "======================================================"
