#!/usr/bin/env bash
# Local-side launcher: polls one or more queued TPU resources and runs
# tpu_bootstrap.sh on each as soon as it transitions to ACTIVE.
#
# Usage:
#   ZONE=us-central2-b QRS="qr-1 qr-2" \
#   WANDB_API_KEY=... HF_TOKEN=... \
#   ./tpu_launch.sh
#
# Optional env:
#   TRAIN_CONFIG     yaml passed through to bootstrap (default: config_35M.yaml)
#   POLL_INTERVAL    seconds between state checks (default: 60)
#   PROJECT          gcloud --project (default: gcloud's active config)
#
# Run under nohup or your own tmux so closing your laptop doesn't kill it:
#   nohup ./tpu_launch.sh > launch.log 2>&1 &

set -euo pipefail

: "${ZONE:?set ZONE, e.g. us-central2-b}"
: "${QRS:?set QRS to a space-separated list of queued-resource IDs}"
: "${WANDB_API_KEY:?set WANDB_API_KEY}"
: "${HF_TOKEN:?set HF_TOKEN}"

TRAIN_CONFIG="${TRAIN_CONFIG:-config_35M.yaml}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"
PROJECT_FLAG=()
[ -n "${PROJECT:-}" ] && PROJECT_FLAG=(--project="$PROJECT")

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BOOTSTRAP="$SCRIPT_DIR/tpu_bootstrap.sh"
[ -f "$BOOTSTRAP" ] || { echo "missing $BOOTSTRAP" >&2; exit 1; }

launch_one() {
  local QR=$1
  local node_id state

  while :; do
    state=$(gcloud compute tpus queued-resources describe "$QR" \
              --zone="$ZONE" "${PROJECT_FLAG[@]}" \
              --format='value(state.state)' 2>/dev/null || echo UNKNOWN)
    case "$state" in
      ACTIVE) echo "[$QR] $(date -Is) ACTIVE"; break ;;
      FAILED|SUSPENDED|CANCELLED)
        echo "[$QR] terminal state $state — giving up" >&2; return 1 ;;
      *) echo "[$QR] $(date -Is) state=$state, sleeping $POLL_INTERVAL s" ;;
    esac
    sleep "$POLL_INTERVAL"
  done

  # Resolve node ID from the queued resource (usually equals QR id).
  node_id=$(gcloud compute tpus queued-resources describe "$QR" \
              --zone="$ZONE" "${PROJECT_FLAG[@]}" \
              --format='value(tpu.nodeSpec.nodeId)')
  [ -n "$node_id" ] || node_id="$QR"
  echo "[$QR] node=$node_id"

  # SSH key bootstrap is implicit on first ssh; retry briefly while it propagates.
  for i in 1 2 3 4 5; do
    gcloud compute tpus tpu-vm scp "$BOOTSTRAP" "$node_id":~/tpu_bootstrap.sh \
      --zone="$ZONE" "${PROJECT_FLAG[@]}" --worker=all && break
    echo "[$QR] scp attempt $i failed, retrying in 15s"; sleep 15
  done

  # Run bootstrap on every worker. v4-8 has 1 worker but be explicit for larger pods.
  gcloud compute tpus tpu-vm ssh "$node_id" \
    --zone="$ZONE" "${PROJECT_FLAG[@]}" --worker=all \
    --command="WANDB_API_KEY='$WANDB_API_KEY' HF_TOKEN='$HF_TOKEN' TRAIN_CONFIG='$TRAIN_CONFIG' bash ~/tpu_bootstrap.sh"

  echo "[$QR] bootstrap complete; tmux session 'train' is running on $node_id"
}

pids=()
for qr in $QRS; do
  launch_one "$qr" &
  pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=$?; done
exit "$rc"
