#!/usr/bin/env bash
# Runs on the TPU VM after queued-resource provisioning. Idempotent:
# safe to re-run if the VM is recycled or the script is re-launched.
#
# Required env vars (passed in by the launcher):
#   WANDB_API_KEY   wandb auth
#   HF_TOKEN        huggingface auth
# Optional:
#   TRAIN_CONFIG    yaml config to pass to tpu_train.py (default: config_35M.yaml)
#   REPO_URL        git remote (default: github.com/JackMclaughlin424/...)
#   REPO_DIR        clone target (default: $HOME/csci739-project-mamba)

set -euxo pipefail

: "${WANDB_API_KEY:?WANDB_API_KEY must be set}"
: "${HF_TOKEN:?HF_TOKEN must be set}"
TRAIN_CONFIG="${TRAIN_CONFIG:-config_35M.yaml}"
REPO_URL="${REPO_URL:-https://github.com/JackMclaughlin424/csci739-project-mamba.git}"
REPO_DIR="${REPO_DIR:-$HOME/csci739-project-mamba}"

# 1. Wait for any in-flight apt/cloud-init, then disable unattended upgrades
#    so they can't grab the dpkg lock mid-install.
while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do sleep 5; done
sudo systemctl disable --now apt-daily.timer apt-daily-upgrade.timer || true

# 2. Install OS deps non-interactively.
export DEBIAN_FRONTEND=noninteractive
sudo -E apt-get update -y
sudo -E apt-get install -y git tmux

# 3. Clone (or update) the repo.
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" pull --ff-only
else
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

# 4. Python deps.
pip install -r requirements_tpu.txt

# 5. Persist env vars for future interactive shells (idempotent).
if ! grep -q PJRT_DEVICE ~/.bashrc; then
  cat >> ~/.bashrc <<'EOF'
export PJRT_DEVICE=TPU
export XLA_PERSISTENT_CACHE_PATH=$HOME/xla_cache
export PT_XLA_DEBUG_LEVEL=0
export XLA_METRICS=1
export PATH="$HOME/.local/bin:$PATH"
export TPU_RUNTIME_METRICS_PORTS=8431,8432,8433,8434
EOF
fi
mkdir -p "$HOME/xla_cache"

# 6. Export for THIS shell — ~/.bashrc is not sourced under non-interactive ssh.
export PJRT_DEVICE=TPU
export XLA_PERSISTENT_CACHE_PATH="$HOME/xla_cache"
export PT_XLA_DEBUG_LEVEL=0
export XLA_METRICS=1
export PATH="$HOME/.local/bin:$PATH"
export TPU_RUNTIME_METRICS_PORTS=8431,8432,8433,8434
export HF_TOKEN

# 7. wandb auth (writes ~/.netrc — also idempotent).
wandb login --relogin "$WANDB_API_KEY"

# 8. Launch training in a detached tmux session. Replaces any prior session.
tmux kill-session -t train 2>/dev/null || true
tmux new -d -s train \
  "cd '$REPO_DIR' && python tpu_train.py --config '$TRAIN_CONFIG' 2>&1 | tee \$HOME/train.log; exec bash"

echo "Training launched in tmux session 'train' with $TRAIN_CONFIG."
echo "Tail with: tmux attach -t train  (or)  tail -f ~/train.log"
