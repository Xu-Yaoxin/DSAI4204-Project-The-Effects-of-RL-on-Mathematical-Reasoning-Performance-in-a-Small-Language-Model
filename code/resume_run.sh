#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -x .venv/bin/python ]]; then
    printf '%s\n' '[ERROR] Prepare the server environment with setup_gpu.sh first.' >&2
    exit 2
fi
command -v setsid >/dev/null
.venv/bin/python scripts/stop_run.py --root "$PWD" --check-only
.venv/bin/python scripts/prepare_resume_upgrade.py
mkdir -p logs
RLVR_RUN_LOG="$(mktemp logs/train-$(date +%Y%m%d-%H%M%S)-XXXXXX.log)"
nohup setsid bash run.sh all --resume "$@" > "$RLVR_RUN_LOG" 2>&1 < /dev/null &
RLVR_RUN_PID=$!
ln -sfn "$RLVR_RUN_LOG" train-current.log
printf '%s\n' "$RLVR_RUN_PID" > logs/latest.pid
sleep 1
if ! kill -0 "$RLVR_RUN_PID" 2>/dev/null; then
    tail -n 50 "$RLVR_RUN_LOG"
    printf '%s\n' '[ERROR] The pipeline exited during startup. See the log above.' >&2
    exit 1
fi
printf '[STARTED] PID=%s | log=%s\n' "$RLVR_RUN_PID" "$RLVR_RUN_LOG"
printf '%s\n' '[LOG] tail -n 50 -f /root/code/train-current.log'
