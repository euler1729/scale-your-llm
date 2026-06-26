#!/usr/bin/env bash
# Launch the llama.cpp backend (port 8000) + the chat UI (port 3000).
# Usage: ./ui/start.sh   (run from repo root or anywhere)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_BIN="$REPO/llama.cpp/build/bin/llama-server"
MODEL_PATH="${MODEL_PATH:-$REPO/models/qwen-Q4_K_M.gguf}"
PYTHON="$REPO/.venv/bin/python"

cleanup() { kill "${BACK_PID:-}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "[start] launching llama-server on :8000 ($MODEL_PATH) — logs: /tmp/llama-server.log"
"$LLAMA_BIN" -m "$MODEL_PATH" --host 0.0.0.0 --port 8000 -c 4096 >/tmp/llama-server.log 2>&1 &
BACK_PID=$!

# Don't block on the backend — start the UI immediately.
echo "[start] launching UI on http://localhost:3000"
LLAMA_SERVER="http://0.0.0.0:8000" MODEL="qwen-Q4_K_M" PORT=3000 "$PYTHON" "$REPO/ui/server.py"
