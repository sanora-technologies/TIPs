#!/bin/bash
# Launch the TIPs inference API. Original behaviour is preserved when jemalloc and
# ngrok are both present; each is now optional so the script runs on a bare box.
ROOT=/home/oaiz/Documents/sanora/TIPs
PORT="${PORT:-7863}"

source "$ROOT/venv/bin/activate"

# jemalloc + PYTHONMALLOC=malloc make freed pages actually return to the OS, which is
# what keeps the systemd MemoryMax guard meaningful. Skip cleanly if not installed.
JEMALLOC=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
if [ -r "$JEMALLOC" ]; then
    export LD_PRELOAD="$JEMALLOC"
    export PYTHONMALLOC=malloc
else
    echo "[start_server] WARN: jemalloc missing at $JEMALLOC - running without it"
fi

export PYTHONUNBUFFERED=1

python -u "$ROOT/api_server_linux.py" --port "$PORT" &
SERVER_PID=$!

# ngrok is only for public exposure; the server already listens on 0.0.0.0.
if command -v ngrok >/dev/null 2>&1; then
    ngrok http "$PORT"
else
    echo "[start_server] ngrok not installed - serving on 0.0.0.0:$PORT (LAN/Tailscale only)"
    wait "$SERVER_PID"
fi
