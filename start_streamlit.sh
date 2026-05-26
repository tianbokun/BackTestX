#!/usr/bin/env bash
# Streamlit 启动脚本 - 自动寻找可用端口
set -e

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_PORT=8501
MAX_PORT=8599

PORT=$DEFAULT_PORT
while true; do
    if ! ss -tlnp "sport = :$PORT" 2>/dev/null | grep -q ":$PORT"; then
        echo "[start_streamlit] Port $PORT is available"
        break
    fi
    echo "[start_streamlit] Port $PORT is in use, trying $((PORT + 1))"
    PORT=$((PORT + 1))
    if [ "$PORT" -gt "$MAX_PORT" ]; then
        echo "[start_streamlit] No available ports found in range $DEFAULT_PORT-$MAX_PORT"
        exit 1
    fi
done

cd "$APP_DIR"

PYTHON=/home/tianbo/.conda/envs/stock_analysis/bin/python3
exec "$PYTHON" -m streamlit run app.py \
    --server.port="$PORT" \
    --server.headless=true \
    --server.address=0.0.0.0 \
    --browser.gatherUsageStats=false
