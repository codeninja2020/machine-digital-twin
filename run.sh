#!/usr/bin/env bash
# run.sh - Starts all Machine Digital Twin services concurrently

set -e

# Ensure clean termination of all child background processes on CTRL+C / exit
cleanup() {
    echo ""
    echo "Stopping all digital twin services..."
    kill $(jobs -p) 2>/dev/null || true
    wait 2>/dev/null || true
    echo "All services stopped."
}
trap cleanup EXIT INT TERM

# Activate virtual environment if present
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "=========================================="
echo " Starting Machine Digital Twin Stack"
echo "=========================================="

echo "[1/3] Starting Inference Server (port 8000)..."
python inference_server.py &
INF_PID=$!

sleep 1

echo "[2/3] Starting Mock IoT Sensor (port 8765)..."
python mock_sensor.py &
SENSOR_PID=$!

sleep 1

echo "[3/3] Starting Digital Twin Engine (port 8080)..."
echo ">> Dashboard: http://localhost:8080"
echo ">> Press Ctrl+C at any time to stop all services."
echo "=========================================="
python twin.py
