#!/usr/bin/env bash
set -euo pipefail

HR_PORT="${HR_PORT:-8001}"
IT_PORT="${IT_PORT:-8002}"
ROOT_DIR="$(cd "$(dirname "$0")/../../" && pwd)"

echo "Starting HR on http://localhost:${HR_PORT} and IT on http://localhost:${IT_PORT}"

uvicorn assistants.hr_assistant.main:app --host 0.0.0.0 --port ${HR_PORT} &
HR_PID=$!
uvicorn assistants.it_assistant.main:app --host 0.0.0.0 --port ${IT_PORT} &
IT_PID=$!

trap "echo Shutting down; kill ${HR_PID} ${IT_PID};" INT TERM
wait ${HR_PID} ${IT_PID}