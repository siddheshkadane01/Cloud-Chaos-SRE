#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-site-reliability-server:local}"
CONTAINER_NAME="${CONTAINER_NAME:-site-reliability-server-local-check}"
HOST_PORT="${HOST_PORT:-7861}"

PYTHON_BIN="python"
OPENENV_BIN="openenv"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi
if [[ -x ".venv/bin/openenv" ]]; then
  OPENENV_BIN=".venv/bin/openenv"
fi

log() {
  printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"
}

cleanup() {
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

log "Running tests"
"$PYTHON_BIN" -m pytest -q

log "Running OpenEnv validation"
if [[ -x "$OPENENV_BIN" ]]; then
  "$OPENENV_BIN" validate
else
  log "openenv CLI not found; running local contract validator"
  "$PYTHON_BIN" scripts/validate_openenv_contract.py
fi

log "Building Docker image: $IMAGE_NAME"
docker build -t "$IMAGE_NAME" .

log "Starting container on localhost:$HOST_PORT"
docker run -d --name "$CONTAINER_NAME" -p "$HOST_PORT":7860 "$IMAGE_NAME" >/dev/null

log "Waiting for /health"
for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:${HOST_PORT}/health" >/dev/null; then
    break
  fi
  sleep 1
done

curl -fsS "http://127.0.0.1:${HOST_PORT}/health" >/dev/null
curl -fsS -X POST "http://127.0.0.1:${HOST_PORT}/reset" -H "Content-Type: application/json" -d '{}' >/dev/null

log "Running baseline inference"
if [[ ( -z "${OPENAI_API_KEY:-}" && -z "${HF_TOKEN:-}" ) || -z "${API_BASE_URL:-}" || -z "${MODEL_NAME:-}" ]]; then
  echo "Missing required env vars. Set OPENAI_API_KEY (preferred) or HF_TOKEN, plus API_BASE_URL and MODEL_NAME" >&2
  exit 2
fi

"$PYTHON_BIN" inference.py

if [[ ! -f baseline_scores.json ]]; then
  echo "baseline_scores.json not found after inference run" >&2
  exit 3
fi

log "All local validation checks passed"
