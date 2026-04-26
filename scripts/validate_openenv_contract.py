#!/usr/bin/env python3
"""Local OpenEnv contract validator for submission readiness.

This script validates the repository's OpenEnv-facing contract without relying
on an external `openenv` CLI binary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OPENENV_YAML = REPO_ROOT / "openenv.yaml"

# Make script runnable from any working directory.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient

from env.environment import SREEnvironment
from main import app


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _parse_openenv_yaml(path: Path) -> dict[str, str]:
    _require(path.exists(), f"Missing required file: {path}")

    parsed: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        # Restrict parsing to top-level mappings only.
        if raw_line.startswith((" ", "\t")):
            continue
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            parsed[key] = value
    return parsed


def validate_yaml() -> None:
    parsed = _parse_openenv_yaml(OPENENV_YAML)

    required = {
        "spec_version": "1",
        "type": "space",
        "runtime": "fastapi",
        "app": "server.app:app",
        "port": "7860",
    }
    for key, expected in required.items():
        _require(key in parsed, f"openenv.yaml missing key: {key}")
        _require(parsed[key] == expected, f"openenv.yaml {key} must be '{expected}', found '{parsed[key]}'")


def validate_environment_class() -> None:
    base_names = {base.__name__ for base in SREEnvironment.__mro__}
    _require("Env" in base_names, "SREEnvironment must inherit from openenv.env.Env")


def validate_api_contract() -> None:
    with TestClient(app) as client:
        health = client.get("/health")
        _require(health.status_code == 200, "/health must return 200")

        reset = client.post("/reset", json={})
        _require(reset.status_code == 200, "/reset must return 200")
        reset_payload = reset.json()
        _require(isinstance(reset_payload, dict), "/reset must return a JSON object")
        _require("task_id" in reset_payload, "/reset payload must include task_id")

        step = client.post(
            "/step",
            json={"action_type": "CHECK_LOGS", "target_service": "api-gateway"},
        )
        _require(step.status_code == 200, "/step must return 200 for a valid action")
        step_payload = step.json()
        _require(isinstance(step_payload, dict), "/step must return a JSON object")
        _require("observation" in step_payload, "/step payload missing observation")
        _require("reward" in step_payload, "/step payload missing reward")
        _require("done" in step_payload, "/step payload missing done")

        state = client.get("/state")
        _require(state.status_code == 200, "/state must return 200")
        state_payload = state.json()
        _require(isinstance(state_payload, dict), "/state must return a JSON object")
        _require("task_id" in state_payload, "/state payload must include task_id")


def main() -> int:
    try:
        validate_yaml()
        validate_environment_class()
        validate_api_contract()
    except Exception as exc:  # noqa: BLE001
        print(f"[openenv-local-validate] FAIL: {exc}")
        return 1

    print("[openenv-local-validate] PASS")
    print(
        json.dumps(
            {
                "yaml": "ok",
                "environment_inheritance": "ok",
                "api_contract": "ok",
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())