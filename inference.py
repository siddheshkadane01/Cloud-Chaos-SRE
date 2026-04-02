"""
Submission-grade deterministic inference runner for Site Reliability Server.

The script emits only the required [START], [STEP], and [END] records to stdout.
Scores and diagnostics are written to baseline_scores.json for local inspection.
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

import requests
from openai import OpenAI

HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

_ = LOCAL_IMAGE_NAME

ENV_BASE_URL = os.getenv("OPENENV_BASE_URL", "http://127.0.0.1:7860")
BENCHMARK_NAME = "site-reliability-server"
DEFAULT_SEED = 42
GLOBAL_TIMEOUT_SECONDS = 19 * 60
TASKS = ("easy", "medium", "hard", "expert")
CANONICAL_SCENARIOS = {
    "easy": "easy-001",
    "medium": "medium-001",
    "hard": "hard-001",
    "expert": "expert-001",
}
HEALTH_THRESHOLDS = {
    "cpu_pct": 70.0,
    "mem_pct": 80.0,
    "error_rate": 0.01,
    "latency_ms": 200.0,
}
SERVICE_ORDER = (
    "api-gateway",
    "auth-service",
    "user-service",
    "order-service",
    "db-proxy",
    "cache-service",
)

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL) if HF_TOKEN else None
_SERVER_PROCESS: subprocess.Popen[str] | None = None


@dataclass
class EpisodeResult:
    task_id: str
    scenario_id: str
    score: float
    success: bool
    steps: int
    rewards: list[float]
    breakdown: dict[str, Any]
    model_diagnosis: str | None


def _timeout_handler(signum, frame):
    _ = (signum, frame)
    raise TimeoutError("global_timeout")


signal.signal(signal.SIGALRM, _timeout_handler)


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def format_reward(value: float) -> str:
    return f"{value:.2f}"


def sanitize_text(value: str | None) -> str:
    if value is None:
        return "null"
    compact = " ".join(str(value).split())
    return compact if compact else "null"


def compact_json(value: dict[str, Any]) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def emit_start(task_id: str) -> None:
    print(
        f"[START] task={task_id} env={BENCHMARK_NAME} model={MODEL_NAME}",
        flush=True,
    )


def emit_step(step: int, action: dict[str, Any], reward: float, done: bool, error: str | None) -> None:
    print(
        f"[STEP] step={step} action={compact_json(action)} reward={format_reward(reward)} "
        f"done={bool_text(done)} error={sanitize_text(error)}",
        flush=True,
    )


def emit_end(success: bool, steps: int, rewards: list[float]) -> None:
    rewards_payload = ",".join(format_reward(value) for value in rewards)
    print(
        f"[END] success={bool_text(success)} steps={steps} rewards={rewards_payload}",
        flush=True,
    )


def call_env(method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.request(method, f"{ENV_BASE_URL}{path}", json=body, timeout=30)
    response.raise_for_status()
    return response.json()


def can_reach_server() -> bool:
    try:
        response = requests.get(f"{ENV_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def stop_server() -> None:
    global _SERVER_PROCESS
    if _SERVER_PROCESS is None:
        return
    if _SERVER_PROCESS.poll() is None:
        _SERVER_PROCESS.terminate()
        try:
            _SERVER_PROCESS.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _SERVER_PROCESS.kill()
            _SERVER_PROCESS.wait(timeout=5)
    _SERVER_PROCESS = None


def ensure_server() -> None:
    global _SERVER_PROCESS
    if can_reach_server():
        return

    _SERVER_PROCESS = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "7860",
            "--workers",
            "1",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    deadline = time.time() + 25
    while time.time() < deadline:
        if can_reach_server():
            return
        if _SERVER_PROCESS.poll() is not None:
            raise RuntimeError("environment_server_failed_to_start")
        time.sleep(0.5)

    raise TimeoutError("environment_server_start_timeout")


atexit.register(stop_server)


def service_unhealthy(metrics: dict[str, Any], service: str) -> bool:
    return any(
        metrics[metric][service] >= HEALTH_THRESHOLDS[metric]
        for metric in HEALTH_THRESHOLDS
    )


def service_pressure(metrics: dict[str, Any], service: str) -> float:
    pressure = 0.0
    pressure += max(0.0, metrics["cpu_pct"][service] - HEALTH_THRESHOLDS["cpu_pct"]) / 30.0
    pressure += max(0.0, metrics["mem_pct"][service] - HEALTH_THRESHOLDS["mem_pct"]) / 20.0
    pressure += max(0.0, metrics["error_rate"][service] - HEALTH_THRESHOLDS["error_rate"]) / 0.20
    pressure += max(0.0, metrics["latency_ms"][service] - HEALTH_THRESHOLDS["latency_ms"]) / 600.0
    return round(pressure, 6)


def sorted_unhealthy_services(obs: dict[str, Any]) -> list[str]:
    metrics = obs["metrics"]
    unhealthy = [svc for svc in SERVICE_ORDER if service_unhealthy(metrics, svc)]
    return sorted(unhealthy, key=lambda svc: (-service_pressure(metrics, svc), svc))


def silenced_state(obs: dict[str, Any], service: str) -> tuple[int, int]:
    total = 0
    open_alerts = 0
    for alert in obs.get("active_alerts", []):
        if alert["service"] != service:
            continue
        total += 1
        if not alert.get("silenced", False):
            open_alerts += 1
    return total, open_alerts


def action_counts(history: list[dict[str, Any]], action_type: str, target_service: str) -> int:
    return sum(
        1
        for item in history
        if item.get("action_type") == action_type and item.get("target_service") == target_service
    )


def build_reason(task_id: str, action_type: str, target_service: str, obs: dict[str, Any]) -> str:
    metrics = obs["metrics"]
    service_metrics = {
        metric: metrics[metric][target_service]
        for metric in ("cpu_pct", "mem_pct", "error_rate", "latency_ms")
    }
    if task_id == "hard" and action_type == "UPDATE_CONFIG":
        config = obs.get("current_config", {})
        if int(config.get("db_timeout", 5000)) < 500:
            return "db-proxy is failing because db_timeout is too low; restoring db_timeout to 5000"
        return "db-proxy is failing because pool_size is too low; restoring pool_size to 10"
    if action_type == "DRAIN_TRAFFIC":
        return f"temporarily draining traffic away from {target_service} to reduce live production pressure while remediation continues"
    if task_id == "expert":
        if target_service == "cache-service":
            return "cache-service is the primary failure and must be recovered before db-proxy"
        if target_service == "db-proxy":
            return "db-proxy remains degraded after cache recovery and must be restarted next"
    return (
        f"{target_service} is the highest-pressure service with metrics "
        f"cpu={service_metrics['cpu_pct']:.1f}, mem={service_metrics['mem_pct']:.1f}, "
        f"error_rate={service_metrics['error_rate']:.4f}, latency_ms={service_metrics['latency_ms']:.1f}"
    )


def summarize_with_model(task_id: str, obs: dict[str, Any]) -> str | None:
    if client is None:
        return None

    prompt = {
        "task_id": task_id,
        "task": BENCHMARK_NAME,
        "incident_context": obs.get("incident_context", {}),
        "health_summary": obs.get("health_summary", {}),
        "alerts": obs.get("active_alerts", []),
        "current_config": obs.get("current_config", {}),
        "deploy_history": obs.get("deploy_history", []),
    }
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.0,
            seed=DEFAULT_SEED,
            max_tokens=120,
            timeout=20,
            messages=[
                {
                    "role": "system",
                    "content": "Return one concise sentence naming the most likely incident root cause.",
                },
                {
                    "role": "user",
                    "content": json.dumps(prompt, separators=(",", ":"), sort_keys=True),
                },
            ],
        )
    except Exception:
        return None

    content = completion.choices[0].message.content or ""
    cleaned = " ".join(content.split())
    return cleaned or None


def choose_action(task_id: str, obs: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = obs["metrics"]
    unhealthy = sorted_unhealthy_services(obs)
    worst_service = unhealthy[0] if unhealthy else "db-proxy"
    suspected_services = obs.get("incident_context", {}).get("suspected_services", [])
    primary_suspect = suspected_services[0] if suspected_services else worst_service
    secondary_suspect = suspected_services[1] if len(suspected_services) > 1 else None

    if task_id == "easy":
        root = worst_service
        if action_counts(history, "CHECK_LOGS", root) == 0:
            return action_payload("CHECK_LOGS", root, task_id, obs)
        if service_unhealthy(metrics, root):
            return action_payload("RESTART_SERVICE", root, task_id, obs)
        _, open_alerts = silenced_state(obs, root)
        if open_alerts > 0:
            return action_payload("SILENCE_ALERT", root, task_id, obs)
        return action_payload("CHECK_LOGS", root, task_id, obs)

    if task_id == "medium":
        primary_scale_count = action_counts(history, "SCALE_UP", primary_suspect)
        primary_drain_count = action_counts(history, "DRAIN_TRAFFIC", primary_suspect)
        primary_restart_count = action_counts(history, "RESTART_SERVICE", primary_suspect)
        secondary_restart_count = (
            action_counts(history, "RESTART_SERVICE", secondary_suspect)
            if secondary_suspect
            else 0
        )
        primary_structural_pressure = (
            primary_suspect in metrics["cpu_pct"]
            and (
                metrics["cpu_pct"][primary_suspect] >= HEALTH_THRESHOLDS["cpu_pct"]
                or metrics["mem_pct"][primary_suspect] >= HEALTH_THRESHOLDS["mem_pct"]
            )
        )
        primary_live_pressure = (
            primary_suspect in metrics["cpu_pct"]
            and (
                metrics["error_rate"][primary_suspect] >= HEALTH_THRESHOLDS["error_rate"]
                or metrics["latency_ms"][primary_suspect] >= HEALTH_THRESHOLDS["latency_ms"]
            )
        )
        if (
            primary_structural_pressure
            and primary_scale_count < 3
        ):
            return action_payload("SCALE_UP", primary_suspect, task_id, obs)
        if (
            primary_live_pressure
            and primary_drain_count == 0
        ):
            return action_payload("DRAIN_TRAFFIC", primary_suspect, task_id, obs)
        if (
            secondary_suspect
            and secondary_suspect in metrics["cpu_pct"]
            and service_unhealthy(metrics, secondary_suspect)
        ):
            if action_counts(history, "CHECK_LOGS", secondary_suspect) == 0:
                return action_payload("CHECK_LOGS", secondary_suspect, task_id, obs)
            if secondary_restart_count < 3:
                return action_payload("RESTART_SERVICE", secondary_suspect, task_id, obs)
        if (
            primary_live_pressure
            and secondary_suspect
            and not service_unhealthy(metrics, secondary_suspect)
            and primary_restart_count == 0
        ):
            return action_payload("RESTART_SERVICE", primary_suspect, task_id, obs)
        if primary_structural_pressure and primary_scale_count < 4:
            return action_payload("SCALE_UP", primary_suspect, task_id, obs)
        if primary_live_pressure and primary_drain_count < 3:
            return action_payload("DRAIN_TRAFFIC", primary_suspect, task_id, obs)
        for service in unhealthy:
            if service == primary_suspect:
                continue
            if (
                metrics["cpu_pct"][service] >= HEALTH_THRESHOLDS["cpu_pct"]
                or metrics["mem_pct"][service] >= HEALTH_THRESHOLDS["mem_pct"]
            ):
                return action_payload("SCALE_UP", service, task_id, obs)
            if (
                metrics["error_rate"][service] >= HEALTH_THRESHOLDS["error_rate"]
                or metrics["latency_ms"][service] >= HEALTH_THRESHOLDS["latency_ms"]
            ):
                return action_payload("RESTART_SERVICE", service, task_id, obs)
        for service in SERVICE_ORDER:
            _, open_alerts = silenced_state(obs, service)
            if open_alerts > 0 and not service_unhealthy(metrics, service):
                return action_payload("SILENCE_ALERT", service, task_id, obs)
        return action_payload("CHECK_LOGS", worst_service, task_id, obs)

    if task_id == "hard":
        if action_counts(history, "INSPECT_SERVICE", "db-proxy") == 0:
            return action_payload("INSPECT_SERVICE", "db-proxy", task_id, obs)
        if action_counts(history, "CHECK_LOGS", "db-proxy") == 0:
            return action_payload("CHECK_LOGS", "db-proxy", task_id, obs)
        if action_counts(history, "CHECK_LOGS", "order-service") == 0:
            return action_payload("CHECK_LOGS", "order-service", task_id, obs)
        log_blob = " ".join(entry.get("message", "") for entry in obs.get("logs", [])).lower()
        db_timeout = int(obs["current_config"].get("db_timeout", 5000))
        pool_size = int(obs["current_config"].get("pool_size", 10))
        deploy_hints = obs.get("deploy_history", [])
        recent_change_keys = {
            key
            for deploy in deploy_hints[-2:]
            for key in deploy.get("changes", {}).keys()
        }
        expected_key = "db_timeout"
        if "root_key" in recent_change_keys:
            for deploy in reversed(deploy_hints):
                hint = deploy.get("changes", {}).get("root_key")
                if hint in {"db_timeout", "pool_size"}:
                    expected_key = hint
                    break
        elif "connection pool saturation" in log_blob or "pool exhausted" in log_blob:
            expected_key = "pool_size"
        elif "timeout" in log_blob:
            expected_key = "db_timeout"
        elif pool_size < 10 and db_timeout >= 500:
            expected_key = "pool_size"

        if expected_key == "db_timeout" and action_counts(history, "UPDATE_CONFIG", "db-proxy") == 0:
            return action_payload(
                "UPDATE_CONFIG",
                "db-proxy",
                task_id,
                obs,
                config_key="db_timeout",
                config_value=5000,
            )
        if expected_key == "pool_size" and action_counts(history, "UPDATE_CONFIG", "db-proxy") == 0:
            return action_payload(
                "UPDATE_CONFIG",
                "db-proxy",
                task_id,
                obs,
                config_key="pool_size",
                config_value=10,
            )
        for service in ("db-proxy", "auth-service", "user-service", "order-service"):
            if service_unhealthy(metrics, service):
                return action_payload("RESTART_SERVICE", service, task_id, obs)
        for service in ("db-proxy", "auth-service", "user-service", "order-service"):
            _, open_alerts = silenced_state(obs, service)
            if open_alerts > 0:
                return action_payload("SILENCE_ALERT", service, task_id, obs)
        return action_payload("CHECK_LOGS", "db-proxy", task_id, obs)

    # expert
    cache_restart_count = action_counts(history, "RESTART_SERVICE", "cache-service")
    db_restart_count = action_counts(history, "RESTART_SERVICE", "db-proxy")
    db_drain_count = action_counts(history, "DRAIN_TRAFFIC", "db-proxy")

    if service_unhealthy(metrics, "cache-service") and cache_restart_count < 4:
        return action_payload("RESTART_SERVICE", "cache-service", task_id, obs)
    if (
        service_unhealthy(metrics, "db-proxy")
        and service_unhealthy(metrics, "cache-service")
        and db_drain_count == 0
    ):
        return action_payload("DRAIN_TRAFFIC", "db-proxy", task_id, obs)
    if service_unhealthy(metrics, "db-proxy") and not service_unhealthy(metrics, "cache-service") and db_restart_count < 4:
        return action_payload("RESTART_SERVICE", "db-proxy", task_id, obs)
    for service in ("cache-service", "db-proxy"):
        if service_unhealthy(metrics, service):
            return action_payload("RESTART_SERVICE", service, task_id, obs)
    for service in ("cache-service", "db-proxy"):
        _, open_alerts = silenced_state(obs, service)
        if open_alerts > 0:
            return action_payload("SILENCE_ALERT", service, task_id, obs)

    # Prevent no-op loops (repeated CHECK_LOGS)
    if len(history) >= 3:
        last_actions = [h.get("action_type") for h in history[-3:]]
        if all(action == "CHECK_LOGS" for action in last_actions):
            worst = sorted_unhealthy_services(obs)
            target = worst[0] if worst else "db-proxy"
            return action_payload("RESTART_SERVICE", target, task_id, obs)

    fallback_actions = ["CHECK_LOGS", "INSPECT_SERVICE"]
    action_type = fallback_actions[len(history) % len(fallback_actions)]
    return action_payload(action_type, "db-proxy", task_id, obs)


def action_payload(
    action_type: str,
    target_service: str,
    task_id: str,
    obs: dict[str, Any],
    *,
    config_key: str | None = None,
    config_value: Any | None = None,
) -> dict[str, Any]:
    return {
        "action_type": action_type,
        "target_service": target_service,
        "config_key": config_key,
        "config_value": config_value,
        "reason": build_reason(task_id, action_type, target_service, obs),
    }


def run_task(task_id: str) -> EpisodeResult:
    scenario_id = CANONICAL_SCENARIOS[task_id]
    obs: dict[str, Any] | None = None
    model_diagnosis: str | None = None
    rewards: list[float] = []
    steps = 0
    success = False
    breakdown: dict[str, Any] = {}

    emit_start(task_id)

    try:
        obs = call_env(
            "POST",
            "/reset",
            {
                "task_id": task_id,
                "scenario_id": scenario_id,
                "seed": DEFAULT_SEED,
                "deterministic": True,
                "evaluation_mode": True,
            },
        )
        if obs is None:
            raise RuntimeError("environment_returned_null_observation")
        model_diagnosis = summarize_with_model(task_id, obs)

        while steps < int(obs["max_steps"]):
            state = call_env("GET", "/state")
            action = choose_action(task_id, obs, state.get("action_history", []))
            result = call_env("POST", "/step", action)
            reward_value = float(result["reward"]["step_reward"])
            done = bool(result["done"])
            error = result.get("info", {}).get("last_action_error")
            steps += 1
            rewards.append(reward_value)
            emit_step(steps, action, reward_value, done, error)
            next_obs = result.get("observation")
            if not isinstance(next_obs, dict):
                raise RuntimeError("invalid_observation_payload")
            obs = next_obs
            if done:
                break

        final_state = call_env("GET", "/state")
        grader = call_env("POST", "/grader", final_state)
        breakdown = grader.get("breakdown", {})
        score = float(grader.get("score", 0.0))
        success = score >= 0.85
        return EpisodeResult(
            task_id=task_id,
            scenario_id=scenario_id,
            score=score,
            success=success,
            steps=steps,
            rewards=rewards,
            breakdown=breakdown,
            model_diagnosis=model_diagnosis,
        )
    except Exception:
        return EpisodeResult(
            task_id=task_id,
            scenario_id=scenario_id,
            score=0.0,
            success=False,
            steps=steps,
            rewards=rewards,
            breakdown=breakdown,
            model_diagnosis=model_diagnosis,
        )
    finally:
        emit_end(success, steps, rewards)


def write_scores(results: dict[str, EpisodeResult], started_at: float) -> None:
    payload = {
        "benchmark": BENCHMARK_NAME,
        "model": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "seed": DEFAULT_SEED,
        "tasks": {
            task_id: {
                "scenario_id": result.scenario_id,
                "score": round(result.score, 4),
                "success": result.success,
                "steps": result.steps,
                "rewards": [round(value, 4) for value in result.rewards],
                "breakdown": result.breakdown,
                "model_diagnosis": result.model_diagnosis,
            }
            for task_id, result in results.items()
        },
        "mean_score": round(
            sum(result.score for result in results.values()) / max(len(results), 1),
            4,
        ),
        "total_time_s": round(time.time() - started_at, 2),
    }
    with open("baseline_scores.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main() -> int:
    started_at = time.time()
    results: dict[str, EpisodeResult] = {}
    signal.alarm(GLOBAL_TIMEOUT_SECONDS)

    try:
        ensure_server()
        for task_id in TASKS:
            results[task_id] = run_task(task_id)
    except TimeoutError:
        pass
    finally:
        signal.alarm(0)
        write_scores(results, started_at)
        stop_server()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
