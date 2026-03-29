# Cloud Chaos SRE — Copilot Build Instructions

## What you are building

An OpenEnv-compliant reinforcement learning environment called **Cloud Chaos SRE**. It simulates a Site Reliability Engineer managing a virtual data centre of six interdependent microservices under active production incidents. An AI agent must diagnose and fix incidents (memory leaks, traffic spikes, hidden misconfigurations) by taking actions step by step.

This is a hackathon submission. Every file, variable name, endpoint, and behaviour described here is a hard requirement — do not improvise or substitute.

---

## Critical rules — read before writing any code

1. The inference script MUST be named `inference.py` and placed in the project root. Any other name causes automatic disqualification.
2. The LLM client MUST use the `openai` Python package. Do not use the `groq` package directly.
3. The LLM API key MUST be read from `os.environ["OPENAI_API_KEY"]`. Do not use `GROQ_API_KEY`.
4. Three additional env vars MUST be read: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`. Never hardcode these values.
5. `inference.py` MUST complete within 19 minutes (use `signal.alarm(19 * 60)`).
6. The environment server MUST run on port `7860`.
7. All simulation is pure in-memory Python — no Redis, no database, no external services.
8. The machine has 2 vCPU and 8GB RAM. Keep memory usage under 512MB.

---

## Project structure — create exactly this

```
cloud-chaos-sre/
├── main.py                        # FastAPI app — all HTTP endpoints
├── inference.py                   # Inference script — MUST be this exact name
├── openenv.yaml                   # OpenEnv metadata
├── Dockerfile                     # Container definition
├── requirements.txt               # Pinned dependencies
├── README.md                      # Already written — do not modify
│
├── env/
│   ├── __init__.py
│   ├── models.py                  # All Pydantic v2 models
│   ├── environment.py             # Core env class: reset(), step(), state()
│   ├── tasks.py                   # Task registry and task configs
│   ├── graders.py                 # Deterministic grader functions
│   ├── simulator.py               # Virtual data centre simulation engine
│   └── data_generator.py          # Synthetic scenario generator
│
└── scenarios/
    ├── easy/                      # 10 pre-generated JSON scenario files
    ├── medium/                    # 10 pre-generated JSON scenario files
    └── hard/                      # 10 pre-generated JSON scenario files
```

---

## File 1 — env/models.py

Create all Pydantic v2 models with full type annotations and `Field(description=...)` on every field.

```python
from pydantic import BaseModel, Field
from typing import Any
from datetime import datetime
from enum import Enum

class ActionType(str, Enum):
    CHECK_LOGS      = "CHECK_LOGS"
    INSPECT_SERVICE = "INSPECT_SERVICE"
    RESTART_SERVICE = "RESTART_SERVICE"
    SCALE_UP        = "SCALE_UP"
    SCALE_DOWN      = "SCALE_DOWN"
    ROLLBACK        = "ROLLBACK"
    UPDATE_CONFIG   = "UPDATE_CONFIG"
    SILENCE_ALERT   = "SILENCE_ALERT"

VALID_SERVICES = [
    "api-gateway", "auth-service", "user-service",
    "order-service", "db-proxy", "cache-service"
]

class SystemMetrics(BaseModel):
    cpu_pct:    dict[str, float] = Field(description="Per-service CPU usage 0-100")
    mem_pct:    dict[str, float] = Field(description="Per-service memory usage 0-100")
    error_rate: dict[str, float] = Field(description="Per-service HTTP error rate 0.0-1.0")
    latency_ms: dict[str, float] = Field(description="Per-service p99 latency in ms")
    timestamp:  datetime

class LogEntry(BaseModel):
    timestamp:   datetime
    service:     str
    severity:    str   # DEBUG, INFO, WARN, ERROR, CRITICAL
    message:     str

class DeployEvent(BaseModel):
    deploy_id:   str
    timestamp:   datetime
    service:     str
    changes:     dict[str, Any]   # config diffs applied in this deploy

class Alert(BaseModel):
    alert_id:    str
    service:     str
    metric:      str
    threshold:   float
    current:     float
    severity:    str
    triggered_at: datetime
    silenced:    bool = False

class HealthSummary(BaseModel):
    per_service: dict[str, float]  # 0.0-1.0 per service
    overall:     float              # mean of per_service

class Observation(BaseModel):
    step:           int             = Field(description="Current step in episode")
    max_steps:      int             = Field(description="Max steps allowed")
    task_id:        str             = Field(description="Active task: easy|medium|hard")
    metrics:        SystemMetrics
    logs:           list[LogEntry]  = Field(description="Last 10 log entries")
    deploy_history: list[DeployEvent] = Field(description="Last 5 deploy events")
    current_config: dict[str, Any]  = Field(description="Live config key-value pairs")
    service_graph:  dict[str, list[str]] = Field(description="Service dependency map")
    active_alerts:  list[Alert]
    health_summary: HealthSummary

class Action(BaseModel):
    action_type:    ActionType
    target_service: str             = Field(description="Must be one of VALID_SERVICES")
    config_key:     str | None      = Field(default=None, description="Required for UPDATE_CONFIG")
    config_value:   Any | None      = Field(default=None, description="Required for UPDATE_CONFIG")
    reason:         str | None      = Field(default=None, description="Agent reasoning — used in Task 1 grader")

class RewardBreakdown(BaseModel):
    health_delta:      float
    cost_efficiency:   float
    latency_delta:     float
    invalid_penalty:   float

class Reward(BaseModel):
    step_reward:   float
    cumulative:    float
    breakdown:     RewardBreakdown

class EpisodeState(BaseModel):
    task_id:         str
    scenario_id:     str
    step:            int
    done:            bool
    observation:     Observation
    action_history:  list[dict]
    reward_history:  list[float]
    cumulative_reward: float
    grader_score:    float | None = None
```

---

## File 2 — env/simulator.py

The virtual data centre engine. All state is a plain Python dict — no external services.

```python
import copy, random
from datetime import datetime, timezone
from .models import SystemMetrics, LogEntry, DeployEvent, Alert, HealthSummary

SERVICES = ["api-gateway", "auth-service", "user-service", "order-service", "db-proxy", "cache-service"]

SERVICE_GRAPH = {
    "api-gateway":  ["auth-service", "order-service", "user-service"],
    "auth-service": ["db-proxy", "cache-service"],
    "user-service": ["db-proxy", "cache-service"],
    "order-service":["db-proxy"],
    "db-proxy":     [],
    "cache-service":[]
}

HEALTH_THRESHOLDS = {
    "cpu_pct":    70.0,
    "mem_pct":    80.0,
    "error_rate": 0.01,
    "latency_ms": 200.0,
}

class VirtualDataCentre:
    """Pure in-memory simulation of a 6-service data centre."""

    def __init__(self, scenario: dict):
        self.scenario = copy.deepcopy(scenario)
        self.state = copy.deepcopy(scenario["initial_state"])
        self.config = copy.deepcopy(scenario["initial_config"])
        self.replicas = {s: 1 for s in SERVICES}
        self.deploy_history = copy.deepcopy(scenario.get("deploy_history", []))
        self.alerts: list[Alert] = []
        self.logs: list[LogEntry] = []
        self._refresh_alerts()

    def get_metrics(self) -> SystemMetrics:
        return SystemMetrics(
            cpu_pct    = {s: self.state[s]["cpu_pct"]    for s in SERVICES},
            mem_pct    = {s: self.state[s]["mem_pct"]    for s in SERVICES},
            error_rate = {s: self.state[s]["error_rate"] for s in SERVICES},
            latency_ms = {s: self.state[s]["latency_ms"] for s in SERVICES},
            timestamp  = datetime.now(timezone.utc),
        )

    def health_score(self) -> HealthSummary:
        scores = {}
        for s in SERVICES:
            st = self.state[s]
            cpu_score = max(0.0, min(1.0, (HEALTH_THRESHOLDS["cpu_pct"] - st["cpu_pct"]) / 30.0 + 1.0))
            mem_score = max(0.0, min(1.0, (HEALTH_THRESHOLDS["mem_pct"] - st["mem_pct"]) / 20.0 + 1.0))
            err_score = max(0.0, min(1.0, 1.0 - st["error_rate"] / 1.0))
            lat_score = max(0.0, min(1.0, (HEALTH_THRESHOLDS["latency_ms"] - st["latency_ms"]) / 1800.0 + 1.0))
            scores[s] = round((cpu_score + mem_score + err_score + lat_score) / 4.0, 4)
        overall = round(sum(scores.values()) / len(scores), 4)
        return HealthSummary(per_service=scores, overall=overall)

    def is_healthy(self) -> bool:
        for s in SERVICES:
            st = self.state[s]
            if (st["cpu_pct"] >= HEALTH_THRESHOLDS["cpu_pct"] or
                st["mem_pct"] >= HEALTH_THRESHOLDS["mem_pct"] or
                st["error_rate"] >= HEALTH_THRESHOLDS["error_rate"] or
                st["latency_ms"] >= HEALTH_THRESHOLDS["latency_ms"]):
                return False
        return True

    def apply_action(self, action_type: str, target: str, config_key=None, config_value=None) -> dict:
        """Apply an action to the simulation. Returns result info dict."""
        result = {"valid": True, "changed": False, "details": ""}

        if target not in SERVICES:
            result["valid"] = False
            result["details"] = f"Unknown service: {target}"
            return result

        if action_type == "CHECK_LOGS":
            self._add_log(target, "INFO", f"Log fetch requested for {target}")
            result["details"] = f"Retrieved logs for {target}"

        elif action_type == "INSPECT_SERVICE":
            result["details"] = f"Inspected {target}: {self.state[target]}"

        elif action_type == "RESTART_SERVICE":
            svc = self.state[target]
            svc["cpu_pct"]    = max(5.0,  svc["cpu_pct"]    * 0.4)
            svc["mem_pct"]    = max(5.0,  svc["mem_pct"]    * 0.3)
            svc["error_rate"] = max(0.0,  svc["error_rate"] * 0.2)
            svc["latency_ms"] = max(10.0, svc["latency_ms"] * 0.5)
            self._propagate_recovery(target)
            self._add_log(target, "INFO", f"{target} restarted")
            result["changed"] = True

        elif action_type == "SCALE_UP":
            if self.replicas[target] >= 5:
                result["valid"] = False
                result["details"] = f"{target} already at max replicas (5)"
                return result
            self.replicas[target] += 1
            svc = self.state[target]
            svc["cpu_pct"]    = max(5.0,  svc["cpu_pct"]    * 0.7)
            svc["latency_ms"] = max(10.0, svc["latency_ms"] * 0.8)
            result["changed"] = True
            result["details"] = f"{target} scaled to {self.replicas[target]} replicas"

        elif action_type == "SCALE_DOWN":
            if self.replicas[target] <= 1:
                result["valid"] = False
                result["details"] = f"{target} already at minimum replicas (1)"
                return result
            self.replicas[target] -= 1
            svc = self.state[target]
            svc["cpu_pct"]    = min(100.0, svc["cpu_pct"]    * 1.3)
            svc["latency_ms"] = min(5000.0, svc["latency_ms"] * 1.2)
            result["changed"] = True

        elif action_type == "ROLLBACK":
            if len(self.deploy_history) >= 2:
                prev = self.deploy_history[-2]
                for k, v in prev.get("changes", {}).items():
                    if k in self.config:
                        self.config[k] = v
                self._apply_config_effects()
                self._add_log(target, "INFO", f"Rolled back {target} to deploy {prev['deploy_id']}")
                result["changed"] = True
            else:
                result["details"] = "No previous deployment to roll back to"

        elif action_type == "UPDATE_CONFIG":
            if not config_key or config_value is None:
                result["valid"] = False
                result["details"] = "UPDATE_CONFIG requires config_key and config_value"
                return result
            self.config[config_key] = config_value
            self._apply_config_effects()
            self._add_log(target, "INFO", f"Config updated: {config_key}={config_value}")
            result["changed"] = True
            result["details"] = f"Set {config_key}={config_value}"

        elif action_type == "SILENCE_ALERT":
            for a in self.alerts:
                if a.service == target and not a.silenced:
                    a.silenced = True
            result["details"] = f"Silenced alerts for {target}"

        self._refresh_alerts()
        return result

    def _propagate_recovery(self, service: str):
        """When a service recovers, partially improve its dependents."""
        for svc, deps in SERVICE_GRAPH.items():
            if service in deps:
                self.state[svc]["error_rate"] = max(0.0, self.state[svc]["error_rate"] * 0.7)
                self.state[svc]["latency_ms"] = max(10.0, self.state[svc]["latency_ms"] * 0.85)

    def _apply_config_effects(self):
        """Apply config changes to service state — key mechanic for Task 3."""
        db_timeout = self.config.get("db_timeout", 5000)
        if db_timeout < 500:
            # Misconfigured — all db-dependent services degrade
            for svc in ["order-service", "user-service", "auth-service"]:
                self.state[svc]["error_rate"] = min(1.0, self.state[svc]["error_rate"] + 0.3)
                self.state[svc]["latency_ms"] = min(5000.0, self.state[svc]["latency_ms"] * 2.0)
        else:
            # Correct config — db-dependent services recover
            for svc in ["order-service", "user-service", "auth-service"]:
                self.state[svc]["error_rate"] = max(0.0, self.state[svc]["error_rate"] - 0.25)
                self.state[svc]["latency_ms"] = max(50.0, self.state[svc]["latency_ms"] * 0.4)

    def _add_log(self, service: str, severity: str, message: str):
        self.logs.append(LogEntry(
            timestamp=datetime.now(timezone.utc),
            service=service, severity=severity, message=message
        ))
        self.logs = self.logs[-10:]  # keep last 10 only

    def _refresh_alerts(self):
        self.alerts = []
        for svc in SERVICES:
            st = self.state[svc]
            if st["cpu_pct"] >= HEALTH_THRESHOLDS["cpu_pct"]:
                self.alerts.append(Alert(alert_id=f"{svc}-cpu", service=svc, metric="cpu_pct",
                    threshold=HEALTH_THRESHOLDS["cpu_pct"], current=st["cpu_pct"],
                    severity="WARN" if st["cpu_pct"] < 90 else "CRITICAL",
                    triggered_at=datetime.now(timezone.utc)))
            if st["error_rate"] >= HEALTH_THRESHOLDS["error_rate"]:
                self.alerts.append(Alert(alert_id=f"{svc}-err", service=svc, metric="error_rate",
                    threshold=HEALTH_THRESHOLDS["error_rate"], current=st["error_rate"],
                    severity="CRITICAL" if st["error_rate"] > 0.1 else "WARN",
                    triggered_at=datetime.now(timezone.utc)))
```

---

## File 3 — env/data_generator.py

Generates 30 scenario JSON files (10 per task). Run standalone: `python env/data_generator.py`.

Each scenario JSON must have this structure:
```json
{
  "scenario_id": "easy-001",
  "task_id": "easy",
  "initial_state": {
    "api-gateway":   {"cpu_pct": 45.0, "mem_pct": 55.0, "error_rate": 0.001, "latency_ms": 120.0},
    "auth-service":  {"cpu_pct": 40.0, "mem_pct": 50.0, "error_rate": 0.001, "latency_ms": 100.0},
    "user-service":  {"cpu_pct": 72.0, "mem_pct": 68.0, "error_rate": 0.15,  "latency_ms": 850.0},
    "order-service": {"cpu_pct": 70.0, "mem_pct": 65.0, "error_rate": 0.12,  "latency_ms": 780.0},
    "db-proxy":      {"cpu_pct": 92.0, "mem_pct": 88.0, "error_rate": 0.45,  "latency_ms": 2100.0},
    "cache-service": {"cpu_pct": 30.0, "mem_pct": 35.0, "error_rate": 0.001, "latency_ms": 15.0}
  },
  "initial_config": {
    "db_timeout": 5000,
    "pool_size": 10,
    "replica_count": 1,
    "ttl": 300
  },
  "deploy_history": [
    {"deploy_id": "d001", "timestamp": "2025-01-01T00:00:00Z", "service": "db-proxy", "changes": {"pool_size": 10}},
    {"deploy_id": "d002", "timestamp": "2025-01-01T01:00:00Z", "service": "db-proxy", "changes": {"pool_size": 5}}
  ],
  "ground_truth": {
    "root_cause_service": "db-proxy",
    "correct_action": "RESTART_SERVICE",
    "correct_config_key": null,
    "correct_config_value": null
  }
}
```

For Task 3 (hard) scenarios, set:
- `initial_config.db_timeout = 100` (the misconfiguration)
- `ground_truth.correct_action = "UPDATE_CONFIG"`
- `ground_truth.correct_config_key = "db_timeout"`
- `ground_truth.correct_config_value = 5000`
- In `deploy_history`, the second-to-last deploy must show `db_timeout` changing from `5000` to `100`

Generate 10 variations per task by randomising which service is the root cause (easy), which metrics are worst (medium), and small variations in the config values (hard).

---

## File 4 — env/graders.py

Three deterministic grader functions. Same input ALWAYS returns same output.

```python
from .models import EpisodeState

def grade_easy(state: EpisodeState) -> tuple[float, dict]:
    """
    Task 1 — The Detective.
    Score breakdown:
      - Correct root service in any action's reason field: 0.40
      - Zero false-positive restarts (healthy services restarted): 0.30
      - System health restored (mean health score >= 0.85): 0.20
      - Efficiency bonus (done in <= 8 steps): 0.10
    """
    scenario = state.observation  # load ground truth from scenario
    root_cause = "db-proxy"  # always db-proxy for easy task

    score = 0.0
    breakdown = {"root_identified": 0.0, "no_false_positives": 0.0, "health_restored": 0.0, "efficiency": 0.0}

    # Check if reason field in any action mentions root cause
    for action in state.action_history:
        reason = (action.get("reason") or "").lower()
        if root_cause.replace("-", "") in reason or root_cause in reason:
            breakdown["root_identified"] = 0.40
            break

    # Check for false positive restarts (restarting healthy services)
    false_positives = sum(
        1 for a in state.action_history
        if a.get("action_type") == "RESTART_SERVICE"
        and a.get("target_service") != root_cause
    )
    breakdown["no_false_positives"] = max(0.0, 0.30 - (false_positives * 0.10))

    # Health score at end of episode
    final_health = state.observation.health_summary.overall
    breakdown["health_restored"] = round(min(0.20, final_health * 0.20), 4)

    # Efficiency bonus
    if state.step <= 8:
        breakdown["efficiency"] = 0.10

    score = sum(breakdown.values())
    return round(min(1.0, score), 4), breakdown


def grade_medium(state: EpisodeState) -> tuple[float, dict]:
    """
    Task 2 — The First Responder.
    Score = mean of 4 independent metric scores (each 0.0–1.0).
    Each metric scored linearly from worst-case to threshold.
    """
    from .simulator import HEALTH_THRESHOLDS, SERVICES

    metrics = state.observation.metrics
    thresholds = HEALTH_THRESHOLDS
    worst_cases = {"cpu_pct": 100.0, "mem_pct": 100.0, "error_rate": 1.0, "latency_ms": 2000.0}

    metric_scores = {}
    for metric in ["cpu_pct", "mem_pct", "error_rate", "latency_ms"]:
        values = getattr(metrics, metric).values()
        worst_val = max(values)
        threshold = thresholds[metric]
        worst = worst_cases[metric]
        if worst_val <= threshold:
            metric_scores[metric] = 1.0
        else:
            metric_scores[metric] = round(max(0.0, 1.0 - (worst_val - threshold) / (worst - threshold)), 4)

    score = round(sum(metric_scores.values()) / 4.0, 4)
    return score, metric_scores


def grade_hard(state: EpisodeState) -> tuple[float, dict]:
    """
    Task 3 — The Architect.
    Score breakdown:
      - Correct config key identified (UPDATE_CONFIG with db_timeout): 0.30
      - Correct config value applied (5000): 0.40
      - System health restored (overall >= 0.80): 0.10
      - Efficiency (found fix in <= 10 steps): 0.20
    """
    CORRECT_KEY   = "db_timeout"
    CORRECT_VALUE = 5000

    score = 0.0
    breakdown = {"correct_key": 0.0, "correct_value": 0.0, "health_restored": 0.0, "efficiency": 0.0}

    for action in state.action_history:
        if action.get("action_type") == "UPDATE_CONFIG":
            if action.get("config_key") == CORRECT_KEY:
                breakdown["correct_key"] = 0.30
                val = action.get("config_value")
                try:
                    if int(val) == CORRECT_VALUE:
                        breakdown["correct_value"] = 0.40
                except (TypeError, ValueError):
                    pass
            break  # only first UPDATE_CONFIG counts

    final_health = state.observation.health_summary.overall
    breakdown["health_restored"] = round(min(0.10, final_health * 0.10), 4)

    if state.step <= 10 and breakdown["correct_value"] > 0:
        breakdown["efficiency"] = 0.20

    score = sum(breakdown.values())
    return round(min(1.0, score), 4), breakdown


GRADERS = {
    "easy":   grade_easy,
    "medium": grade_medium,
    "hard":   grade_hard,
}
```

---

## File 5 — env/environment.py

The core environment class.

```python
import json, random
from pathlib import Path
from .models import (Observation, Action, Reward, RewardBreakdown,
                     EpisodeState, HealthSummary)
from .simulator import VirtualDataCentre, SERVICE_GRAPH, SERVICES
from .graders import GRADERS

SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"

class SREEnvironment:
    def __init__(self):
        self._state: EpisodeState | None = None
        self._vdc: VirtualDataCentre | None = None
        self._prev_health: float = 0.0

    def reset(self, task_id: str, scenario_id: str | None = None) -> Observation:
        scenarios = list((SCENARIOS_DIR / task_id).glob("*.json"))
        if not scenarios:
            raise ValueError(f"No scenarios found for task: {task_id}")

        if scenario_id:
            path = SCENARIOS_DIR / task_id / f"{scenario_id}.json"
        else:
            path = random.choice(scenarios)

        scenario = json.loads(path.read_text())
        self._vdc = VirtualDataCentre(scenario)
        obs = self._build_observation(task_id, 0, scenario["scenario_id"])
        self._prev_health = obs.health_summary.overall

        self._state = EpisodeState(
            task_id=task_id,
            scenario_id=scenario["scenario_id"],
            step=0,
            done=False,
            observation=obs,
            action_history=[],
            reward_history=[],
            cumulative_reward=0.0,
        )
        return obs

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        if self._state is None or self._state.done:
            raise RuntimeError("Call reset() before step()")

        max_steps = {"easy": 15, "medium": 15, "hard": 20}[self._state.task_id]
        self._state.step += 1

        # Apply action
        result = self._vdc.apply_action(
            action.action_type.value,
            action.target_service,
            action.config_key,
            action.config_value,
        )

        # Build new observation
        obs = self._build_observation(self._state.task_id, self._state.step, self._state.scenario_id)

        # Compute step reward
        new_health  = obs.health_summary.overall
        health_delta = new_health - self._prev_health
        self._prev_health = new_health

        cost_efficiency  = -0.05 if action.action_type.value == "SCALE_UP" else 0.0
        latency_vals     = list(obs.metrics.latency_ms.values())
        prev_latency_avg = 0.0  # simplified
        latency_penalty  = 0.0 if result["valid"] else 0.0
        invalid_penalty  = 0.0 if result["valid"] else 0.10

        raw_reward = (
            health_delta     * 0.50
          + cost_efficiency  * 0.20
          - latency_penalty  * 0.20
          - invalid_penalty  * 0.10
        )
        step_reward = round(max(-1.0, min(1.0, raw_reward)), 4)

        self._state.cumulative_reward += step_reward
        self._state.reward_history.append(step_reward)
        self._state.action_history.append({
            "step":           self._state.step,
            "action_type":    action.action_type.value,
            "target_service": action.target_service,
            "config_key":     action.config_key,
            "config_value":   action.config_value,
            "reason":         action.reason,
            "valid":          result["valid"],
        })
        self._state.observation = obs

        done = self._vdc.is_healthy() or self._state.step >= max_steps
        self._state.done = done

        reward = Reward(
            step_reward=step_reward,
            cumulative=round(self._state.cumulative_reward, 4),
            breakdown=RewardBreakdown(
                health_delta=round(health_delta * 0.50, 4),
                cost_efficiency=round(cost_efficiency * 0.20, 4),
                latency_delta=round(-latency_penalty * 0.20, 4),
                invalid_penalty=round(-invalid_penalty * 0.10, 4),
            )
        )

        info = {
            "reward_breakdown": reward.breakdown.model_dump(),
            "health_scores": obs.health_summary.model_dump(),
            "step": self._state.step,
            "action_valid": result["valid"],
            "action_details": result.get("details", ""),
        }
        return obs, reward, done, info

    def state(self) -> EpisodeState:
        if self._state is None:
            raise RuntimeError("Call reset() first")
        return self._state

    def grade(self) -> tuple[float, dict]:
        if self._state is None:
            raise RuntimeError("Call reset() and run an episode first")
        grader = GRADERS[self._state.task_id]
        return grader(self._state)

    def _build_observation(self, task_id: str, step: int, scenario_id: str) -> Observation:
        max_steps = {"easy": 15, "medium": 15, "hard": 20}[task_id]
        return Observation(
            step=step,
            max_steps=max_steps,
            task_id=task_id,
            metrics=self._vdc.get_metrics(),
            logs=list(self._vdc.logs),
            deploy_history=[
                {"deploy_id": d["deploy_id"], "timestamp": d["timestamp"],
                 "service": d.get("service",""), "changes": d.get("changes",{})}
                for d in self._vdc.deploy_history[-5:]
            ],
            current_config=dict(self._vdc.config),
            service_graph=SERVICE_GRAPH,
            active_alerts=list(self._vdc.alerts),
            health_summary=self._vdc.health_score(),
        )
```

---

## File 6 — main.py

FastAPI app. All endpoints required by the hackathon spec.

```python
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from env.environment import SREEnvironment
from env.models import Action, EpisodeState
from env.data_generator import generate_all_scenarios
from pathlib import Path

env = SREEnvironment()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-generate scenarios if missing
    if not Path("scenarios/easy").exists():
        generate_all_scenarios()
    yield

app = FastAPI(
    title="Cloud Chaos SRE",
    description="OpenEnv environment simulating SRE incident response",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    return {"status": "ok", "env": "cloud-chaos-sre", "version": "1.0.0"}

@app.post("/reset")
def reset(body: dict):
    task_id     = body.get("task_id", "easy")
    scenario_id = body.get("scenario_id", None)
    obs = env.reset(task_id=task_id, scenario_id=scenario_id)
    return obs.model_dump()

@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step(action)
    return {"observation": obs.model_dump(), "reward": reward.model_dump(),
            "done": done, "info": info}

@app.get("/state")
def state():
    return env.state().model_dump()

@app.get("/tasks")
def tasks():
    from env.models import Action
    return {
        "tasks": [
            {"id": "easy",   "name": "The Detective",       "difficulty": "easy",
             "description": "Identify the root-cause microservice in a cascading failure",
             "max_steps": 15, "action_schema": Action.model_json_schema()},
            {"id": "medium", "name": "The First Responder",  "difficulty": "medium",
             "description": "Restore all system health metrics during a traffic spike",
             "max_steps": 15, "action_schema": Action.model_json_schema()},
            {"id": "hard",   "name": "The Architect",        "difficulty": "hard",
             "description": "Diagnose and fix a hidden database timeout misconfiguration",
             "max_steps": 20, "action_schema": Action.model_json_schema()},
        ]
    }

@app.post("/grader")
def grader(state: EpisodeState):
    # Inject provided state into env for grading
    env._state = state
    score, breakdown = env.grade()
    return {"task_id": state.task_id, "score": score, "breakdown": breakdown}

@app.post("/baseline")
def baseline():
    """Trigger inference.py inline and return scores."""
    import subprocess, json, time
    start = time.time()
    result = subprocess.run(
        ["python", "inference.py", "--output-json"],
        capture_output=True, text=True, timeout=19*60
    )
    elapsed = round(time.time() - start, 1)
    try:
        scores = json.loads(result.stdout)
    except Exception:
        scores = {"error": result.stderr or "inference.py failed"}
    scores["total_time_s"] = elapsed
    return scores
```

---

## File 7 — inference.py  ← CRITICAL: exact filename, root directory

```python
"""
inference.py — OpenEnv hackathon inference script.
MUST be named inference.py. MUST be in root directory.
Reads: OPENAI_API_KEY, API_BASE_URL, MODEL_NAME, HF_TOKEN
"""
import os, sys, json, signal, time, argparse
import requests
from openai import OpenAI

# --- Config from environment variables (NEVER hardcode) ---
API_BASE_URL   = os.environ.get("API_BASE_URL",   "https://api.groq.com/openai/v1")
MODEL_NAME     = os.environ.get("MODEL_NAME",     "llama-3.1-70b-versatile")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
HF_TOKEN       = os.environ.get("HF_TOKEN",       "")

ENV_BASE_URL   = "http://localhost:7860"
TEMPERATURE    = 0.0
MAX_TOKENS     = 512
SEED           = 42
TASKS          = ["easy", "medium", "hard"]
FALLBACK_ACTION = json.dumps({
    "action_type": "CHECK_LOGS",
    "target_service": "api-gateway",
    "reason": "Fallback: checking gateway logs"
})

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer responding to a production incident.
You will receive system observations step by step. At each step you must:
1. Analyse the metrics, logs, deploy history, and current config provided
2. Reason about the most likely root cause
3. Choose exactly one action from: CHECK_LOGS, INSPECT_SERVICE, RESTART_SERVICE, SCALE_UP, SCALE_DOWN, ROLLBACK, UPDATE_CONFIG, SILENCE_ALERT
4. Target one of: api-gateway, auth-service, user-service, order-service, db-proxy, cache-service
5. Provide a clear reason for your choice

Respond ONLY in valid JSON. No other text. Schema:
{"action_type": "...", "target_service": "...", "config_key": null, "config_value": null, "reason": "..."}"""

# --- Hard timeout: must finish within 20 min ---
def _timeout_handler(signum, frame):
    print("TIMEOUT: 19-minute limit reached. Saving partial scores.")
    raise TimeoutError()

signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm(19 * 60)

client = OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)

def call_env(method: str, path: str, body: dict | None = None) -> dict:
    url = f"{ENV_BASE_URL}{path}"
    resp = requests.request(method, url, json=body, timeout=30)
    resp.raise_for_status()
    return resp.json()

def parse_action(text: str) -> dict:
    """Parse LLM response to action dict. Returns fallback on failure."""
    import re
    text = text.strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return json.loads(FALLBACK_ACTION)

def format_observation(obs: dict, step: int) -> str:
    metrics = obs.get("metrics", {})
    config  = obs.get("current_config", {})
    alerts  = [f"{a['service']}:{a['metric']}={a['current']:.2f}" for a in obs.get("active_alerts", [])]
    logs    = [f"[{l['severity']}] {l['service']}: {l['message']}" for l in obs.get("logs", [])[-5:]]
    deploys = [f"Deploy {d['deploy_id']}: {d['changes']}" for d in obs.get("deploy_history", [])]
    health  = obs.get("health_summary", {}).get("per_service", {})

    return f"""Step {step} | Task: {obs.get('task_id')} | Max steps: {obs.get('max_steps')}

HEALTH SCORES: {json.dumps(health, indent=None)}

METRICS:
  CPU%:       {json.dumps(metrics.get('cpu_pct', {}))}
  Memory%:    {json.dumps(metrics.get('mem_pct', {}))}
  Error rate: {json.dumps(metrics.get('error_rate', {}))}
  Latency ms: {json.dumps(metrics.get('latency_ms', {}))}

ACTIVE ALERTS: {', '.join(alerts) or 'none'}
CURRENT CONFIG: {json.dumps(config)}
RECENT LOGS: {chr(10).join(logs) or 'none'}
DEPLOY HISTORY: {chr(10).join(deploys) or 'none'}"""

def run_task(task_id: str) -> dict:
    print(f"\n{'='*50}")
    print(f"Running task: {task_id}")
    print(f"{'='*50}")

    obs = call_env("POST", "/reset", {"task_id": task_id})
    max_steps = obs.get("max_steps", 15)
    done = False
    step = 0
    history = []

    while not done and step < max_steps:
        step += 1
        user_content = format_observation(obs, step)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user",   "content": user_content},
        ]

        # Retry up to 3 times on parse failure
        response_text = FALLBACK_ACTION
        for attempt in range(3):
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    seed=SEED,
                    stream=False,
                )
                response_text = completion.choices[0].message.content or FALLBACK_ACTION
                break
            except Exception as exc:
                print(f"  Attempt {attempt+1} failed: {exc}")

        action_dict = parse_action(response_text)
        print(f"Step {step}: {action_dict.get('action_type')} → {action_dict.get('target_service')}")

        result = call_env("POST", "/step", action_dict)
        obs    = result.get("observation", obs)
        reward = result.get("reward", {}).get("step_reward", 0.0)
        done   = result.get("done", False)

        history.append(f"Step {step}: {action_dict.get('action_type')} → {action_dict.get('target_service')} | reward {reward:+.3f}")

        if done:
            print("Episode complete.")
            break

    # Get grader score
    state = call_env("GET", "/state")
    grader_result = call_env("POST", "/grader", state)
    score = grader_result.get("score", 0.0)

    print(f"Final score: {score:.4f}")
    return {"task_id": task_id, "score": score, "steps": step,
            "breakdown": grader_result.get("breakdown", {})}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", action="store_true", help="Output JSON only (for /baseline endpoint)")
    args = parser.parse_args()

    scores = {}
    start = time.time()

    try:
        for task_id in TASKS:
            result = run_task(task_id)
            scores[task_id] = result
    except TimeoutError:
        print("Inference timed out — partial scores saved.")
    finally:
        signal.alarm(0)

    mean_score = round(sum(r["score"] for r in scores.values()) / max(len(scores), 1), 4)
    output = {
        "model":        MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "seed":         SEED,
        "scores":       scores,
        "mean_score":   mean_score,
        "total_time_s": round(time.time() - start, 1),
    }

    # Save to file
    with open("baseline_scores.json", "w") as f:
        json.dump(output, f, indent=2)

    if args.output_json:
        print(json.dumps(output))
    else:
        print("\n" + "="*50)
        print("BASELINE RESULTS")
        print("="*50)
        for task_id, r in scores.items():
            print(f"  {task_id:8s}: {r['score']:.4f}  ({r['steps']} steps)")
        print(f"  {'MEAN':8s}: {mean_score:.4f}")
        print(f"\nSaved to baseline_scores.json")

if __name__ == "__main__":
    main()
```

---

## File 8 — openenv.yaml

```yaml
name: cloud-chaos-sre
version: 1.0.0
description: >
  An OpenEnv environment simulating a Site Reliability Engineer managing
  a virtual data centre under active production incidents. The agent must
  diagnose and remediate memory leaks, traffic spikes, and hidden
  misconfigurations across six interdependent microservices.
author:
  name: Cloud Chaos SRE Team
  email: your@email.com

tasks:
  - id: easy
    name: The Detective
    difficulty: easy
    description: Identify the root-cause microservice in a cascading failure
    max_steps: 15
  - id: medium
    name: The First Responder
    difficulty: medium
    description: Restore all system health metrics during an active traffic spike
    max_steps: 15
  - id: hard
    name: The Architect
    difficulty: hard
    description: Diagnose and fix a hidden database timeout misconfiguration
    max_steps: 20

observation_space:
  type: object
  properties:
    step:           { type: integer }
    metrics:        { type: object, description: "Per-service CPU, memory, error rate, latency" }
    logs:           { type: array,  description: "Last 10 structured log entries" }
    deploy_history: { type: array,  description: "Last 5 deployment events with config diffs" }
    current_config: { type: object, description: "Live configuration key-value pairs" }
    service_graph:  { type: object, description: "Microservice dependency map" }
    active_alerts:  { type: array,  description: "Currently triggered alarms" }
    health_summary: { type: object, description: "Per-service and overall health 0.0-1.0" }

action_space:
  type: object
  required: [action_type, target_service]
  properties:
    action_type:
      type: string
      enum: [CHECK_LOGS, INSPECT_SERVICE, RESTART_SERVICE, SCALE_UP, SCALE_DOWN, ROLLBACK, UPDATE_CONFIG, SILENCE_ALERT]
    target_service:
      type: string
      enum: [api-gateway, auth-service, user-service, order-service, db-proxy, cache-service]
    config_key:
      type: string
      description: Required for UPDATE_CONFIG actions
    config_value:
      description: Required for UPDATE_CONFIG actions
    reason:
      type: string
      description: Agent reasoning — used in Task 1 grader

reward:
  min: -1.0
  max: 1.0
  description: >
    Non-sparse per-step reward. Components: health improvement delta (×0.50),
    cost efficiency (×0.20), latency worsening penalty (×0.20),
    invalid action penalty (×0.10).

episode:
  max_steps: 20
  termination_conditions:
    - All health metrics below threshold (success)
    - Max steps reached (partial credit)

tags:
  - openenv
  - sre
  - reinforcement-learning
  - infrastructure
  - nlp
  - agent-evaluation
  - operations
```

---

## File 9 — Dockerfile

```dockerfile
FROM python:3.11-slim

# Designed for 2 vCPU / 8GB RAM
# All simulation is in-memory Python — no external services required
# Typical peak memory usage: < 512MB

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY . .

# Pre-generate scenarios at build time
RUN python env/data_generator.py

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
```

---

## File 10 — requirements.txt

```
fastapi==0.115.0
uvicorn==0.30.6
pydantic==2.8.2
openai==1.50.0
requests==2.32.3
python-multipart==0.0.12
httpx==0.27.2
```

---

## Validation checklist — run before submitting

After writing all files, verify every item:

```bash
# 1. Scenarios generated
python env/data_generator.py
ls scenarios/easy/ scenarios/medium/ scenarios/hard/   # must show JSON files

# 2. Server starts cleanly
uvicorn main:app --host 0.0.0.0 --port 7860 &

# 3. Health check
curl http://localhost:7860/health
# Expected: {"status":"ok","env":"cloud-chaos-sre","version":"1.0.0"}

# 4. Tasks endpoint
curl http://localhost:7860/tasks
# Expected: JSON with 3 tasks, each with action_schema

# 5. Full episode test — easy task
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id":"easy"}'
curl -X POST http://localhost:7860/step  -H "Content-Type: application/json" \
  -d '{"action_type":"CHECK_LOGS","target_service":"db-proxy","reason":"checking root cause"}'
curl http://localhost:7860/state
curl -X POST http://localhost:7860/grader -H "Content-Type: application/json" -d "$(curl -s http://localhost:7860/state)"
# Expected: {"task_id":"easy","score": <float 0-1>, "breakdown": {...}}

# 6. OpenEnv validation
pip install openenv
openenv validate openenv.yaml

# 7. Docker build and run
docker build -t cloud-chaos-sre .
docker run -p 7860:7860 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e API_BASE_URL=$API_BASE_URL \
  -e MODEL_NAME=$MODEL_NAME \
  -e HF_TOKEN=$HF_TOKEN \
  cloud-chaos-sre

# 8. inference.py runs and produces scores
export OPENAI_API_KEY=gsk_your_groq_key
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.1-70b-versatile
export HF_TOKEN=hf_your_token
python inference.py
# Expected: prints scores for easy/medium/hard and saves baseline_scores.json
# Must complete in under 19 minutes

# 9. Grader scores are in range
# Check that easy score ≈ 0.65-0.80, medium ≈ 0.45-0.65, hard ≈ 0.25-0.40
# If all three return the same score → DISQUALIFICATION — fix your graders
```

---

## Common mistakes to avoid

- Do NOT name the inference script anything other than `inference.py`
- Do NOT read from `GROQ_API_KEY` — use `OPENAI_API_KEY`
- Do NOT hardcode the model name or API URL — always read from env vars
- Do NOT forget `signal.alarm(19 * 60)` in inference.py
- Do NOT return the same grader score for all three tasks
- Do NOT make the reward function return 0.0 at every non-terminal step
- Do NOT use `position: fixed` or any web UI in the server — it breaks HF Spaces iframe
- Do NOT import heavy ML libraries (torch, transformers) — exceeds 8GB RAM limit
- Do NOT run the environment server inside inference.py — they are separate processes