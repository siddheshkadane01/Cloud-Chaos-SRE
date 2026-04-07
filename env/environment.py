import json
import os
import random
from pathlib import Path
from typing import Literal

from .graders import GRADERS
from .models import Action, DeployEvent, EpisodeState, IncidentContext, Observation, Reward, RewardBreakdown
from .simulator import SERVICE_GRAPH, VirtualDataCentre

SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"

_MAX_STEPS = {"easy": 15, "medium": 15, "hard": 20, "expert": 25}
_VALIDATOR_MIN_SCORE = 0.0001
_VALIDATOR_MAX_SCORE = 0.9999
TaskId = Literal["easy", "medium", "hard", "expert"]


class SREEnvironment:
    def __init__(
        self,
        *,
        deterministic: bool | None = None,
        evaluation_mode: bool | None = None,
        default_seed: int = 42,
    ):
        if deterministic is None:
            deterministic = os.getenv("OPENENV_DETERMINISTIC", "1") != "0"
        if evaluation_mode is None:
            evaluation_mode = os.getenv("OPENENV_EVALUATION_MODE", "1") != "0"

        self.deterministic = deterministic
        self.evaluation_mode = evaluation_mode
        self.default_seed = default_seed
        self._state: EpisodeState | None = None
        self._vdc: VirtualDataCentre | None = None
        self._prev_health: float = 0.0
        self._prev_mean_latency: float = 0.0
        self._prev_service_health: dict[str, float] = {}
        self._last_seed: int = default_seed

    def reset(
        self,
        task_id: TaskId,
        scenario_id: str | None = None,
        *,
        seed: int | None = None,
        deterministic: bool | None = None,
        evaluation_mode: bool | None = None,
    ) -> Observation:
        scenarios = sorted((SCENARIOS_DIR / task_id).glob("*.json"))
        if not scenarios:
            raise ValueError(f"No scenarios found for task: {task_id}")

        deterministic = self.deterministic if deterministic is None else deterministic
        evaluation_mode = self.evaluation_mode if evaluation_mode is None else evaluation_mode
        seed = self.default_seed if seed is None else seed
        self._last_seed = seed

        if scenario_id:
            path = SCENARIOS_DIR / task_id / f"{scenario_id}.json"
        elif deterministic:
            path = scenarios[0]
        else:
            path = random.Random(seed).choice(scenarios)

        scenario = json.loads(path.read_text())
        scenario_seed = self._scenario_seed(task_id, scenario["scenario_id"], seed)
        self._vdc = VirtualDataCentre(
            scenario,
            enable_drift=not evaluation_mode,
            seed=scenario_seed,
        )
        obs = self._build_observation(task_id, 0, scenario["scenario_id"])
        self._prev_health = obs.health_summary.overall
        self._prev_mean_latency = sum(obs.metrics.latency_ms.values()) / max(len(obs.metrics.latency_ms), 1)
        self._prev_service_health = dict(obs.health_summary.per_service)

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
        if self._vdc is None:
            raise RuntimeError("Environment simulator is not initialized")

        max_steps = _MAX_STEPS[self._state.task_id]
        self._state.step += 1

        result = self._vdc.apply_action(
            action.action_type.value,
            action.target_service,
            action.config_key,
            action.config_value,
        )

        obs = self._build_observation(self._state.task_id, self._state.step, self._state.scenario_id)

        # --- Reward components ---
        new_health = obs.health_summary.overall
        health_delta = new_health - self._prev_health
        self._prev_health = new_health

        new_mean_latency = sum(obs.metrics.latency_ms.values()) / max(len(obs.metrics.latency_ms), 1)
        latency_delta_norm = max(-1.0, min(1.0, (self._prev_mean_latency - new_mean_latency) / 400.0))
        self._prev_mean_latency = new_mean_latency

        important_services = self._important_services()
        service_deltas = [
            obs.health_summary.per_service[service] - self._prev_service_health.get(service, 0.0)
            for service in important_services
            if service in obs.health_summary.per_service
        ]
        task_progress = (sum(service_deltas) / len(service_deltas)) if service_deltas else 0.0
        self._prev_service_health = dict(obs.health_summary.per_service)

        # SCALE_UP incurs a cloud-cost penalty (×0.10 weight)
        if action.action_type.value == "SCALE_UP":
            cost_efficiency = -0.04
        elif action.action_type.value == "DRAIN_TRAFFIC":
            cost_efficiency = -0.02
        else:
            cost_efficiency = 0.0

        # Repeated identical action on same service incurs a small penalty
        repeated_action_penalty = 0.0
        if self._state.action_history:
            last = self._state.action_history[-1]
            if (
                last.get("action_type") == action.action_type.value
                and last.get("target_service") == action.target_service
            ):
                repeated_action_penalty = 0.05

        invalid_penalty = 0.0 if result["valid"] else 0.25
        risk_penalty = result.get("risk_penalty", 0.0)

        # SILENCE_ALERT cleanup bonus: awarded when agent silences a fixed service
        silence_bonus = 0.02 if result.get("silence_bonus") else 0.0

        # Weights aligned across openenv.yaml and code:
        #   health_delta       × 0.40
        #   latency_delta      × 0.20
        #   task_progress      × 0.25
        #   cost_efficiency    × 0.05   (negative for SCALE_UP)
        #   invalid_penalty    × 0.05   (negative)
        #   repeat_penalty     × 0.05   (negative, shared bucket with invalid)
        #   risk_penalty       direct subtraction
        #   silence_bonus      flat +0.02
        raw_reward = (
            health_delta * 0.40
            + latency_delta_norm * 0.20
            + task_progress * 0.25
            + cost_efficiency * 0.05
            - invalid_penalty * 0.05
            - repeated_action_penalty * 0.05
            - risk_penalty
            + silence_bonus
        )
        step_reward = round(max(-1.0, min(1.0, raw_reward)), 4)

        self._state.cumulative_reward += step_reward
        self._state.reward_history.append(step_reward)
        self._state.action_history.append(
            {
                "step": self._state.step,
                "action_type": action.action_type.value,
                "target_service": action.target_service,
                "config_key": action.config_key,
                "config_value": action.config_value,
                "reason": action.reason,
                "valid": result["valid"],
                "silence_bonus": result.get("silence_bonus", False),
                "step_reward": step_reward,
                "overall_health": new_health,
                "critical_services_healthy": self._critical_services_healthy(obs),
                "open_alerts": sum(1 for alert in obs.active_alerts if not alert.silenced),
                "action_details": result.get("details", ""),
            }
        )
        self._state.observation = obs

        done = self._task_complete(obs) or self._state.step >= max_steps
        self._state.done = done

        reward = Reward(
            step_reward=step_reward,
            cumulative=round(self._state.cumulative_reward, 4),
            breakdown=RewardBreakdown(
                health_delta=round(health_delta * 0.40, 4),
                task_progress=round(task_progress * 0.25, 4),
                cost_efficiency=round(cost_efficiency * 0.05, 4),
                latency_delta=round(latency_delta_norm * 0.20, 4),
                invalid_penalty=round(
                    -(invalid_penalty * 0.05 + repeated_action_penalty * 0.05), 4
                ),
                risk_penalty=round(-risk_penalty, 4),
            ),
        )

        info = {
            "reward_breakdown": reward.breakdown.model_dump(),
            "health_scores": obs.health_summary.model_dump(),
            "step": self._state.step,
            "action_valid": result["valid"],
            "action_details": result.get("details", ""),
            "last_action_error": None if result["valid"] else result.get("details", "invalid_action"),
            "silence_bonus": result.get("silence_bonus", False),
            "task_complete": done,
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
        raw_score, breakdown = grader(self._state)
        safe_score = max(_VALIDATOR_MIN_SCORE, min(_VALIDATOR_MAX_SCORE, float(raw_score)))
        return round(safe_score, 4), breakdown

    def _build_observation(self, task_id: TaskId, step: int, scenario_id: str) -> Observation:
        _ = scenario_id
        if self._vdc is None:
            raise RuntimeError("Environment simulator is not initialized")

        max_steps = _MAX_STEPS[task_id]
        return Observation(
            step=step,
            max_steps=max_steps,
            task_id=task_id,
            metrics=self._vdc.get_metrics(),
            logs=list(self._vdc.logs),
            deploy_history=[
                DeployEvent(
                    deploy_id=event["deploy_id"],
                    timestamp=event["timestamp"],
                    service=event.get("service", ""),
                    changes=event.get("changes", {}),
                )
                for event in self._vdc.deploy_history[-5:]
            ],
            current_config=dict(self._vdc.config),
            service_graph=SERVICE_GRAPH,
            active_alerts=list(self._vdc.alerts),
            health_summary=self._vdc.health_score(),
            incident_context=IncidentContext(**self._vdc.scenario["incident_context"]),
        )

    def _scenario_seed(self, task_id: str, scenario_id: str, seed: int) -> int:
        token = f"{task_id}:{scenario_id}:{seed}"
        return sum(ord(ch) for ch in token)

    def _important_services(self) -> list[str]:
        if self._vdc is None:
            return []
        ground_truth = self._vdc.scenario.get("ground_truth", {})
        services = [ground_truth.get("root_cause_service")]
        secondary = ground_truth.get("secondary_cause_service")
        if secondary:
            services.append(secondary)
        return [service for service in services if service]

    def _critical_services_healthy(self, obs: Observation) -> bool:
        important_services = self._important_services()
        if not important_services:
            return obs.health_summary.overall >= 0.95
        return all(
            obs.health_summary.per_service.get(service, 0.0) >= 0.95
            for service in important_services
        )

    def _task_complete(self, obs: Observation) -> bool:
        task_id = obs.task_id
        open_alerts = [
            alert
            for alert in obs.active_alerts
            if not alert.silenced and alert.service in set(self._important_services())
        ]

        if task_id == "easy":
            return self._critical_services_healthy(obs) and obs.health_summary.overall >= 0.95

        if task_id == "medium":
            return (
                self._critical_services_healthy(obs)
                and obs.health_summary.overall >= 0.94
                and not open_alerts
            )

        if task_id == "hard":
            upstream = ("auth-service", "user-service", "order-service")
            upstream_healthy = all(
                obs.health_summary.per_service.get(service, 0.0) >= 0.92 for service in upstream
            )
            return (
                self._critical_services_healthy(obs)
                and upstream_healthy
                and obs.health_summary.overall >= 0.94
                and not open_alerts
            )

        return (
            self._critical_services_healthy(obs)
            and obs.health_summary.overall >= 0.94
            and not open_alerts
        )
