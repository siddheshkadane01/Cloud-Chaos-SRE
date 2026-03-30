import json
import random
from pathlib import Path

from .graders import GRADERS
from .models import Action, DeployEvent, EpisodeState, Observation, Reward, RewardBreakdown
from .simulator import SERVICE_GRAPH, VirtualDataCentre

SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"


class SREEnvironment:
    def __init__(self):
        self._state: EpisodeState | None = None
        self._vdc: VirtualDataCentre | None = None
        self._prev_health: float = 0.0
        self._prev_mean_latency: float = 0.0

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
        self._prev_mean_latency = sum(obs.metrics.latency_ms.values()) / max(len(obs.metrics.latency_ms), 1)

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

        max_steps = {"easy": 15, "medium": 15, "hard": 20}[self._state.task_id]
        self._state.step += 1

        result = self._vdc.apply_action(
            action.action_type.value,
            action.target_service,
            action.config_key,
            action.config_value,
        )

        obs = self._build_observation(self._state.task_id, self._state.step, self._state.scenario_id)

        new_health = obs.health_summary.overall
        health_delta = new_health - self._prev_health
        self._prev_health = new_health

        new_mean_latency = sum(obs.metrics.latency_ms.values()) / max(len(obs.metrics.latency_ms), 1)
        latency_delta_norm = max(-1.0, min(1.0, (self._prev_mean_latency - new_mean_latency) / 400.0))
        self._prev_mean_latency = new_mean_latency

        cost_efficiency = -0.05 if action.action_type.value == "SCALE_UP" else 0.0
        repeated_action_penalty = 0.0
        if self._state.action_history:
            last = self._state.action_history[-1]
            if (
                last.get("action_type") == action.action_type.value
                and last.get("target_service") == action.target_service
            ):
                repeated_action_penalty = 0.05
        invalid_penalty = 0.0 if result["valid"] else 0.25

        raw_reward = (
            health_delta * 0.55
            + cost_efficiency * 0.10
            + latency_delta_norm * 0.30
            - invalid_penalty * 0.30
            - repeated_action_penalty * 0.10
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
            }
        )
        self._state.observation = obs

        done = self._vdc.is_healthy() or self._state.step >= max_steps
        self._state.done = done

        reward = Reward(
            step_reward=step_reward,
            cumulative=round(self._state.cumulative_reward, 4),
            breakdown=RewardBreakdown(
                health_delta=round(health_delta * 0.55, 4),
                cost_efficiency=round(cost_efficiency * 0.10, 4),
                latency_delta=round(latency_delta_norm * 0.30, 4),
                invalid_penalty=round(-(invalid_penalty * 0.30 + repeated_action_penalty * 0.10), 4),
            ),
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
        _ = scenario_id
        if self._vdc is None:
            raise RuntimeError("Environment simulator is not initialized")

        max_steps = {"easy": 15, "medium": 15, "hard": 20}[task_id]
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
        )
