import json
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Literal

from .graders import GRADERS
from .models import Action, DeployEvent, EpisodeState, IncidentContext, Observation, Reward, RewardBreakdown
from .simulator import SERVICE_GRAPH, VirtualDataCentre

SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"

_MAX_STEPS = {"easy": 15, "medium": 15, "hard": 20, "expert": 25, "enterprise": 25}
_VALIDATOR_MIN_SCORE = 0.0001
_VALIDATOR_MAX_SCORE = 0.9999
TaskId = Literal["easy", "medium", "hard", "expert", "enterprise"]

INFRA_ACTIONS = {
    "CHECK_LOGS",
    "INSPECT_SERVICE",
    "DRAIN_TRAFFIC",
    "RESTART_SERVICE",
    "SCALE_UP",
    "SCALE_DOWN",
    "ROLLBACK",
    "UPDATE_CONFIG",
    "SILENCE_ALERT",
}

ENTERPRISE_ACTIONS = {
    "ACKNOWLEDGE_PAGERDUTY",
    "SEND_SLACK_MESSAGE",
    "RESOLVE_PAGERDUTY",
}

_DEFAULT_ROLE_PERMISSIONS: dict[str, set[str]] = {
    "incident_commander": {"ACKNOWLEDGE_PAGERDUTY", "RESOLVE_PAGERDUTY", "CHECK_LOGS", "INSPECT_SERVICE"},
    "investigator": {"CHECK_LOGS", "INSPECT_SERVICE"},
    "remediator": {"RESTART_SERVICE", "SCALE_UP", "SCALE_DOWN", "DRAIN_TRAFFIC", "ROLLBACK", "UPDATE_CONFIG", "SILENCE_ALERT"},
    "comms_officer": {"SEND_SLACK_MESSAGE", "CHECK_LOGS", "INSPECT_SERVICE"},
}


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
        self._apps_state: dict = {}
        self._protocol_status: dict[str, bool] = {
            "is_acknowledged": False,
            "is_team_notified": False,
            "is_resolved": False,
        }
        self._enterprise_reward_flags: dict[str, bool] = {
            "ack_bonus_awarded": False,
            "notify_bonus_awarded": False,
            "completion_bonus_awarded": False,
        }
        self._mode: str = "single_agent"
        self._multi_agent_state: dict = {}
        self._multi_agent_kpis: dict[str, float] = {}
        self._multi_agent_permissions: dict[str, set[str]] = {}

    def reset(
        self,
        task_id: TaskId,
        scenario_id: str | None = None,
        *,
        seed: int | None = None,
        deterministic: bool | None = None,
        evaluation_mode: bool | None = None,
        mode: str = "single_agent",
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
        self._mode = mode
        self._init_multi_agent_state(scenario)
        self._init_enterprise_state()
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
            protocol_status=dict(self._protocol_status),
            mode=self._mode,
            multi_agent_kpis=dict(self._multi_agent_kpis),
        )
        return obs

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        if self._state is None or self._state.done:
            raise RuntimeError("Call reset() before step()")
        if self._vdc is None:
            raise RuntimeError("Environment simulator is not initialized")

        max_steps = _MAX_STEPS[self._state.task_id]
        self._state.step += 1

        action_type = action.action_type.value
        incident_id = action.incident_id or action.params.get("incident_id")
        channel_name = action.channel_name or action.params.get("channel_name")
        message_text = action.message_text or action.params.get("message_text")
        actor_role = action.actor_role
        handoff_to = action.handoff_to

        protocol_penalty = 0.0
        protocol_progress_bonus = 0.0
        completion_bonus = 0.0
        coordination_bonus = 0.0
        coordination_penalty = 0.0

        # Multi-agent protocol gate — runs before any action dispatch
        ma_result = {"valid": True, "details": "", "bonus": 0.0, "penalty": 0.0}
        if self._mode == "multi_agent":
            ma_result = self._apply_multi_agent_protocol(
                action_type=action_type,
                actor_role=actor_role,
                handoff_to=handoff_to,
                target_service=action.target_service,
            )
            coordination_bonus += ma_result["bonus"]
            coordination_penalty += ma_result["penalty"]

        if self._enterprise_enabled() and action_type in INFRA_ACTIONS:
            if not self._protocol_status.get("is_acknowledged", False):
                protocol_penalty = 0.2

        if not ma_result["valid"]:
            result: dict = {
                "valid": False,
                "changed": False,
                "details": ma_result["details"] or "multi_agent_protocol_violation",
            }
        elif action_type in ENTERPRISE_ACTIONS:
            result = self._apply_enterprise_action(
                action_type=action_type,
                incident_id=incident_id,
                channel_name=channel_name,
                message_text=message_text,
            )
            if result.get("ack_bonus") and not self._enterprise_reward_flags["ack_bonus_awarded"]:
                protocol_progress_bonus += 0.1
                self._enterprise_reward_flags["ack_bonus_awarded"] = True
            if result.get("notify_bonus") and not self._enterprise_reward_flags["notify_bonus_awarded"]:
                protocol_progress_bonus += 0.1
                self._enterprise_reward_flags["notify_bonus_awarded"] = True
        else:
            result = self._vdc.apply_action(
                action_type,
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
        if action_type == "SCALE_UP":
            cost_efficiency = -0.04
        elif action_type == "DRAIN_TRAFFIC":
            cost_efficiency = -0.02
        else:
            cost_efficiency = 0.0

        # Repeated identical action on same service incurs a small penalty
        repeated_action_penalty = 0.0
        if self._state.action_history:
            last = self._state.action_history[-1]
            if (
                last.get("action_type") == action_type
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
            - protocol_penalty
            + protocol_progress_bonus
            + coordination_bonus
            - coordination_penalty
        )

        if (
            self._enterprise_enabled()
            and action_type == "RESOLVE_PAGERDUTY"
            and result.get("valid")
            and self._enterprise_health_complete(obs)
            and not self._enterprise_reward_flags["completion_bonus_awarded"]
        ):
            completion_bonus = 0.5
            self._enterprise_reward_flags["completion_bonus_awarded"] = True

        raw_reward += completion_bonus
        step_reward = round(max(-1.0, min(1.0, raw_reward)), 4)

        self._state.cumulative_reward += step_reward
        self._state.reward_history.append(step_reward)
        self._state.action_history.append(
            {
                "step": self._state.step,
                "action_type": action_type,
                "target_service": action.target_service,
                "config_key": action.config_key,
                "config_value": action.config_value,
                "incident_id": incident_id,
                "channel_name": channel_name,
                "message_text": message_text,
                "actor_role": actor_role,
                "handoff_to": handoff_to,
                "reason": action.reason,
                "valid": result["valid"],
                "silence_bonus": result.get("silence_bonus", False),
                "step_reward": step_reward,
                "overall_health": new_health,
                "critical_services_healthy": self._critical_services_healthy(obs),
                "open_alerts": sum(1 for alert in obs.active_alerts if not alert.silenced),
                "action_details": result.get("details", ""),
                "protocol_status": dict(self._protocol_status),
            }
        )
        self._state.observation = obs
        self._state.protocol_status = dict(self._protocol_status)
        self._state.mode = self._mode
        self._state.multi_agent_kpis = dict(self._multi_agent_kpis)

        if self._enterprise_enabled():
            done = self._enterprise_episode_complete(obs) or self._state.step >= max_steps
        else:
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
                protocol_penalty=round(-protocol_penalty, 4),
                protocol_progress_bonus=round(protocol_progress_bonus, 4),
                completion_bonus=round(completion_bonus, 4),
                coordination_bonus=round(coordination_bonus, 4),
                coordination_penalty=round(-coordination_penalty, 4),
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
            "protocol_status": dict(self._protocol_status),
            "enterprise_enabled": self._enterprise_enabled(),
            "task_complete": done,
            "multi_agent_mode": self._mode == "multi_agent",
            "multi_agent_kpis": dict(self._multi_agent_kpis),
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
            apps_state=deepcopy(self._apps_state),
            protocol_status=dict(self._protocol_status),
            mode=self._mode,
            collaboration_state=deepcopy(self._multi_agent_state),
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

    def _enterprise_enabled(self) -> bool:
        if self._vdc is None:
            return False
        enterprise = self._vdc.scenario.get("enterprise_workflow", {})
        return bool(enterprise.get("enabled", False))

    def _init_enterprise_state(self) -> None:
        incident_id = ""
        if self._vdc is not None:
            incident_id = self._vdc.scenario.get("incident_context", {}).get("incident_id", "")

        self._protocol_status = {
            "is_acknowledged": False,
            "is_team_notified": False,
            "is_resolved": False,
        }
        self._enterprise_reward_flags = {
            "ack_bonus_awarded": False,
            "notify_bonus_awarded": False,
            "completion_bonus_awarded": False,
        }

        if not self._enterprise_enabled():
            self._apps_state = {}
            return

        self._apps_state = {
            "pagerduty": {
                "active_incident_ids": [incident_id] if incident_id else [],
                "tickets": {
                    incident_id: {
                        "status": "triggered",
                        "acknowledged": False,
                        "resolved": False,
                    }
                }
                if incident_id
                else {},
            },
            "slack": {
                "active_channels": ["incident-response", "sre-war-room"],
                "recent_messages": [],
            },
        }

    def _apply_enterprise_action(
        self,
        *,
        action_type: str,
        incident_id: str | None,
        channel_name: str | None,
        message_text: str | None,
    ) -> dict:
        result: dict = {
            "valid": True,
            "changed": False,
            "details": "",
            "silence_bonus": False,
            "ack_bonus": False,
            "notify_bonus": False,
        }

        if not self._enterprise_enabled():
            result["valid"] = False
            result["details"] = "Enterprise app actions are disabled for this scenario"
            return result

        if self._vdc is None or self._state is None:
            result["valid"] = False
            result["details"] = "Environment is not initialized"
            return result

        pagerduty = self._apps_state.get("pagerduty", {})
        slack = self._apps_state.get("slack", {})
        default_incident = self._vdc.scenario.get("incident_context", {}).get("incident_id", "")
        incident_id = incident_id or default_incident

        ticket = pagerduty.get("tickets", {}).get(incident_id)
        if action_type == "ACKNOWLEDGE_PAGERDUTY":
            if not incident_id or ticket is None:
                result["valid"] = False
                result["details"] = "ACKNOWLEDGE_PAGERDUTY requires a valid incident_id"
                return result
            if ticket.get("resolved"):
                result["details"] = f"Incident {incident_id} is already resolved"
                return result
            if not ticket.get("acknowledged"):
                ticket["acknowledged"] = True
                ticket["status"] = "acknowledged"
                self._protocol_status["is_acknowledged"] = True
                result["changed"] = True
                result["ack_bonus"] = True
            result["details"] = f"PagerDuty incident {incident_id} acknowledged"
            return result

        if action_type == "SEND_SLACK_MESSAGE":
            if not self._protocol_status.get("is_acknowledged", False):
                result["valid"] = False
                result["details"] = "SEND_SLACK_MESSAGE requires PagerDuty acknowledgement first"
                return result
            if not channel_name:
                result["valid"] = False
                result["details"] = "SEND_SLACK_MESSAGE requires channel_name"
                return result
            if not message_text:
                result["valid"] = False
                result["details"] = "SEND_SLACK_MESSAGE requires message_text"
                return result

            slack.setdefault("active_channels", [])
            if channel_name not in slack["active_channels"]:
                slack["active_channels"].append(channel_name)

            slack.setdefault("recent_messages", []).append(
                {
                    "channel_name": channel_name,
                    "message_text": message_text,
                    "incident_id": incident_id,
                    "step": self._state.step,
                }
            )
            slack["recent_messages"] = slack["recent_messages"][-10:]

            if not self._protocol_status.get("is_team_notified", False):
                self._protocol_status["is_team_notified"] = True
                result["notify_bonus"] = True
            result["changed"] = True
            result["details"] = f"Slack message posted to {channel_name}"
            return result

        if action_type == "RESOLVE_PAGERDUTY":
            if not incident_id or ticket is None:
                result["valid"] = False
                result["details"] = "RESOLVE_PAGERDUTY requires a valid incident_id"
                return result
            if not self._protocol_status.get("is_acknowledged", False):
                result["valid"] = False
                result["details"] = "RESOLVE_PAGERDUTY blocked until incident is acknowledged"
                return result
            if not self._protocol_status.get("is_team_notified", False):
                result["valid"] = False
                result["details"] = "RESOLVE_PAGERDUTY blocked until team is notified in Slack"
                return result
            if not self._enterprise_health_complete(self._build_observation(self._state.task_id, self._state.step, self._state.scenario_id)):
                result["valid"] = False
                result["details"] = "RESOLVE_PAGERDUTY blocked until infrastructure health target is met"
                return result

            ticket["resolved"] = True
            ticket["status"] = "resolved"
            self._protocol_status["is_resolved"] = True
            result["changed"] = True
            result["details"] = f"PagerDuty incident {incident_id} resolved"
            return result

        result["valid"] = False
        result["details"] = f"Unsupported enterprise action: {action_type}"
        return result

    def _enterprise_health_complete(self, obs: Observation) -> bool:
        if self._vdc is None:
            return False
        cfg = self._vdc.scenario.get("enterprise_workflow", {})
        completion_rule = cfg.get("completion_rule", {})
        mode = str(completion_rule.get("mode", "task_complete")).lower()

        if mode == "exact":
            target = float(completion_rule.get("value", 1.0))
            return obs.health_summary.overall >= target

        if mode == "threshold":
            target = float(completion_rule.get("value", 0.95))
            return obs.health_summary.overall >= target

        return self._task_complete(obs)

    def _enterprise_episode_complete(self, obs: Observation) -> bool:
        return self._protocol_status.get("is_resolved", False) and self._enterprise_health_complete(obs)

    # ------------------------------------------------------------------
    # Multi-agent helpers
    # ------------------------------------------------------------------

    def _init_multi_agent_state(self, scenario: dict) -> None:
        if self._mode != "multi_agent":
            self._multi_agent_state = {}
            self._multi_agent_kpis = {}
            self._multi_agent_permissions = {}
            return

        cfg = scenario.get("multi_agent", {})
        roles: list[str] = cfg.get(
            "roles",
            ["incident_commander", "investigator", "remediator", "comms_officer"],
        )
        raw_perms: dict = cfg.get("permissions", {})
        permissions: dict[str, set[str]] = {
            role: set(raw_perms.get(role, _DEFAULT_ROLE_PERMISSIONS.get(role, set())))
            for role in roles
        }
        self._multi_agent_permissions = permissions
        self._multi_agent_state = {
            "roles": roles,
            "permissions": {r: sorted(list(a)) for r, a in permissions.items()},
            "active_role": cfg.get("initial_role", "incident_commander"),
            "handoff_sequence": cfg.get(
                "handoff_sequence",
                ["incident_commander", "investigator", "remediator", "comms_officer", "incident_commander"],
            ),
            "role_objectives": cfg.get("role_objectives", {}),
        }
        self._multi_agent_kpis = {
            "handoff_count": 0.0,
            "conflict_count": 0.0,
            "redundant_action_rate": 0.0,
            "protocol_violations": 0.0,
            "time_to_consensus": 0.0,
        }

    def _apply_multi_agent_protocol(
        self,
        *,
        action_type: str,
        actor_role: str | None,
        handoff_to: str | None,
        target_service: str,
    ) -> dict:
        result: dict = {"valid": True, "details": "", "bonus": 0.0, "penalty": 0.0}
        if self._mode != "multi_agent":
            return result

        active_role = self._multi_agent_state.get("active_role")
        role = actor_role or active_role
        permissions = self._multi_agent_permissions

        if not role or role not in permissions:
            result["valid"] = False
            result["details"] = f"Unknown actor_role: {role!r}"
            result["penalty"] += 0.08
            self._multi_agent_kpis["protocol_violations"] += 1
            return result

        if role != active_role:
            result["penalty"] += 0.03
            self._multi_agent_kpis["conflict_count"] += 1

        if action_type not in permissions[role]:
            result["valid"] = False
            result["details"] = f"Role '{role}' is not permitted to execute {action_type}"
            result["penalty"] += 0.10
            self._multi_agent_kpis["protocol_violations"] += 1
            return result

        if self._state and self._state.action_history:
            last = self._state.action_history[-1]
            if last.get("action_type") == action_type and last.get("target_service") == target_service:
                result["penalty"] += 0.02
                self._multi_agent_kpis["conflict_count"] += 1

        if handoff_to:
            if handoff_to not in permissions:
                result["valid"] = False
                result["details"] = f"Invalid handoff_to role: {handoff_to!r}"
                result["penalty"] += 0.08
                self._multi_agent_kpis["protocol_violations"] += 1
                return result
            self._multi_agent_state["active_role"] = handoff_to
            self._multi_agent_kpis["handoff_count"] += 1
            result["bonus"] += 0.03

        seq: list[str] = self._multi_agent_state.get("handoff_sequence", [])
        idx = int(self._multi_agent_kpis["time_to_consensus"])
        if idx < len(seq) and role == seq[idx]:
            result["bonus"] += 0.01
            self._multi_agent_kpis["time_to_consensus"] += 1

        step_count = max(self._state.step if self._state else 1, 1)
        self._multi_agent_kpis["redundant_action_rate"] = round(
            self._multi_agent_kpis["conflict_count"] / step_count, 4
        )
        return result
