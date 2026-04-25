from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ModeType = Literal["single_agent", "multi_agent"]
AgentRole = Literal["incident_commander", "investigator", "remediator", "comms_officer"]


class ActionType(str, Enum):
    CHECK_LOGS = "CHECK_LOGS"
    INSPECT_SERVICE = "INSPECT_SERVICE"
    DRAIN_TRAFFIC = "DRAIN_TRAFFIC"
    RESTART_SERVICE = "RESTART_SERVICE"
    SCALE_UP = "SCALE_UP"
    SCALE_DOWN = "SCALE_DOWN"
    ROLLBACK = "ROLLBACK"
    UPDATE_CONFIG = "UPDATE_CONFIG"
    SILENCE_ALERT = "SILENCE_ALERT"
    ACKNOWLEDGE_PAGERDUTY = "ACKNOWLEDGE_PAGERDUTY"
    SEND_SLACK_MESSAGE = "SEND_SLACK_MESSAGE"
    RESOLVE_PAGERDUTY = "RESOLVE_PAGERDUTY"


VALID_SERVICES = [
    "api-gateway",
    "auth-service",
    "user-service",
    "order-service",
    "db-proxy",
    "cache-service",
]


class SystemMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cpu_pct: dict[str, float] = Field(description="Per-service CPU usage 0-100")
    mem_pct: dict[str, float] = Field(description="Per-service memory usage 0-100")
    error_rate: dict[str, float] = Field(description="Per-service HTTP error rate 0.0-1.0")
    latency_ms: dict[str, float] = Field(description="Per-service p99 latency in ms")
    timestamp: datetime = Field(description="Timestamp for the metric snapshot")


class LogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime = Field(description="Timestamp of the log entry")
    service: str = Field(description="Service that emitted the log")
    severity: str = Field(description="Severity level: DEBUG, INFO, WARN, ERROR, CRITICAL")
    message: str = Field(description="Human-readable log message")


class DeployEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deploy_id: str = Field(description="Unique deployment identifier")
    timestamp: datetime = Field(description="Deployment timestamp")
    service: str = Field(description="Service affected by deployment")
    changes: dict[str, Any] = Field(description="Config diffs applied in this deploy")


class Alert(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alert_id: str = Field(description="Unique alert identifier")
    service: str = Field(description="Service associated with the alert")
    metric: str = Field(description="Metric that breached threshold")
    threshold: float = Field(description="Threshold value for the metric")
    current: float = Field(description="Current observed metric value")
    severity: str = Field(description="Alert severity level")
    triggered_at: datetime = Field(description="Timestamp when the alert was triggered")
    silenced: bool = Field(default=False, description="Whether the alert has been silenced")


class HealthSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    per_service: dict[str, float] = Field(description="Per-service health score in range 0.0-1.0")
    overall: float = Field(description="Mean health score across all services")


class IncidentContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    incident_id: str = Field(description="Stable incident identifier")
    title: str = Field(description="Short incident title")
    severity: Literal["SEV-1", "SEV-2", "SEV-3"] = Field(description="Business severity")
    business_service: str = Field(description="Customer-facing capability affected")
    customer_impact: str = Field(description="Summary of user-visible impact")
    symptom_summary: str = Field(description="Primary symptom reported by monitoring or humans")
    suspected_services: list[str] = Field(description="Services initially suspected by triage")
    failure_mode: str = Field(description="Operational pattern behind the incident")
    success_criteria: str = Field(description="What successful mitigation looks like")


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int = Field(description="Current step in episode")
    max_steps: int = Field(description="Max steps allowed")
    task_id: Literal["easy", "medium", "hard", "expert", "enterprise"] = Field(
        description="Active task: easy|medium|hard|expert|enterprise"
    )
    metrics: SystemMetrics = Field(description="Current system metrics")
    logs: list[LogEntry] = Field(description="Last 10 log entries")
    deploy_history: list[DeployEvent] = Field(description="Last 5 deploy events")
    current_config: dict[str, Any] = Field(description="Live config key-value pairs")
    service_graph: dict[str, list[str]] = Field(description="Service dependency map")
    active_alerts: list[Alert] = Field(description="Active alerts")
    health_summary: HealthSummary = Field(description="Current health summary")
    incident_context: IncidentContext = Field(description="Current incident ticket and operational context")
    apps_state: dict[str, Any] = Field(
        default_factory=dict,
        description="Mock enterprise app state, e.g. PagerDuty tickets and Slack channels/messages",
    )
    protocol_status: dict[str, bool] = Field(
        default_factory=dict,
        description="Current enterprise workflow flags (is_acknowledged, is_team_notified, is_resolved)",
    )
    mode: ModeType = Field(
        default="single_agent",
        description="Execution mode: single_agent or multi_agent",
    )
    collaboration_state: dict[str, Any] = Field(
        default_factory=dict,
        description="Multi-agent coordination state (active role, permissions, objectives)",
    )


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: ActionType = Field(description="Type of action to perform")
    target_service: Literal[
        "api-gateway",
        "auth-service",
        "user-service",
        "order-service",
        "db-proxy",
        "cache-service",
    ] = Field(description="Must be one of VALID_SERVICES")
    config_key: str | None = Field(default=None, description="Required for UPDATE_CONFIG")
    config_value: Any | None = Field(default=None, description="Required for UPDATE_CONFIG")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Generic action arguments for enterprise workflows, e.g. "
            "incident_id, channel_name, message_text"
        ),
    )
    incident_id: str | None = Field(default=None, description="PagerDuty incident id for ACK/RESOLVE")
    channel_name: str | None = Field(default=None, description="Slack channel for SEND_SLACK_MESSAGE")
    message_text: str | None = Field(default=None, description="Slack message content")
    reason: str | None = Field(default=None, description="Agent reasoning used by Task 1 grader")
    actor_role: AgentRole | None = Field(
        default=None,
        description="Acting role in multi-agent mode",
    )
    handoff_to: AgentRole | None = Field(
        default=None,
        description="Role to hand control to after this action (multi-agent mode)",
    )


class RewardBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    health_delta: float = Field(description="Weighted health delta contribution")
    task_progress: float = Field(description="Weighted progress on the critical incident services")
    cost_efficiency: float = Field(description="Weighted cost efficiency contribution")
    latency_delta: float = Field(description="Weighted latency contribution")
    invalid_penalty: float = Field(description="Weighted invalid action penalty")
    risk_penalty: float = Field(description="Penalty for disruptive or misleading actions")
    protocol_penalty: float = Field(default=0.0, description="Penalty for breaking enterprise protocol")
    protocol_progress_bonus: float = Field(default=0.0, description="Bonus for protocol milestones")
    completion_bonus: float = Field(default=0.0, description="Bonus for successful enterprise completion")
    coordination_bonus: float = Field(default=0.0, description="Bonus for valid handoffs/sequencing in multi-agent mode")
    coordination_penalty: float = Field(default=0.0, description="Penalty for role violations/conflicts in multi-agent mode")


class Reward(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_reward: float = Field(description="Reward for the current step")
    cumulative: float = Field(description="Cumulative reward for the episode")
    breakdown: RewardBreakdown = Field(description="Reward component breakdown")


class EpisodeState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: Literal["easy", "medium", "hard", "expert", "enterprise"] = Field(description="Current task id")
    scenario_id: str = Field(description="Active scenario id")
    step: int = Field(description="Current step count")
    done: bool = Field(description="Whether the episode has terminated")
    observation: Observation = Field(description="Latest environment observation")
    action_history: list[dict] = Field(description="Sequence of action payloads")
    reward_history: list[float] = Field(description="Per-step reward values")
    cumulative_reward: float = Field(description="Total accumulated reward")
    protocol_status: dict[str, bool] = Field(
        default_factory=lambda: {
            "is_acknowledged": False,
            "is_team_notified": False,
            "is_resolved": False,
        },
        description="Internal enterprise protocol flags",
    )
    mode: ModeType = Field(
        default="single_agent",
        description="Execution mode for this episode",
    )
    multi_agent_kpis: dict[str, float] = Field(
        default_factory=dict,
        description="Collaboration KPIs tracked in multi-agent mode",
    )
    grader_score: float | None = Field(default=None, description="Optional grader score")
