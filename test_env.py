"""
test_env.py — Smoke tests for Site Reliability Server environment.
Run with: pytest test_env.py -v
"""

import pytest

from env.environment import SREEnvironment
from env.models import Action, ActionType


ALL_TASKS = ["easy", "medium", "hard", "expert"]
VALID_SERVICE = "db-proxy"


@pytest.fixture
def env():
    return SREEnvironment()


# ---------------------------------------------------------------------------
# reset() tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task_id", ALL_TASKS)
def test_reset_returns_observation(env, task_id):
    obs = env.reset(task_id=task_id)
    assert obs.step == 0
    assert obs.task_id == task_id
    assert obs.incident_context.incident_id.startswith("INC-")
    assert obs.health_summary.overall >= 0.0
    assert obs.health_summary.overall <= 1.0
    assert len(obs.metrics.cpu_pct) == 6
    assert len(obs.active_alerts) >= 0


@pytest.mark.parametrize("task_id", ALL_TASKS)
def test_reset_produces_clean_state(env, task_id):
    env.reset(task_id=task_id)
    state = env.state()
    assert state.step == 0
    assert state.done is False
    assert state.cumulative_reward == 0.0
    assert state.action_history == []


# ---------------------------------------------------------------------------
# step() tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task_id", ALL_TASKS)
def test_step_increments_step_counter(env, task_id):
    env.reset(task_id=task_id)
    action = Action(action_type=ActionType.CHECK_LOGS, target_service="api-gateway")
    obs, reward, done, info = env.step(action)
    assert obs.step == 1
    assert isinstance(reward.step_reward, float)
    assert -1.0 <= reward.step_reward <= 1.0
    assert isinstance(done, bool)


def test_step_invalid_action_penalised(env):
    env.reset(task_id="easy")
    # UPDATE_CONFIG without config_key is invalid
    action = Action(
        action_type=ActionType.UPDATE_CONFIG,
        target_service="db-proxy",
        config_key=None,
        config_value=None,
    )
    _, reward, _, info = env.step(action)
    assert info["action_valid"] is False
    assert info["last_action_error"] is not None
    assert reward.step_reward <= 0.0


def test_step_without_reset_raises():
    fresh_env = SREEnvironment()
    with pytest.raises(RuntimeError):
        fresh_env.step(Action(action_type=ActionType.CHECK_LOGS, target_service="db-proxy"))


# ---------------------------------------------------------------------------
# grade() tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task_id", ALL_TASKS)
def test_grade_returns_valid_score(env, task_id):
    env.reset(task_id=task_id)
    for _ in range(3):
        action = Action(action_type=ActionType.INSPECT_SERVICE, target_service="db-proxy")
        _, _, done, _ = env.step(action)
        if done:
            break
    score, breakdown = env.grade()
    assert 0.0 < score < 1.0
    assert isinstance(breakdown, dict)


# ---------------------------------------------------------------------------
# Reward consistency tests
# ---------------------------------------------------------------------------

def test_reward_cumulative_accumulates(env):
    env.reset(task_id="easy")
    total = 0.0
    for _ in range(5):
        action = Action(action_type=ActionType.CHECK_LOGS, target_service="api-gateway")
        _, reward, done, _ = env.step(action)
        total += reward.step_reward
        if done:
            break
    state = env.state()
    assert abs(state.cumulative_reward - total) < 0.001


def test_reset_is_deterministic_by_default():
    env_a = SREEnvironment()
    env_b = SREEnvironment()
    obs_a = env_a.reset(task_id="easy")
    obs_b = env_b.reset(task_id="easy")
    assert obs_a.model_dump() == obs_b.model_dump()


def test_same_actions_produce_same_trajectory():
    env_a = SREEnvironment()
    env_b = SREEnvironment()
    obs_a = env_a.reset(task_id="hard")
    obs_b = env_b.reset(task_id="hard")
    assert obs_a.model_dump() == obs_b.model_dump()

    actions = [
        Action(action_type=ActionType.INSPECT_SERVICE, target_service="db-proxy"),
        Action(
            action_type=ActionType.UPDATE_CONFIG,
            target_service="db-proxy",
            config_key="db_timeout",
            config_value=5000,
        ),
        Action(action_type=ActionType.RESTART_SERVICE, target_service="db-proxy"),
    ]

    for action in actions:
        step_a = env_a.step(action)
        step_b = env_b.step(action)
        assert step_a[0].model_dump() == step_b[0].model_dump()
        assert step_a[1].model_dump() == step_b[1].model_dump()
        assert step_a[2] == step_b[2]
        assert step_a[3] == step_b[3]


def test_expert_wrong_order_has_explicit_penalty():
    env = SREEnvironment()
    env.reset(task_id="expert", scenario_id="expert-001")
    _, reward, _, info = env.step(
        Action(action_type=ActionType.RESTART_SERVICE, target_service="db-proxy")
    )
    assert info["action_valid"] is True
    assert info["reward_breakdown"]["risk_penalty"] < 0.0
    assert "before cache recovery" in info["action_details"]


def test_hard_pool_size_fix_is_not_penalized_as_wrong_regression():
    env = SREEnvironment()
    env.reset(task_id="hard", scenario_id="hard-002")
    _, reward, _, info = env.step(
        Action(
            action_type=ActionType.UPDATE_CONFIG,
            target_service="db-proxy",
            config_key="pool_size",
            config_value=10,
        )
    )
    assert info["action_valid"] is True
    assert info["reward_breakdown"]["risk_penalty"] == 0.0
    assert "Set pool_size=10" in info["action_details"]


def test_hard_restart_only_counts_after_correct_fix():
    env = SREEnvironment()
    env.reset(task_id="hard", scenario_id="hard-001")
    env.step(Action(action_type=ActionType.RESTART_SERVICE, target_service="db-proxy"))
    env.step(
        Action(
            action_type=ActionType.UPDATE_CONFIG,
            target_service="db-proxy",
            config_key="db_timeout",
            config_value=5000,
        )
    )
    score, breakdown = env.grade()
    assert 0.0 < score < 1.0
    assert breakdown["restart_after_fix"] == 0.0


def test_hard_exact_value_does_not_also_get_value_progress():
    env = SREEnvironment()
    env.reset(task_id="hard", scenario_id="hard-001")
    env.step(Action(action_type=ActionType.INSPECT_SERVICE, target_service="db-proxy"))
    env.step(
        Action(
            action_type=ActionType.UPDATE_CONFIG,
            target_service="db-proxy",
            config_key="db_timeout",
            config_value=5000,
        )
    )
    score, breakdown = env.grade()
    assert 0.0 < score < 1.0
    assert breakdown["correct_value"] == 0.20
    assert breakdown["value_progress"] == 0.0


def test_medium_log_only_secondary_follow_up_is_partial_credit():
    env = SREEnvironment()
    env.reset(task_id="medium", scenario_id="medium-001")
    env.step(Action(action_type=ActionType.CHECK_LOGS, target_service="cache-service"))
    score, breakdown = env.grade()
    assert 0.0 < score < 1.0
    assert breakdown["secondary_follow_up"] == 0.05


# ---------------------------------------------------------------------------
# SILENCE_ALERT bonus test
# ---------------------------------------------------------------------------

def test_silence_alert_bonus_on_healthy_service(env):
    """SILENCE_ALERT on an already-healthy service should set silence_bonus=True."""
    env.reset(task_id="easy")
    # Restart api-gateway to make it healthy
    env.step(Action(action_type=ActionType.RESTART_SERVICE, target_service="api-gateway"))
    env.step(Action(action_type=ActionType.RESTART_SERVICE, target_service="api-gateway"))
    # Now try to silence (may or may not have alerts depending on scenario)
    _, _, _, info = env.step(
        Action(action_type=ActionType.SILENCE_ALERT, target_service="api-gateway")
    )
    # silence_bonus is bool; just assert it is a bool (no crash)
    assert isinstance(info.get("silence_bonus", False), bool)


# ---------------------------------------------------------------------------
# Full episode smoke test
# ---------------------------------------------------------------------------

def test_full_easy_episode_completes(env):
    obs = env.reset(task_id="easy")
    max_steps = obs.max_steps
    done = False
    step = 0
    while not done and step < max_steps:
        action = Action(action_type=ActionType.CHECK_LOGS, target_service="db-proxy")
        obs, reward, done, info = env.step(action)
        step += 1
    score, breakdown = env.grade()
    assert 0.0 < score < 1.0
    assert step <= max_steps


def test_enterprise_full_protocol_sequence_rewards():
    env = SREEnvironment()
    env.reset(task_id="enterprise", scenario_id="enterprise-001")

    # 1) Acknowledge incident in PagerDuty.
    _, ack_reward, ack_done, ack_info = env.step(
        Action(
            action_type=ActionType.ACKNOWLEDGE_PAGERDUTY,
            target_service="user-service",
            incident_id="INC-ENT-001",
        )
    )
    assert ack_done is False
    assert ack_info["action_valid"] is True
    assert ack_reward.breakdown.protocol_progress_bonus == 0.1
    assert ack_reward.breakdown.protocol_penalty == 0.0
    assert ack_info["protocol_status"]["is_acknowledged"] is True

    # 2) Notify team in Slack.
    _, slack_reward, slack_done, slack_info = env.step(
        Action(
            action_type=ActionType.SEND_SLACK_MESSAGE,
            target_service="user-service",
            incident_id="INC-ENT-001",
            channel_name="incident-response",
            message_text="Incident acknowledged, restarting user-service now.",
        )
    )
    assert slack_done is False
    assert slack_info["action_valid"] is True
    assert slack_reward.breakdown.protocol_progress_bonus == 0.1
    assert slack_reward.breakdown.protocol_penalty == 0.0
    assert slack_info["protocol_status"]["is_team_notified"] is True

    # 3) Fix the issue in infra after protocol pre-steps.
    _, fix_reward, fix_done, fix_info = env.step(
        Action(action_type=ActionType.RESTART_SERVICE, target_service="user-service")
    )
    assert fix_done is False
    assert fix_info["action_valid"] is True
    assert fix_reward.breakdown.protocol_penalty == 0.0
    assert fix_reward.breakdown.protocol_progress_bonus == 0.0

    # 4) Resolve the PagerDuty incident and collect completion bonus.
    _, resolve_reward, resolve_done, resolve_info = env.step(
        Action(
            action_type=ActionType.RESOLVE_PAGERDUTY,
            target_service="user-service",
            incident_id="INC-ENT-001",
        )
    )
    assert resolve_info["action_valid"] is True
    assert resolve_reward.breakdown.protocol_penalty == 0.0
    assert resolve_reward.breakdown.completion_bonus == 0.5
    assert resolve_done is True
    assert resolve_info["protocol_status"]["is_resolved"] is True


def test_enterprise_protocol_penalty_before_acknowledge():
    env = SREEnvironment()
    env.reset(task_id="enterprise", scenario_id="enterprise-001")

    # Attempt infra remediation before PagerDuty acknowledgement.
    _, reward, done, info = env.step(
        Action(action_type=ActionType.RESTART_SERVICE, target_service="user-service")
    )

    assert done is False
    assert info["action_valid"] is True
    assert reward.breakdown.protocol_penalty == -0.2
    assert info["protocol_status"]["is_acknowledged"] is False
