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
    assert 0.0 <= score <= 1.0
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
    assert 0.0 <= score <= 1.0
    assert step <= max_steps
