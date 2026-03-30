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
    import json
    from pathlib import Path

    root_cause = "db-proxy"
    scenario_path = Path(__file__).parent.parent / "scenarios" / state.task_id / f"{state.scenario_id}.json"
    if scenario_path.exists():
        try:
            payload = json.loads(scenario_path.read_text())
            root_cause = payload.get("ground_truth", {}).get("root_cause_service", root_cause)
        except Exception:
            pass

    breakdown = {
        "root_identified": 0.0,
        "no_false_positives": 0.0,
        "health_restored": 0.0,
        "efficiency": 0.0,
    }

    for action in state.action_history:
        reason = (action.get("reason") or "").lower()
        if root_cause.replace("-", "") in reason or root_cause in reason:
            breakdown["root_identified"] = 0.40
            break

    false_positives = sum(
        1
        for action in state.action_history
        if action.get("action_type") == "RESTART_SERVICE" and action.get("target_service") != root_cause
    )
    breakdown["no_false_positives"] = max(0.0, 0.30 - (false_positives * 0.10))

    final_health = state.observation.health_summary.overall
    breakdown["health_restored"] = round(min(0.20, final_health * 0.20), 4)

    if state.step <= 8:
        breakdown["efficiency"] = 0.10

    score = sum(breakdown.values())
    return round(min(1.0, score), 4), breakdown


def grade_medium(state: EpisodeState) -> tuple[float, dict]:
    """
    Task 2 — The First Responder.
    Score = mean of 4 independent metric scores (each 0.0-1.0).
    Each metric scored linearly from worst-case to threshold.
    """
    from .simulator import HEALTH_THRESHOLDS

    metrics = state.observation.metrics
    thresholds = HEALTH_THRESHOLDS
    worst_cases = {
        "cpu_pct": 100.0,
        "mem_pct": 100.0,
        "error_rate": 1.0,
        "latency_ms": 2000.0,
    }

    metric_scores: dict[str, float] = {}
    for metric in ["cpu_pct", "mem_pct", "error_rate", "latency_ms"]:
        values = getattr(metrics, metric).values()
        worst_val = max(values)
        threshold = thresholds[metric]
        worst = worst_cases[metric]
        if worst_val <= threshold:
            metric_scores[metric] = 1.0
        else:
            metric_scores[metric] = round(
                max(0.0, 1.0 - (worst_val - threshold) / (worst - threshold)),
                4,
            )

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
    correct_key = "db_timeout"
    correct_value = 5000

    breakdown = {
        "diagnosis": 0.0,
        "correct_key": 0.0,
        "value_progress": 0.0,
        "correct_value": 0.0,
        "health_restored": 0.0,
        "efficiency": 0.0,
    }

    if any(
        action.get("target_service") == "db-proxy"
        and action.get("action_type") in {"CHECK_LOGS", "INSPECT_SERVICE"}
        for action in state.action_history
    ):
        breakdown["diagnosis"] = 0.15

    for action in state.action_history:
        if action.get("action_type") != "UPDATE_CONFIG":
            continue
        if action.get("config_key") != correct_key:
            continue

        breakdown["correct_key"] = 0.25
        val = action.get("config_value")
        if val is None:
            continue
        try:
            val_int = int(val)
        except (TypeError, ValueError):
            continue

        if val_int == correct_value:
            breakdown["correct_value"] = 0.35
            breakdown["value_progress"] = max(breakdown["value_progress"], 0.20)
        elif 2500 <= val_int <= 8000:
            breakdown["value_progress"] = max(breakdown["value_progress"], 0.15)
        elif 1000 <= val_int < 2500:
            breakdown["value_progress"] = max(breakdown["value_progress"], 0.08)

    final_health = state.observation.health_summary.overall
    breakdown["health_restored"] = round(min(0.03, final_health * 0.03), 4)

    if state.step <= 10 and breakdown["correct_value"] > 0:
        breakdown["efficiency"] = 0.02

    score = sum(breakdown.values())
    return round(min(1.0, score), 4), breakdown


GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}
