from .models import EpisodeState


_VALIDATOR_EPS = 0.0001


def _validator_safe_score(raw_score: float) -> float:
    return round(_VALIDATOR_EPS + raw_score * (1 - 2 * _VALIDATOR_EPS), 4)


def _load_ground_truth(state: EpisodeState) -> dict:
    import json
    from pathlib import Path

    scenario_path = (
        Path(__file__).parent.parent
        / "scenarios"
        / state.task_id
        / f"{state.scenario_id}.json"
    )
    if not scenario_path.exists():
        return {}
    try:
        payload = json.loads(scenario_path.read_text())
        return payload.get("ground_truth", {})
    except Exception:
        return {}


def _count_oscillations(state: EpisodeState, service: str, action_a: str, action_b: str) -> int:
    relevant = [
        action.get("action_type")
        for action in state.action_history
        if action.get("target_service") == service
        and action.get("action_type") in {action_a, action_b}
    ]
    return sum(1 for left, right in zip(relevant, relevant[1:]) if left != right)


def _first_index(
    state: EpisodeState,
    *,
    action_type: str | None = None,
    target_service: str | None = None,
    config_key: str | None = None,
) -> int | None:
    for index, action in enumerate(state.action_history):
        if action_type is not None and action.get("action_type") != action_type:
            continue
        if target_service is not None and action.get("target_service") != target_service:
            continue
        if config_key is not None and action.get("config_key") != config_key:
            continue
        return index
    return None


def _enterprise_enabled(state: EpisodeState) -> bool:
    apps_state = getattr(state.observation, "apps_state", {}) or {}
    return "pagerduty" in apps_state and "slack" in apps_state


def _enterprise_breakdown(state: EpisodeState) -> tuple[float, dict]:
    if not _enterprise_enabled(state):
        return 0.0, {
            "enterprise_enabled": False,
            "enterprise_protocol_component": 0.0,
        }

    infra_actions = {
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

    ack_idx = _first_index(state, action_type="ACKNOWLEDGE_PAGERDUTY")
    notify_idx = _first_index(state, action_type="SEND_SLACK_MESSAGE")
    resolve_idx = _first_index(state, action_type="RESOLVE_PAGERDUTY")
    first_infra_idx = next(
        (
            idx
            for idx, action in enumerate(state.action_history)
            if action.get("action_type") in infra_actions
        ),
        None,
    )

    milestones = [ack_idx is not None, notify_idx is not None, resolve_idx is not None]
    milestone_ratio = sum(milestones) / 3.0
    order_ok = (
        ack_idx is not None
        and notify_idx is not None
        and resolve_idx is not None
        and ack_idx < notify_idx < resolve_idx
    )
    infra_after_ack = (
        first_infra_idx is None
        or ack_idx is None
        or first_infra_idx > ack_idx
    )
    compliance_ratio = (
        milestone_ratio * 0.6
        + (0.2 if order_ok else 0.0)
        + (0.2 if infra_after_ack else 0.0)
    )
    enterprise_component = round(0.05 * compliance_ratio, 4)

    return enterprise_component, {
        "enterprise_enabled": True,
        "enterprise_ack_index": ack_idx,
        "enterprise_notify_index": notify_idx,
        "enterprise_resolve_index": resolve_idx,
        "enterprise_first_infra_index": first_infra_idx,
        "enterprise_sequence_ok": order_ok,
        "enterprise_infra_after_ack": infra_after_ack,
        "enterprise_protocol_compliance": round(compliance_ratio, 4),
        "enterprise_protocol_component": enterprise_component,
    }


def _multi_agent_breakdown(state: EpisodeState) -> tuple[float, dict]:
    if getattr(state, "mode", "single_agent") != "multi_agent":
        return 0.0, {
            "multi_agent_enabled": False,
            "collaboration_quality": 0.0,
            "role_discipline": 0.0,
            "coordination_component": 0.0,
        }

    kpis = getattr(state, "multi_agent_kpis", {}) or {}
    handoff_count = float(kpis.get("handoff_count", 0.0))
    conflict_count = float(kpis.get("conflict_count", 0.0))
    redundant_rate = float(kpis.get("redundant_action_rate", 0.0))
    violations = float(kpis.get("protocol_violations", 0.0))
    consensus = float(kpis.get("time_to_consensus", 0.0))

    collab = max(0.0, min(1.0,
        0.35
        + min(0.35, handoff_count * 0.08)
        + min(0.30, consensus * 0.04)
        - min(0.30, conflict_count * 0.06),
    ))
    discipline = max(0.0,
        1.0 - min(0.7, violations * 0.18) - min(0.3, redundant_rate * 0.5)
    )
    component = round(0.08 * (0.6 * collab + 0.4 * discipline), 4)

    return component, {
        "multi_agent_enabled": True,
        "collaboration_quality": round(collab, 4),
        "role_discipline": round(discipline, 4),
        "handoff_count": handoff_count,
        "conflict_count": conflict_count,
        "redundant_action_rate": round(redundant_rate, 4),
        "protocol_violations": violations,
        "time_to_consensus": consensus,
        "coordination_component": component,
    }


def grade_easy(state: EpisodeState) -> tuple[float, dict]:
    """
    Task 1 — The Detective.
    Score breakdown:
      - Root cause identified in reasoning and action selection: 0.40
      - Targeted remediation on the actual failing service: 0.20
      - Operational discipline (avoid disruptive side actions / premature silencing): 0.20
      - Final health restoration: 0.15
      - Efficiency bonus for fast clean resolution: 0.05

    Grader requires BOTH the `reason` field to mention the root service AND
    at least one action targeting that service (CHECK_LOGS, INSPECT_SERVICE,
    or RESTART_SERVICE) — simple string-matching on reason alone is insufficient.
    """
    root_cause = _load_ground_truth(state).get("root_cause_service", "db-proxy")

    breakdown = {
        "root_identified": 0.0,
        "targeted_remediation": 0.0,
        "operational_discipline": 0.0,
        "health_restored": 0.0,
        "efficiency": 0.0,
    }

    diagnostic_index = next(
        (
            index
            for index, action in enumerate(state.action_history)
            if action.get("target_service") == root_cause
            and action.get("action_type") in {"CHECK_LOGS", "INSPECT_SERVICE"}
        ),
        None,
    )
    first_targeted_restart = _first_index(
        state,
        action_type="RESTART_SERVICE",
        target_service=root_cause,
    )

    if (
        diagnostic_index is not None
        and first_targeted_restart is not None
        and diagnostic_index < first_targeted_restart
    ):
        breakdown["root_identified"] = 0.40
    elif diagnostic_index is not None:
        breakdown["root_identified"] = 0.20

    targeted_restarts = sum(
        1
        for action in state.action_history
        if action.get("action_type") == "RESTART_SERVICE"
        and action.get("target_service") == root_cause
    )
    if targeted_restarts > 0:
        breakdown["targeted_remediation"] = 0.20

    disruptive_actions = sum(
        1
        for action in state.action_history
        if action.get("target_service") != root_cause
        and action.get("action_type") in {"RESTART_SERVICE", "ROLLBACK", "SCALE_DOWN"}
    )
    premature_silence = sum(
        1
        for index, action in enumerate(state.action_history)
        if action.get("action_type") == "SILENCE_ALERT"
        and action.get("target_service") == root_cause
        and (first_targeted_restart is None or index < first_targeted_restart)
    )
    breakdown["operational_discipline"] = max(0.0, 0.20 - disruptive_actions * 0.10 - premature_silence * 0.05)

    final_health = state.observation.health_summary.overall
    breakdown["health_restored"] = round(min(0.15, final_health * 0.15), 4)

    if state.step <= 6 and targeted_restarts > 0:
        breakdown["efficiency"] = 0.05

    score = sum(breakdown.values())
    enterprise_component, enterprise = _enterprise_breakdown(state)
    score += enterprise_component
    breakdown.update(enterprise)
    ma_component, ma_breakdown = _multi_agent_breakdown(state)
    score += ma_component
    breakdown.update(ma_breakdown)
    return _validator_safe_score(score), breakdown


def grade_medium(state: EpisodeState) -> tuple[float, dict]:
    """
    Task 2 — The First Responder.
    Score blends:
      - Metric restoration quality across CPU / memory / errors / latency: 50%
      - Primary capacity action quality: 18%
      - Temporary load-shedding support: 07%
      - Follow-up action on the stressed dependency: 12%
      - Efficiency bonus: 08%
      - Operational discipline: 05%
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

    metric_component = round(sum(metric_scores.values()) / 4.0, 4)

    worst_service_per_metric: dict[str, str] = {}
    worst_value_per_metric: dict[str, float] = {}
    for metric in ["cpu_pct", "mem_pct", "error_rate", "latency_ms"]:
        series = getattr(metrics, metric)
        service, value = max(series.items(), key=lambda item: item[1])
        worst_service_per_metric[metric] = service
        worst_value_per_metric[metric] = value

    breakdown: dict[str, object] = dict(metric_scores)
    breakdown["worst_service_per_metric"] = worst_service_per_metric
    breakdown["worst_value_per_metric"] = worst_value_per_metric
    breakdown["thresholds"] = thresholds

    root = _load_ground_truth(state).get("root_cause_service")
    secondary = _load_ground_truth(state).get("secondary_cause_service")
    if root:
        breakdown["scenario_root_cause_service"] = root
    if secondary:
        breakdown["scenario_secondary_service"] = secondary

    scaled_primary = any(
        action.get("target_service") == root and action.get("action_type") == "SCALE_UP"
        for action in state.action_history
    )
    drained_primary = any(
        action.get("target_service") == root and action.get("action_type") == "DRAIN_TRAFFIC"
        for action in state.action_history
    )
    restarted_primary = any(
        action.get("target_service") == root and action.get("action_type") == "RESTART_SERVICE"
        for action in state.action_history
    )
    secondary_logs = bool(
        secondary
        and any(
            action.get("target_service") == secondary
            and action.get("action_type") == "CHECK_LOGS"
            for action in state.action_history
        )
    )
    secondary_action = bool(
        secondary
        and any(
            action.get("target_service") == secondary
            and action.get("action_type") in {"RESTART_SERVICE", "SCALE_UP", "DRAIN_TRAFFIC"}
            for action in state.action_history
        )
    )
    first_secondary_logs_index = _first_index(
        state,
        action_type="CHECK_LOGS",
        target_service=secondary,
    ) if secondary else None
    first_secondary_action_index = next(
        (
            index
            for index, action in enumerate(state.action_history)
            if action.get("target_service") == secondary
            and action.get("action_type") in {"RESTART_SERVICE", "SCALE_UP", "DRAIN_TRAFFIC"}
        ),
        None,
    )
    if (
        secondary_logs
        and secondary_action
        and first_secondary_logs_index is not None
        and first_secondary_action_index is not None
        and first_secondary_logs_index < first_secondary_action_index
    ):
        secondary_follow_up_credit = 0.12
    elif secondary_logs or secondary_action:
        secondary_follow_up_credit = 0.05
    else:
        secondary_follow_up_credit = 0.0
    unnecessary_scale_down = sum(
        1 for action in state.action_history if action.get("action_type") == "SCALE_DOWN"
    )
    oscillation_penalty = _count_oscillations(state, root or "", "SCALE_UP", "RESTART_SERVICE")
    repeated_primary_restarts = sum(
        1
        for action in state.action_history
        if action.get("target_service") == root and action.get("action_type") == "RESTART_SERVICE"
    )
    contradictory_penalty = 0.0
    if scaled_primary and restarted_primary and oscillation_penalty > 0:
        contradictory_penalty += 0.02
    if oscillation_penalty > 1:
        contradictory_penalty += min(0.03, (oscillation_penalty - 1) * 0.03)
    efficiency = 1.0 if state.step <= 7 else 0.45 if state.step <= 10 else 0.0
    discipline = max(
        0.0,
        0.05
        - min(0.03, unnecessary_scale_down * 0.03)
        - min(0.03, max(0, repeated_primary_restarts - 1) * 0.015)
        - min(0.04, oscillation_penalty * 0.01)
        - min(0.05, contradictory_penalty),
    )

    score = (
        metric_component * 0.50
        + (0.18 if scaled_primary else 0.0)
        + (0.07 if drained_primary else 0.0)
        + secondary_follow_up_credit
        + (0.08 * efficiency)
        + discipline
    )
    breakdown["metric_component"] = metric_component
    breakdown["scaled_primary"] = 1.0 if scaled_primary else 0.0
    breakdown["drained_primary"] = 1.0 if drained_primary else 0.0
    breakdown["restarted_primary"] = 1.0 if restarted_primary else 0.0
    breakdown["secondary_follow_up"] = round(secondary_follow_up_credit, 4)
    breakdown["secondary_logs"] = 1.0 if secondary_logs else 0.0
    breakdown["secondary_action"] = 1.0 if secondary_action else 0.0
    breakdown["first_secondary_logs_index"] = first_secondary_logs_index
    breakdown["first_secondary_action_index"] = first_secondary_action_index
    breakdown["oscillation_count"] = oscillation_penalty
    breakdown["contradictory_penalty"] = round(contradictory_penalty, 4)
    breakdown["primary_restart_count"] = repeated_primary_restarts
    breakdown["efficiency_component"] = round(0.08 * efficiency, 4)
    breakdown["discipline_component"] = round(discipline, 4)
    breakdown["unnecessary_scale_down_count"] = unnecessary_scale_down

    enterprise_component, enterprise = _enterprise_breakdown(state)
    score += enterprise_component
    breakdown.update(enterprise)

    ma_component, ma_breakdown = _multi_agent_breakdown(state)
    score += ma_component
    breakdown.update(ma_breakdown)

    return _validator_safe_score(score), breakdown


def grade_hard(state: EpisodeState) -> tuple[float, dict]:
    """
    Task 3 — The Architect.
    Score breakdown:
      - Diagnosis on db-proxy plus evidence gathering from logs/upstream: 0.20
      - Correct config key targeted: 0.15
      - Correct config value applied: 0.20
      - Partial value progress for plausible timeout fixes: up to 0.10
      - Restart after applying the fix: 0.10
      - Health restored: up to 0.15
      - Discipline bonus for avoiding misleading actions: up to 0.04
      - Efficiency bonus for resolving quickly: 0.06
    """
    expected = _load_ground_truth(state)
    correct_key = expected.get("correct_config_key", "db_timeout")
    correct_value = expected.get("correct_config_value", 5000)

    components: dict[str, float] = {
        "diagnosis": 0.0,
        "correct_key": 0.0,
        "value_progress": 0.0,
        "correct_value": 0.0,
        "restart_after_fix": 0.0,
        "health_restored": 0.0,
        "discipline": 0.0,
        "efficiency": 0.0,
    }

    first_correct_fix_index = _first_index(
        state,
        action_type="UPDATE_CONFIG",
        target_service="db-proxy",
        config_key=correct_key,
    )
    inspected_db_index = next(
        (
            index
            for index, action in enumerate(state.action_history)
            if action.get("target_service") == "db-proxy"
            and action.get("action_type") in {"CHECK_LOGS", "INSPECT_SERVICE"}
        ),
        None,
    )
    upstream_evidence_index = next(
        (
            index
            for index, action in enumerate(state.action_history)
            if action.get("target_service") in {"auth-service", "user-service", "order-service"}
            and action.get("action_type") in {"CHECK_LOGS", "INSPECT_SERVICE"}
        ),
        None,
    )
    if inspected_db_index is not None:
        if first_correct_fix_index is None or inspected_db_index < first_correct_fix_index:
            components["diagnosis"] += 0.10
        else:
            components["diagnosis"] += 0.05
    if upstream_evidence_index is not None:
        if first_correct_fix_index is None or upstream_evidence_index < first_correct_fix_index:
            components["diagnosis"] += 0.10
        else:
            components["diagnosis"] += 0.05

    for action in state.action_history:
        if action.get("action_type") != "UPDATE_CONFIG":
            continue
        if action.get("config_key") != correct_key:
            continue

        components["correct_key"] = 0.15
        val = action.get("config_value")
        if val is None:
            continue
        try:
            val_int = int(val)
        except (TypeError, ValueError):
            continue

        if val_int == correct_value:
            components["correct_value"] = 0.20
        else:
            closeness = max(
                0.0,
                1.0 - (abs(val_int - correct_value) / max(abs(correct_value), 1)),
            )
            components["value_progress"] = max(
                components["value_progress"],
                round(min(0.10, 0.10 * closeness), 4),
            )

    first_restart_index = _first_index(
        state,
        action_type="RESTART_SERVICE",
        target_service="db-proxy",
    )
    if (
        first_restart_index is not None
        and first_correct_fix_index is not None
        and first_restart_index > first_correct_fix_index
    ):
        components["restart_after_fix"] = 0.10

    final_health = state.observation.health_summary.overall
    components["health_restored"] = round(min(0.15, final_health * 0.15), 4)

    rollback_attempts = sum(
        1 for action in state.action_history if action.get("action_type") == "ROLLBACK"
    )
    wrong_config_updates = sum(
        1
        for action in state.action_history
        if action.get("action_type") == "UPDATE_CONFIG" and action.get("config_key") != correct_key
    )
    components["discipline"] = max(0.0, 0.04 - rollback_attempts * 0.025 - wrong_config_updates * 0.015)

    if state.step <= 8 and components["correct_value"] > 0 and components["restart_after_fix"] > 0:
        components["efficiency"] = 0.06

    breakdown: dict[str, object] = dict(components)
    breakdown["scenario_expected_key"] = str(correct_key)
    breakdown["scenario_expected_value"] = int(correct_value)
    breakdown["first_correct_fix_index"] = first_correct_fix_index
    breakdown["first_restart_index"] = first_restart_index
    breakdown["update_actions_attempted"] = sum(
        1 for action in state.action_history if action.get("action_type") == "UPDATE_CONFIG"
    )

    score = sum(components.values())
    enterprise_component, enterprise = _enterprise_breakdown(state)
    score += enterprise_component
    breakdown.update(enterprise)
    ma_component, ma_breakdown = _multi_agent_breakdown(state)
    score += ma_component
    breakdown.update(ma_breakdown)
    return _validator_safe_score(score), breakdown


def grade_expert(state: EpisodeState) -> tuple[float, dict]:
    """
    Task 4 — The Storm Chaser.
    Score breakdown:
      - cache-service restarted before db-proxy: 0.25
      - cache-service recovered: up to 0.15
      - db-proxy recovered: up to 0.15
      - overall system health restored: up to 0.15
      - No unnecessary restarts of healthy services: up to 0.10
      - Alert hygiene (no premature silencing): up to 0.05
      - Efficient completion after recovery: 0.10
      - Minimal post-recovery stalling: 0.05
    """
    components: dict[str, float] = {
        "correct_order": 0.0,
        "cache_recovered": 0.0,
        "db_recovered": 0.0,
        "health_restored": 0.0,
        "no_collateral": 0.0,
        "alert_hygiene": 0.0,
        "efficiency": 0.0,
        "finish_quality": 0.0,
    }

    SIDE_SERVICES = {"api-gateway", "auth-service", "user-service", "order-service"}

    restart_order: list[str] = [
        action["target_service"]
        for action in state.action_history
        if action.get("action_type") == "RESTART_SERVICE"
    ]
    action_history_cache_idx = next(
        (
            index
            for index, action in enumerate(state.action_history)
            if action.get("action_type") == "RESTART_SERVICE"
            and action.get("target_service") == "cache-service"
        ),
        None,
    )
    action_history_db_idx = next(
        (
            index
            for index, action in enumerate(state.action_history)
            if action.get("action_type") == "RESTART_SERVICE"
            and action.get("target_service") == "db-proxy"
        ),
        None,
    )

    cache_idx = next(
        (i for i, s in enumerate(restart_order) if s == "cache-service"), None
    )
    db_idx = next(
        (i for i, s in enumerate(restart_order) if s == "db-proxy"), None
    )

    if cache_idx is not None and db_idx is not None and cache_idx < db_idx:
        components["correct_order"] = 0.25
    elif cache_idx is not None:
        components["correct_order"] = 0.10

    cache_health = state.observation.health_summary.per_service.get("cache-service", 0.0)
    db_health = state.observation.health_summary.per_service.get("db-proxy", 0.0)
    components["cache_recovered"] = round(min(0.15, cache_health * 0.15), 4)
    components["db_recovered"] = round(min(0.15, db_health * 0.15), 4)

    final_health = state.observation.health_summary.overall
    components["health_restored"] = round(min(0.15, final_health * 0.15), 4)

    collateral_restarts = sum(
        1
        for action in state.action_history
        if action.get("action_type") == "RESTART_SERVICE"
        and action.get("target_service") in SIDE_SERVICES
    )
    cache_restart_count = sum(
        1
        for action in state.action_history
        if action.get("action_type") == "RESTART_SERVICE"
        and action.get("target_service") == "cache-service"
    )
    db_restart_count = sum(
        1
        for action in state.action_history
        if action.get("action_type") == "RESTART_SERVICE"
        and action.get("target_service") == "db-proxy"
    )
    excess_critical_restarts = max(0, cache_restart_count - 3) + max(0, db_restart_count - 3)
    components["no_collateral"] = max(
        0.0,
        0.10 - collateral_restarts * 0.05 - excess_critical_restarts * 0.05,
    )

    stabilization_index = (
        max(action_history_cache_idx, action_history_db_idx)
        if action_history_cache_idx is not None and action_history_db_idx is not None
        else None
    )
    premature_silence = sum(
        1
        for index, action in enumerate(state.action_history)
        if action.get("action_type") == "SILENCE_ALERT"
        and action.get("target_service") in {"cache-service", "db-proxy"}
        and (stabilization_index is None or index < stabilization_index)
    )
    components["alert_hygiene"] = max(0.0, 0.05 - premature_silence * 0.05)

    recovery_complete_index = next(
        (
            index
            for index, action in enumerate(state.action_history)
            if action.get("critical_services_healthy")
        ),
        None,
    )
    post_recovery_actions = (
        len(state.action_history) - recovery_complete_index - 1
        if recovery_complete_index is not None
        else len(state.action_history)
    )
    post_recovery_segment = (
        state.action_history[recovery_complete_index + 1 :]
        if recovery_complete_index is not None
        else []
    )
    post_recovery_noops = sum(
        1
        for action in post_recovery_segment
        if action.get("action_type") in {"CHECK_LOGS", "INSPECT_SERVICE"}
    )

    if recovery_complete_index is not None and recovery_complete_index <= 9:
        components["efficiency"] = 0.10
    components["finish_quality"] = max(
        0.0,
        0.05
        - min(0.04, post_recovery_actions * 0.025)
        - min(0.04, post_recovery_noops * 0.035),
    )

    breakdown: dict[str, object] = dict(components)
    breakdown["restart_order"] = restart_order
    breakdown["cache_restarted_first"] = (
        cache_idx is not None and db_idx is not None and cache_idx < db_idx
    )
    breakdown["cache_restart_count"] = cache_restart_count
    breakdown["db_restart_count"] = db_restart_count
    breakdown["excess_critical_restarts"] = excess_critical_restarts
    breakdown["recovery_complete_index"] = recovery_complete_index
    breakdown["post_recovery_actions"] = post_recovery_actions
    breakdown["post_recovery_noops"] = post_recovery_noops

    score = sum(components.values())
    enterprise_component, enterprise = _enterprise_breakdown(state)
    score += enterprise_component
    breakdown.update(enterprise)
    ma_component, ma_breakdown = _multi_agent_breakdown(state)
    score += ma_component
    breakdown.update(ma_breakdown)
    return _validator_safe_score(score), breakdown


GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
    "expert": grade_expert,
    "enterprise": grade_expert,
}
