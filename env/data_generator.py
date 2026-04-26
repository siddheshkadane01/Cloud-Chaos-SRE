import json
import random
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

SERVICES = [
    "api-gateway",
    "auth-service",
    "user-service",
    "order-service",
    "db-proxy",
    "cache-service",
]

SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"

SCENARIO_COUNTS = {
    "easy": 200,
    "medium": 150,
    "hard": 1,
    "expert": 1,
    "enterprise": 1,
}

TASK_SEED_OFFSETS = {
    "easy": 11,
    "medium": 29,
    "hard": 47,
    "expert": 71,
    "enterprise": 89,
}

BASELINE_STATE = {
    "api-gateway": {"cpu_pct": 40.0, "mem_pct": 50.0, "error_rate": 0.001, "latency_ms": 110.0},
    "auth-service": {"cpu_pct": 38.0, "mem_pct": 45.0, "error_rate": 0.001, "latency_ms": 95.0},
    "user-service": {"cpu_pct": 42.0, "mem_pct": 52.0, "error_rate": 0.001, "latency_ms": 120.0},
    "order-service": {"cpu_pct": 44.0, "mem_pct": 54.0, "error_rate": 0.001, "latency_ms": 130.0},
    "db-proxy": {"cpu_pct": 36.0, "mem_pct": 42.0, "error_rate": 0.001, "latency_ms": 50.0},
    "cache-service": {"cpu_pct": 28.0, "mem_pct": 32.0, "error_rate": 0.001, "latency_ms": 12.0},
}

BASE_CONFIG = {
    "db_timeout": 5000,
    "pool_size": 10,
    "replica_count": 1,
    "ttl": 300,
}


def _incident_context(
    incident_id: str,
    title: str,
    severity: str,
    business_service: str,
    customer_impact: str,
    symptom_summary: str,
    suspected_services: list[str],
    failure_mode: str,
    success_criteria: str,
) -> dict:
    return {
        "incident_id": incident_id,
        "title": title,
        "severity": severity,
        "business_service": business_service,
        "customer_impact": customer_impact,
        "symptom_summary": symptom_summary,
        "suspected_services": suspected_services,
        "failure_mode": failure_mode,
        "success_criteria": success_criteria,
    }


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _base_deploy_history(service: str, changes_second: dict) -> list[dict]:
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return [
        {
            "deploy_id": "d001",
            "timestamp": _iso(base),
            "service": service,
            "changes": {"pool_size": 10},
        },
        {
            "deploy_id": "d002",
            "timestamp": _iso(base + timedelta(hours=1)),
            "service": service,
            "changes": changes_second,
        },
    ]


# ---------------------------------------------------------------------------
# Easy — The Detective
# Rotate root cause across ALL 6 services (not just db-proxy).
# Add red-herring noise on 1-2 other services so the signal isn't trivial.
# ---------------------------------------------------------------------------

def _make_easy(index: int) -> dict:
    state = deepcopy(BASELINE_STATE)
    config = deepcopy(BASE_CONFIG)
    root = random.choice(SERVICES)

    # Root service is clearly degraded
    state[root]["cpu_pct"] = round(random.uniform(82.0, 97.0), 2)
    state[root]["mem_pct"] = round(random.uniform(78.0, 94.0), 2)
    state[root]["error_rate"] = round(random.uniform(0.20, 0.55), 4)
    state[root]["latency_ms"] = round(random.uniform(1200.0, 2500.0), 2)

    # 1-2 red-herring services with mild degradation (not root cause)
    red_herrings = [s for s in SERVICES if s != root]
    random.shuffle(red_herrings)
    for svc in red_herrings[: random.randint(1, 2)]:
        state[svc]["error_rate"] = round(
            min(0.09, state[svc]["error_rate"] + random.uniform(0.02, 0.07)), 4
        )
        state[svc]["latency_ms"] = round(
            min(700.0, state[svc]["latency_ms"] + random.uniform(80.0, 400.0)), 2
        )

    return {
        "scenario_id": f"easy-{index:03d}",
        "task_id": "easy",
        "initial_state": state,
        "initial_config": config,
        "deploy_history": _base_deploy_history(root, {"pool_size": random.choice([5, 6, 7])}),
        "incident_context": _incident_context(
            incident_id=f"INC-EASY-{index:03d}",
            title=f"Customer-facing latency spike traced to {root}",
            severity="SEV-2",
            business_service="Core platform APIs",
            customer_impact="Requests are timing out intermittently for a subset of users.",
            symptom_summary="PagerDuty fired on elevated latency and error rate after a routine service restart window.",
            suspected_services=[root] + red_herrings[:2],
            failure_mode="single-service degradation with noisy downstream alerts",
            success_criteria="Restore the primary failing service and leave the rest of the stack stable.",
        ),
        "ground_truth": {
            "root_cause_service": root,
            "correct_action": "RESTART_SERVICE",
            "correct_config_key": None,
            "correct_config_value": None,
        },
    }


# ---------------------------------------------------------------------------
# Medium — The First Responder
# Traffic spike on api-gateway, order-service, or auth-service.
# Always degrade 3-4 metrics on the hotspot.
# Add secondary pressure to cache-service when api-gateway is the hotspot.
# ---------------------------------------------------------------------------

def _make_medium(index: int) -> dict:
    state = deepcopy(BASELINE_STATE)
    config = deepcopy(BASE_CONFIG)
    hotspot = random.choice(["api-gateway", "order-service", "auth-service"])
    secondary = "cache-service"

    # Always degrade all four metrics on the hotspot for a genuine spike
    state[hotspot]["cpu_pct"] = round(random.uniform(88.0, 97.0), 2)
    state[hotspot]["mem_pct"] = round(random.uniform(85.0, 96.0), 2)
    state[hotspot]["error_rate"] = round(random.uniform(0.06, 0.22), 4)
    state[hotspot]["latency_ms"] = round(random.uniform(900.0, 1800.0), 2)

    # Secondary pressure — makes brute-force SCALE_UP of one service insufficient
    if hotspot == "api-gateway":
        state["cache-service"]["latency_ms"] = round(random.uniform(180.0, 450.0), 2)
        state["cache-service"]["error_rate"] = round(random.uniform(0.03, 0.09), 4)
        state["cache-service"]["mem_pct"] = round(random.uniform(78.0, 88.0), 2)
        secondary = "cache-service"
    elif hotspot == "order-service":
        state["db-proxy"]["latency_ms"] = round(random.uniform(180.0, 400.0), 2)
        state["db-proxy"]["cpu_pct"] = round(random.uniform(68.0, 82.0), 2)
        secondary = "db-proxy"
    elif hotspot == "auth-service":
        state["cache-service"]["error_rate"] = round(random.uniform(0.02, 0.07), 4)
        state["cache-service"]["latency_ms"] = round(random.uniform(220.0, 520.0), 2)
        secondary = "cache-service"

    return {
        "scenario_id": f"medium-{index:03d}",
        "task_id": "medium",
        "initial_state": state,
        "initial_config": config,
        "deploy_history": _base_deploy_history(hotspot, {"replica_count": random.choice([1, 2])}),
        "incident_context": _incident_context(
            incident_id=f"INC-MED-{index:03d}",
            title=f"Traffic surge saturating {hotspot}",
            severity="SEV-2",
            business_service="Checkout and account services",
            customer_impact="High request latency and elevated 5xx rate during a traffic surge.",
            symptom_summary="Autoscaling has not reacted fast enough and a dependency is beginning to flap.",
            suspected_services=[hotspot, secondary],
            failure_mode="capacity pressure with dependent-service stress",
            success_criteria="Stabilize all service health metrics without masking the incident with unnecessary actions.",
        ),
        "ground_truth": {
            "root_cause_service": hotspot,
            "secondary_cause_service": secondary,
            "correct_action": "SCALE_UP",
            "correct_config_key": None,
            "correct_config_value": None,
        },
    }


# ---------------------------------------------------------------------------
# Hard — The Architect
# db_timeout misconfiguration buried among 3 deploy events.
# Red-herring configs (retry_count, connection_pool) that look suspicious
# but are not the actual root cause.
# ---------------------------------------------------------------------------

def _make_hard(index: int) -> dict:
    state = deepcopy(BASELINE_STATE)
    config = deepcopy(BASE_CONFIG)
    root_key = random.choice(["db_timeout", "pool_size"])
    if root_key == "db_timeout":
        config["db_timeout"] = 100
        config["pool_size"] = random.choice([7, 8, 9])
        correct_value = 5000
    else:
        config["pool_size"] = random.choice([4, 5])
        config["db_timeout"] = random.choice([2500, 3000, 4000])
        correct_value = 10
    config["retry_count"] = random.choice([1, 2])
    config["connection_pool"] = random.choice([4, 5])
    config["ttl"] = random.choice([180, 240, 300])

    for svc in ["order-service", "user-service", "auth-service"]:
        state[svc]["error_rate"] = round(random.uniform(0.25, 0.55), 4)
        state[svc]["latency_ms"] = round(random.uniform(1200.0, 2600.0), 2)

    state["db-proxy"]["cpu_pct"] = round(random.uniform(80.0, 93.0), 2)
    state["db-proxy"]["mem_pct"] = round(random.uniform(75.0, 90.0), 2)
    state["db-proxy"]["error_rate"] = round(random.uniform(0.22, 0.45), 4)
    state["db-proxy"]["latency_ms"] = round(random.uniform(1400.0, 2600.0), 2)

    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    deploy_history = [
        # d001: innocuous pool tune
        {
            "deploy_id": "d001",
            "timestamp": _iso(base),
            "service": "db-proxy",
            "changes": {"pool_size": random.choice([10, 12])},
        },
        # d002: the bad deploy — db_timeout set too low
        {
            "deploy_id": "d002",
            "timestamp": _iso(base + timedelta(hours=1)),
            "service": "db-proxy",
            "changes": {root_key: config[root_key], "retry_count": config["retry_count"]},
        },
        # d003: a distracting red-herring deploy that changes a different key
        {
            "deploy_id": "d003",
            "timestamp": _iso(base + timedelta(hours=2)),
            "service": "db-proxy",
            "changes": {"connection_pool": config["connection_pool"], "ttl": config["ttl"]},
        },
    ]

    return {
        "scenario_id": f"hard-{index:03d}",
        "task_id": "hard",
        "initial_state": state,
        "initial_config": config,
        "deploy_history": deploy_history,
        "incident_context": _incident_context(
            incident_id=f"INC-HARD-{index:03d}",
            title="Database timeout regression after config rollout",
            severity="SEV-1",
            business_service="Order creation and profile reads",
            customer_impact="Checkout requests are timing out globally and queues are backing up.",
            symptom_summary="A recent database-side rollout introduced a subtle data-plane regression with multiple plausible config suspects.",
            suspected_services=["db-proxy", "order-service", "user-service"],
            failure_mode="hidden config regression with misleading deployment history",
            success_criteria="Identify the real config fault, restore the correct data-plane setting, and recover the data path cleanly.",
        ),
        "ground_truth": {
            "root_cause_service": "db-proxy",
            "correct_action": "UPDATE_CONFIG",
            "correct_config_key": root_key,
            "correct_config_value": correct_value,
            "misleading_action": "ROLLBACK",
        },
    }


# ---------------------------------------------------------------------------
# Expert — The Storm Chaser
# Dual failure: cache-service pool exhaustion AND db-proxy spike.
# Correct fix order: RESTART cache-service FIRST (its recovery eases db load),
# then RESTART db-proxy. Reverse order leaves db-proxy degraded.
# ---------------------------------------------------------------------------

def _make_expert(index: int) -> dict:
    state = deepcopy(BASELINE_STATE)
    config = deepcopy(BASE_CONFIG)

    # cache-service: pool exhaustion
    state["cache-service"]["cpu_pct"] = round(random.uniform(83.0, 96.0), 2)
    state["cache-service"]["mem_pct"] = round(random.uniform(86.0, 97.0), 2)
    state["cache-service"]["error_rate"] = round(random.uniform(0.18, 0.45), 4)
    state["cache-service"]["latency_ms"] = round(random.uniform(900.0, 2000.0), 2)

    # db-proxy: overloaded because cache is missed — cascades from cache
    state["db-proxy"]["cpu_pct"] = round(random.uniform(78.0, 92.0), 2)
    state["db-proxy"]["mem_pct"] = round(random.uniform(72.0, 88.0), 2)
    state["db-proxy"]["error_rate"] = round(random.uniform(0.12, 0.35), 4)
    state["db-proxy"]["latency_ms"] = round(random.uniform(800.0, 1800.0), 2)

    # Downstream: services depending on both are degraded
    for svc in ["auth-service", "user-service"]:
        state[svc]["error_rate"] = round(random.uniform(0.05, 0.18), 4)
        state[svc]["latency_ms"] = round(random.uniform(350.0, 900.0), 2)

    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    deploy_history = [
        {
            "deploy_id": "d001",
            "timestamp": _iso(base),
            "service": "cache-service",
            "changes": {"ttl": random.choice([240, 300])},
        },
        {
            "deploy_id": "d002",
            "timestamp": _iso(base + timedelta(minutes=30)),
            "service": "cache-service",
            "changes": {"pool_size": random.choice([4, 5])},
        },
        {
            "deploy_id": "d003",
            "timestamp": _iso(base + timedelta(hours=1)),
            "service": "db-proxy",
            "changes": {"replica_count": 1},
        },
    ]

    return {
        "scenario_id": f"expert-{index:03d}",
        "task_id": "expert",
        "initial_state": state,
        "initial_config": config,
        "deploy_history": deploy_history,
        "incident_context": _incident_context(
            incident_id=f"INC-EXP-{index:03d}",
            title="Cache pool exhaustion cascading into database saturation",
            severity="SEV-1",
            business_service="Checkout path and session validation",
            customer_impact="Users are seeing failed checkouts and repeated session retries across regions.",
            symptom_summary="Cache instability is driving database load, and fixing the wrong service first causes immediate relapse.",
            suspected_services=["cache-service", "db-proxy", "auth-service"],
            failure_mode="multi-stage cascade requiring ordered remediation",
            success_criteria="Recover cache first, then the database, while avoiding collateral restarts on healthy services.",
        ),
        "ground_truth": {
            "root_cause_service": "cache-service",
            "secondary_cause_service": "db-proxy",
            "correct_action": "RESTART_SERVICE",
            "correct_order": ["cache-service", "db-proxy"],
            "correct_config_key": None,
            "correct_config_value": None,
        },
    }


# ---------------------------------------------------------------------------
# Enterprise — Incident Commander
# Strict workflow: acknowledge -> notify -> remediate -> resolve.
# ---------------------------------------------------------------------------

def _make_enterprise(index: int) -> dict:
    state = deepcopy(BASELINE_STATE)
    config = deepcopy(BASE_CONFIG)

    # Keep enterprise-001 deterministic for protocol tests.
    if index == 1:
        root = "user-service"
        secondary = "db-proxy"
    else:
        root = random.choice(["user-service", "order-service", "auth-service"])
        secondary = "db-proxy" if root != "auth-service" else "cache-service"

    state[root]["cpu_pct"] = round(random.uniform(84.0, 96.0), 2)
    state[root]["mem_pct"] = round(random.uniform(80.0, 95.0), 2)
    state[root]["error_rate"] = round(random.uniform(0.16, 0.42), 4)
    state[root]["latency_ms"] = round(random.uniform(1000.0, 2200.0), 2)

    state[secondary]["cpu_pct"] = round(random.uniform(62.0, 84.0), 2)
    state[secondary]["mem_pct"] = round(random.uniform(60.0, 82.0), 2)
    state[secondary]["error_rate"] = round(random.uniform(0.03, 0.12), 4)
    state[secondary]["latency_ms"] = round(random.uniform(260.0, 760.0), 2)

    deploy_history = _base_deploy_history(root, {"replica_count": random.choice([1, 2])})

    incident_id = f"INC-ENT-{index:03d}"
    return {
        "scenario_id": f"enterprise-{index:03d}",
        "task_id": "enterprise",
        "initial_state": state,
        "initial_config": config,
        "deploy_history": deploy_history,
        "incident_context": _incident_context(
            incident_id=incident_id,
            title=f"Enterprise incident across {root}",
            severity="SEV-1",
            business_service="Payments and account workflows",
            customer_impact="Users are experiencing failed transactions and elevated 5xx responses.",
            symptom_summary="Paging fired for sustained service degradation requiring formal incident protocol.",
            suspected_services=[root, secondary],
            failure_mode="protocol-gated remediation with active incident management",
            success_criteria="Acknowledge PagerDuty, notify Slack, remediate root service, and resolve incident.",
        ),
        "ground_truth": {
            "root_cause_service": root,
            "secondary_cause_service": secondary,
            "correct_action": "RESTART_SERVICE",
            "correct_config_key": None,
            "correct_config_value": None,
        },
        "enterprise_workflow": {
            "enabled": True,
            "completion_rule": {"mode": "threshold", "value": 0.94},
            "required_sequence": [
                "ACKNOWLEDGE_PAGERDUTY",
                "SEND_SLACK_MESSAGE",
                "RESTART_SERVICE",
                "RESOLVE_PAGERDUTY",
            ],
        },
        "multi_agent": {
            "roles": [
                "incident_commander",
                "investigator",
                "remediator",
                "comms_officer",
            ],
            "initial_role": "incident_commander",
            "handoff_sequence": [
                "incident_commander",
                "investigator",
                "remediator",
                "comms_officer",
                "incident_commander",
            ],
            "permissions": {
                "incident_commander": [
                    "ACKNOWLEDGE_PAGERDUTY",
                    "RESOLVE_PAGERDUTY",
                    "CHECK_LOGS",
                    "INSPECT_SERVICE",
                ],
                "investigator": ["CHECK_LOGS", "INSPECT_SERVICE"],
                "remediator": [
                    "RESTART_SERVICE",
                    "SCALE_UP",
                    "SCALE_DOWN",
                    "DRAIN_TRAFFIC",
                    "ROLLBACK",
                    "UPDATE_CONFIG",
                    "SILENCE_ALERT",
                ],
                "comms_officer": ["SEND_SLACK_MESSAGE", "CHECK_LOGS", "INSPECT_SERVICE"],
            },
            "role_objectives": {
                "incident_commander": "Own incident lifecycle and state transitions.",
                "investigator": "Collect evidence and validate root cause.",
                "remediator": "Execute low-risk remediation actions on impacted services.",
                "comms_officer": "Broadcast clear updates and coordination messages.",
            },
        },
    }


def _scenario_signature(scenario: dict) -> str:
    payload = {
        key: value for key, value in scenario.items() if key not in {"scenario_id"}
    }
    incident_context = payload.get("incident_context", {})
    if isinstance(incident_context, dict):
        payload["incident_context"] = {
            key: value
            for key, value in incident_context.items()
            if key not in {"incident_id", "title"}
        }
    return json.dumps(payload, sort_keys=True)


def _build_unique_scenario(
    task: str,
    builder,
    index: int,
    seed: int,
    seen_signatures: set[str],
    max_retries: int = 200,
) -> dict:
    """Generate a scenario with unique learning content for the given task."""
    for attempt in range(max_retries):
        derived_seed = (
            seed * 1_000_000
            + TASK_SEED_OFFSETS[task] * 10_000
            + index * 100
            + attempt
        )
        random.seed(derived_seed)
        scenario = builder(index)
        signature = _scenario_signature(scenario)
        if signature not in seen_signatures:
            seen_signatures.add(signature)
            return scenario
    raise RuntimeError(
        f"Could not generate unique {task} scenario for index={index} after {max_retries} retries"
    )


def generate_all_scenarios(seed: int = 42):
    for task in ["easy", "medium", "hard", "expert", "enterprise"]:
        (SCENARIOS_DIR / task).mkdir(parents=True, exist_ok=True)

    generators = {
        "easy": _make_easy,
        "medium": _make_medium,
        "hard": _make_hard,
        "expert": _make_expert,
        "enterprise": _make_enterprise,
    }

    for task, builder in generators.items():
        seen_signatures: set[str] = set()
        for idx in range(1, SCENARIO_COUNTS[task] + 1):
            scenario = _build_unique_scenario(task, builder, idx, seed, seen_signatures)
            out_path = SCENARIOS_DIR / task / f"{scenario['scenario_id']}.json"
            out_path.write_text(json.dumps(scenario, indent=2))


if __name__ == "__main__":
    generate_all_scenarios()
    print(
        "Generated scenarios: "
        f"easy={SCENARIO_COUNTS['easy']}, "
        f"medium={SCENARIO_COUNTS['medium']}, "
        f"hard={SCENARIO_COUNTS['hard']}, "
        f"expert={SCENARIO_COUNTS['expert']}, "
        f"enterprise={SCENARIO_COUNTS['enterprise']}"
    )
