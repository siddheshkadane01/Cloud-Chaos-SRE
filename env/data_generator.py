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

BASELINE_STATE = {
    "api-gateway": {"cpu_pct": 45.0, "mem_pct": 55.0, "error_rate": 0.001, "latency_ms": 120.0},
    "auth-service": {"cpu_pct": 40.0, "mem_pct": 50.0, "error_rate": 0.001, "latency_ms": 100.0},
    "user-service": {"cpu_pct": 72.0, "mem_pct": 68.0, "error_rate": 0.15, "latency_ms": 850.0},
    "order-service": {"cpu_pct": 70.0, "mem_pct": 65.0, "error_rate": 0.12, "latency_ms": 780.0},
    "db-proxy": {"cpu_pct": 92.0, "mem_pct": 88.0, "error_rate": 0.45, "latency_ms": 2100.0},
    "cache-service": {"cpu_pct": 30.0, "mem_pct": 35.0, "error_rate": 0.001, "latency_ms": 15.0},
}

BASE_CONFIG = {
    "db_timeout": 5000,
    "pool_size": 10,
    "replica_count": 1,
    "ttl": 300,
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


def _make_easy(index: int) -> dict:
    state = deepcopy(BASELINE_STATE)
    config = deepcopy(BASE_CONFIG)
    root = random.choice(SERVICES)

    state[root]["cpu_pct"] = round(random.uniform(82.0, 98.0), 2)
    state[root]["mem_pct"] = round(random.uniform(78.0, 95.0), 2)
    state[root]["error_rate"] = round(random.uniform(0.20, 0.60), 4)
    state[root]["latency_ms"] = round(random.uniform(1200.0, 2600.0), 2)

    for service in SERVICES:
        if service == root:
            continue
        if random.random() < 0.4:
            state[service]["error_rate"] = round(min(1.0, state[service]["error_rate"] + random.uniform(0.02, 0.12)), 4)
            state[service]["latency_ms"] = round(min(5000.0, state[service]["latency_ms"] + random.uniform(100.0, 700.0)), 2)

    return {
        "scenario_id": f"easy-{index:03d}",
        "task_id": "easy",
        "initial_state": state,
        "initial_config": config,
        "deploy_history": _base_deploy_history(root, {"pool_size": random.choice([5, 6, 7])}),
        "ground_truth": {
            "root_cause_service": root,
            "correct_action": "RESTART_SERVICE",
            "correct_config_key": None,
            "correct_config_value": None,
        },
    }


def _make_medium(index: int) -> dict:
    state = deepcopy(BASELINE_STATE)
    config = deepcopy(BASE_CONFIG)
    hotspot = random.choice(["api-gateway", "order-service", "auth-service"])

    worst_metrics = random.sample(["cpu_pct", "mem_pct", "error_rate", "latency_ms"], k=random.choice([2, 3, 4]))
    for metric in worst_metrics:
        if metric == "cpu_pct":
            state[hotspot][metric] = round(random.uniform(88.0, 98.0), 2)
        elif metric == "mem_pct":
            state[hotspot][metric] = round(random.uniform(86.0, 97.0), 2)
        elif metric == "error_rate":
            state[hotspot][metric] = round(random.uniform(0.06, 0.25), 4)
        elif metric == "latency_ms":
            state[hotspot][metric] = round(random.uniform(900.0, 1900.0), 2)

    if hotspot == "api-gateway":
        state["cache-service"]["latency_ms"] = round(random.uniform(120.0, 350.0), 2)
        state["cache-service"]["error_rate"] = round(random.uniform(0.02, 0.08), 4)

    return {
        "scenario_id": f"medium-{index:03d}",
        "task_id": "medium",
        "initial_state": state,
        "initial_config": config,
        "deploy_history": _base_deploy_history(hotspot, {"replica_count": random.choice([1, 2])}),
        "ground_truth": {
            "root_cause_service": hotspot,
            "correct_action": "SCALE_UP",
            "correct_config_key": None,
            "correct_config_value": None,
        },
    }


def _make_hard(index: int) -> dict:
    state = deepcopy(BASELINE_STATE)
    config = deepcopy(BASE_CONFIG)

    config["db_timeout"] = 100
    config["pool_size"] = random.choice([8, 10, 12])
    config["ttl"] = random.choice([240, 300, 360])

    for svc in ["order-service", "user-service", "auth-service"]:
        state[svc]["error_rate"] = round(random.uniform(0.25, 0.55), 4)
        state[svc]["latency_ms"] = round(random.uniform(1200.0, 2600.0), 2)

    state["db-proxy"]["cpu_pct"] = round(random.uniform(80.0, 93.0), 2)
    state["db-proxy"]["mem_pct"] = round(random.uniform(75.0, 90.0), 2)
    state["db-proxy"]["error_rate"] = round(random.uniform(0.22, 0.45), 4)
    state["db-proxy"]["latency_ms"] = round(random.uniform(1400.0, 2600.0), 2)

    deploy_history = [
        {
            "deploy_id": "d001",
            "timestamp": "2025-01-01T00:00:00Z",
            "service": "db-proxy",
            "changes": {"pool_size": random.choice([10, 12])},
        },
        {
            "deploy_id": "d002",
            "timestamp": "2025-01-01T01:00:00Z",
            "service": "db-proxy",
            "changes": {"db_timeout": 100},
        },
        {
            "deploy_id": "d003",
            "timestamp": "2025-01-01T02:00:00Z",
            "service": "db-proxy",
            "changes": {"ttl": config["ttl"]},
        },
    ]

    return {
        "scenario_id": f"hard-{index:03d}",
        "task_id": "hard",
        "initial_state": state,
        "initial_config": config,
        "deploy_history": deploy_history,
        "ground_truth": {
            "root_cause_service": "db-proxy",
            "correct_action": "UPDATE_CONFIG",
            "correct_config_key": "db_timeout",
            "correct_config_value": 5000,
        },
    }


def generate_all_scenarios(seed: int = 42):
    random.seed(seed)
    for task in ["easy", "medium", "hard"]:
        (SCENARIOS_DIR / task).mkdir(parents=True, exist_ok=True)

    generators = {
        "easy": _make_easy,
        "medium": _make_medium,
        "hard": _make_hard,
    }

    for task, builder in generators.items():
        for idx in range(1, 11):
            scenario = builder(idx)
            out_path = SCENARIOS_DIR / task / f"{scenario['scenario_id']}.json"
            out_path.write_text(json.dumps(scenario, indent=2))


if __name__ == "__main__":
    generate_all_scenarios()
    print("Generated scenarios: easy=10, medium=10, hard=10")
