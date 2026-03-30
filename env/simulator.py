import copy
from datetime import datetime, timezone

from .models import Alert, HealthSummary, LogEntry, SystemMetrics

SERVICES = [
    "api-gateway",
    "auth-service",
    "user-service",
    "order-service",
    "db-proxy",
    "cache-service",
]

SERVICE_GRAPH = {
    "api-gateway": ["auth-service", "order-service", "user-service"],
    "auth-service": ["db-proxy", "cache-service"],
    "user-service": ["db-proxy", "cache-service"],
    "order-service": ["db-proxy"],
    "db-proxy": [],
    "cache-service": [],
}

HEALTH_THRESHOLDS = {
    "cpu_pct": 70.0,
    "mem_pct": 80.0,
    "error_rate": 0.01,
    "latency_ms": 200.0,
}


class VirtualDataCentre:
    """Pure in-memory simulation of a 6-service data centre."""

    def __init__(self, scenario: dict):
        self.scenario = copy.deepcopy(scenario)
        self.state = copy.deepcopy(scenario["initial_state"])
        self.config = copy.deepcopy(scenario["initial_config"])
        self.replicas = {s: 1 for s in SERVICES}
        self.deploy_history = copy.deepcopy(scenario.get("deploy_history", []))
        self.alerts: list[Alert] = []
        self.logs: list[LogEntry] = []
        self._refresh_alerts()

    def get_metrics(self) -> SystemMetrics:
        return SystemMetrics(
            cpu_pct={s: self.state[s]["cpu_pct"] for s in SERVICES},
            mem_pct={s: self.state[s]["mem_pct"] for s in SERVICES},
            error_rate={s: self.state[s]["error_rate"] for s in SERVICES},
            latency_ms={s: self.state[s]["latency_ms"] for s in SERVICES},
            timestamp=datetime.now(timezone.utc),
        )

    def health_score(self) -> HealthSummary:
        scores: dict[str, float] = {}
        for s in SERVICES:
            st = self.state[s]
            cpu_score = max(0.0, min(1.0, (HEALTH_THRESHOLDS["cpu_pct"] - st["cpu_pct"]) / 30.0 + 1.0))
            mem_score = max(0.0, min(1.0, (HEALTH_THRESHOLDS["mem_pct"] - st["mem_pct"]) / 20.0 + 1.0))
            err_score = max(0.0, min(1.0, 1.0 - st["error_rate"] / 1.0))
            lat_score = max(
                0.0,
                min(1.0, (HEALTH_THRESHOLDS["latency_ms"] - st["latency_ms"]) / 1800.0 + 1.0),
            )
            scores[s] = round((cpu_score + mem_score + err_score + lat_score) / 4.0, 4)

        overall = round(sum(scores.values()) / len(scores), 4)
        return HealthSummary(per_service=scores, overall=overall)

    def is_healthy(self) -> bool:
        for s in SERVICES:
            st = self.state[s]
            if (
                st["cpu_pct"] >= HEALTH_THRESHOLDS["cpu_pct"]
                or st["mem_pct"] >= HEALTH_THRESHOLDS["mem_pct"]
                or st["error_rate"] >= HEALTH_THRESHOLDS["error_rate"]
                or st["latency_ms"] >= HEALTH_THRESHOLDS["latency_ms"]
            ):
                return False
        return True

    def apply_action(self, action_type: str, target: str, config_key=None, config_value=None) -> dict:
        """Apply an action to the simulation. Returns result info dict."""
        result = {"valid": True, "changed": False, "details": ""}

        if target not in SERVICES:
            result["valid"] = False
            result["details"] = f"Unknown service: {target}"
            return result

        if action_type == "CHECK_LOGS":
            # Add deterministic diagnostic clues for hard scenarios.
            if target == "db-proxy" and int(self.config.get("db_timeout", 5000)) < 500:
                self._add_log(
                    target,
                    "ERROR",
                    f"Database timeout misconfigured: db_timeout={self.config.get('db_timeout')}ms causing upstream request failures",
                )
                result["details"] = "Retrieved logs for db-proxy: db timeout appears misconfigured"
            elif target in ["order-service", "user-service", "auth-service"] and int(self.config.get("db_timeout", 5000)) < 500:
                self._add_log(
                    target,
                    "WARN",
                    "Frequent dependency timeout from db-proxy; check database client timeout configuration",
                )
                result["details"] = f"Retrieved logs for {target}: dependency timeout errors observed"
            else:
                self._add_log(target, "INFO", f"Log fetch requested for {target}")
                result["details"] = f"Retrieved logs for {target}"

        elif action_type == "INSPECT_SERVICE":
            if target == "db-proxy":
                result["details"] = f"Inspected db-proxy: {self.state[target]}, db_timeout={self.config.get('db_timeout')}"
            else:
                result["details"] = f"Inspected {target}: {self.state[target]}"

        elif action_type == "RESTART_SERVICE":
            svc = self.state[target]
            svc["cpu_pct"] = max(5.0, svc["cpu_pct"] * 0.4)
            svc["mem_pct"] = max(5.0, svc["mem_pct"] * 0.3)
            svc["error_rate"] = max(0.0, svc["error_rate"] * 0.2)
            svc["latency_ms"] = max(10.0, svc["latency_ms"] * 0.5)
            self._propagate_recovery(target)
            self._add_log(target, "INFO", f"{target} restarted")
            result["changed"] = True

        elif action_type == "SCALE_UP":
            if self.replicas[target] >= 5:
                result["valid"] = False
                result["details"] = f"{target} already at max replicas (5)"
                return result
            self.replicas[target] += 1
            svc = self.state[target]
            svc["cpu_pct"] = max(5.0, svc["cpu_pct"] * 0.7)
            svc["latency_ms"] = max(10.0, svc["latency_ms"] * 0.8)
            result["changed"] = True
            result["details"] = f"{target} scaled to {self.replicas[target]} replicas"

        elif action_type == "SCALE_DOWN":
            if self.replicas[target] <= 1:
                result["valid"] = False
                result["details"] = f"{target} already at minimum replicas (1)"
                return result
            self.replicas[target] -= 1
            svc = self.state[target]
            svc["cpu_pct"] = min(100.0, svc["cpu_pct"] * 1.3)
            svc["latency_ms"] = min(5000.0, svc["latency_ms"] * 1.2)
            result["changed"] = True

        elif action_type == "ROLLBACK":
            if len(self.deploy_history) >= 2:
                prev = self.deploy_history[-2]
                for k, v in prev.get("changes", {}).items():
                    if k in self.config:
                        self.config[k] = v
                self._apply_config_effects()
                self._add_log(target, "INFO", f"Rolled back {target} to deploy {prev['deploy_id']}")
                result["changed"] = True
            else:
                result["details"] = "No previous deployment to roll back to"

        elif action_type == "UPDATE_CONFIG":
            if not config_key or config_value is None:
                result["valid"] = False
                result["details"] = "UPDATE_CONFIG requires config_key and config_value"
                return result
            self.config[config_key] = config_value
            self._apply_config_effects()
            self._add_log(target, "INFO", f"Config updated: {config_key}={config_value}")
            result["changed"] = True
            result["details"] = f"Set {config_key}={config_value}"

        elif action_type == "SILENCE_ALERT":
            for alert in self.alerts:
                if alert.service == target and not alert.silenced:
                    alert.silenced = True
            result["details"] = f"Silenced alerts for {target}"

        self._refresh_alerts()
        return result

    def _propagate_recovery(self, service: str):
        """When a service recovers, partially improve its dependents."""
        for svc, deps in SERVICE_GRAPH.items():
            if service in deps:
                self.state[svc]["error_rate"] = max(0.0, self.state[svc]["error_rate"] * 0.7)
                self.state[svc]["latency_ms"] = max(10.0, self.state[svc]["latency_ms"] * 0.85)

    def _apply_config_effects(self):
        """Apply config changes to service state key mechanic for Task 3."""
        db_timeout = self.config.get("db_timeout", 5000)
        if db_timeout < 500:
            for svc in ["order-service", "user-service", "auth-service"]:
                self.state[svc]["error_rate"] = min(1.0, self.state[svc]["error_rate"] + 0.35)
                self.state[svc]["latency_ms"] = min(5000.0, self.state[svc]["latency_ms"] * 2.2)
            self.state["db-proxy"]["error_rate"] = min(1.0, self.state["db-proxy"]["error_rate"] + 0.20)
            self.state["db-proxy"]["latency_ms"] = min(5000.0, self.state["db-proxy"]["latency_ms"] * 1.6)
            self.state["api-gateway"]["latency_ms"] = min(5000.0, self.state["api-gateway"]["latency_ms"] * 1.25)
        else:
            for svc in ["order-service", "user-service", "auth-service"]:
                self.state[svc]["error_rate"] = max(0.0, self.state[svc]["error_rate"] * 0.25)
                self.state[svc]["latency_ms"] = max(40.0, self.state[svc]["latency_ms"] * 0.30)
            self.state["db-proxy"]["error_rate"] = max(0.0, self.state["db-proxy"]["error_rate"] * 0.35)
            self.state["db-proxy"]["latency_ms"] = max(30.0, self.state["db-proxy"]["latency_ms"] * 0.35)
            self.state["api-gateway"]["latency_ms"] = max(40.0, self.state["api-gateway"]["latency_ms"] * 0.75)

    def _add_log(self, service: str, severity: str, message: str):
        self.logs.append(
            LogEntry(
                timestamp=datetime.now(timezone.utc),
                service=service,
                severity=severity,
                message=message,
            )
        )
        self.logs = self.logs[-10:]

    def _refresh_alerts(self):
        self.alerts = []
        for svc in SERVICES:
            st = self.state[svc]
            if st["cpu_pct"] >= HEALTH_THRESHOLDS["cpu_pct"]:
                self.alerts.append(
                    Alert(
                        alert_id=f"{svc}-cpu",
                        service=svc,
                        metric="cpu_pct",
                        threshold=HEALTH_THRESHOLDS["cpu_pct"],
                        current=st["cpu_pct"],
                        severity="WARN" if st["cpu_pct"] < 90 else "CRITICAL",
                        triggered_at=datetime.now(timezone.utc),
                    )
                )
            if st["error_rate"] >= HEALTH_THRESHOLDS["error_rate"]:
                self.alerts.append(
                    Alert(
                        alert_id=f"{svc}-err",
                        service=svc,
                        metric="error_rate",
                        threshold=HEALTH_THRESHOLDS["error_rate"],
                        current=st["error_rate"],
                        severity="CRITICAL" if st["error_rate"] > 0.1 else "WARN",
                        triggered_at=datetime.now(timezone.utc),
                    )
                )
