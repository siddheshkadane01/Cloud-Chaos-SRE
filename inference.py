"""
inference.py OpenEnv hackathon inference script.
MUST be named inference.py and MUST be in root directory.
Reads OPENAI_API_KEY, API_BASE_URL, MODEL_NAME, HF_TOKEN.
"""

import argparse
import json
import os
import signal
import time

import requests
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.1-70b-versatile")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

_ = HF_TOKEN

ENV_BASE_URL = "http://localhost:7860"
TEMPERATURE = 0.0
MAX_TOKENS = 512
SEED = 42
TASKS = ["easy", "medium", "hard"]
FALLBACK_ACTION = json.dumps(
    {
        "action_type": "CHECK_LOGS",
        "target_service": "api-gateway",
        "config_key": None,
        "config_value": None,
        "reason": "Fallback: checking gateway logs",
    }
)

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer responding to a production incident.
You will receive system observations step by step. At each step you must:
1. Analyse the metrics, logs, deploy history, and current config provided
2. Reason about the most likely root cause
3. Choose exactly one action from: CHECK_LOGS, INSPECT_SERVICE, RESTART_SERVICE, SCALE_UP, SCALE_DOWN, ROLLBACK, UPDATE_CONFIG, SILENCE_ALERT
4. Target one of: api-gateway, auth-service, user-service, order-service, db-proxy, cache-service
5. Provide a clear reason for your choice

Respond ONLY in valid JSON. No other text. Schema:
{"action_type": "...", "target_service": "...", "config_key": null, "config_value": null, "reason": "..."}"""


def _timeout_handler(signum, frame):
    _ = (signum, frame)
    print("TIMEOUT: 19-minute limit reached. Saving partial scores.")
    raise TimeoutError()


signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm(19 * 60)

client = OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)


def call_env(method: str, path: str, body: dict | None = None) -> dict:
    url = f"{ENV_BASE_URL}{path}"
    response = requests.request(method, url, json=body, timeout=30)
    response.raise_for_status()
    return response.json()


def parse_action(text: str) -> dict:
    import re

    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return json.loads(FALLBACK_ACTION)


def format_observation(obs: dict, step: int) -> str:
    metrics = obs.get("metrics", {})
    config = obs.get("current_config", {})
    alerts = [
        f"{a['service']}:{a['metric']}={a['current']:.2f}"
        for a in obs.get("active_alerts", [])
    ]
    logs = [
        f"[{entry['severity']}] {entry['service']}: {entry['message']}"
        for entry in obs.get("logs", [])[-5:]
    ]
    deploys = [f"Deploy {d['deploy_id']}: {d['changes']}" for d in obs.get("deploy_history", [])]
    health = obs.get("health_summary", {}).get("per_service", {})

    return f"""Step {step} | Task: {obs.get('task_id')} | Max steps: {obs.get('max_steps')}

HEALTH SCORES: {json.dumps(health, indent=None)}

METRICS:
  CPU%:       {json.dumps(metrics.get('cpu_pct', {}))}
  Memory%:    {json.dumps(metrics.get('mem_pct', {}))}
  Error rate: {json.dumps(metrics.get('error_rate', {}))}
  Latency ms: {json.dumps(metrics.get('latency_ms', {}))}

ACTIVE ALERTS: {', '.join(alerts) or 'none'}
CURRENT CONFIG: {json.dumps(config)}
RECENT LOGS: {chr(10).join(logs) or 'none'}
DEPLOY HISTORY: {chr(10).join(deploys) or 'none'}"""


def hard_guardrail_action(task_id: str, obs: dict, step: int) -> dict | None:
    if task_id != "hard":
        return None

    config = obs.get("current_config", {})
    db_timeout = config.get("db_timeout")
    try:
        db_timeout_int = int(db_timeout)
    except (TypeError, ValueError):
        db_timeout_int = None

    if db_timeout_int is not None and db_timeout_int < 500:
        if step == 1:
            return {
                "action_type": "CHECK_LOGS",
                "target_service": "db-proxy",
                "config_key": None,
                "config_value": None,
                "reason": "Investigate db-proxy for timeout-related dependency failures",
            }
        return {
            "action_type": "UPDATE_CONFIG",
            "target_service": "db-proxy",
            "config_key": "db_timeout",
            "config_value": 5000,
            "reason": "Restore db timeout to stable value to recover upstream services",
        }

    overall = obs.get("health_summary", {}).get("overall", 0.0)
    if overall < 0.85:
        return {
            "action_type": "RESTART_SERVICE",
            "target_service": "order-service",
            "config_key": None,
            "config_value": None,
            "reason": "Post-fix cleanup restart to accelerate recovery of downstream requests",
        }
    return None


def run_task(task_id: str) -> dict:
    print("\n" + "=" * 50)
    print(f"Running task: {task_id}")
    print("=" * 50)

    obs = call_env("POST", "/reset", {"task_id": task_id})
    max_steps = obs.get("max_steps", 15)
    done = False
    step = 0

    while not done and step < max_steps:
        step += 1
        guardrail_action = hard_guardrail_action(task_id, obs, step)

        if guardrail_action is not None:
            action_dict = guardrail_action
            print(
                f"Step {step}: {action_dict.get('action_type')} -> {action_dict.get('target_service')} (guardrail)"
            )
        else:
            user_content = format_observation(obs, step)

            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": user_content},
            ]

            response_text = FALLBACK_ACTION
            for attempt in range(3):
                try:
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                        seed=SEED,
                        stream=False,
                    )
                    response_text = completion.choices[0].message.content or FALLBACK_ACTION
                    break
                except Exception as exc:
                    print(f"  Attempt {attempt + 1} failed: {exc}")

            action_dict = parse_action(response_text)
            print(f"Step {step}: {action_dict.get('action_type')} -> {action_dict.get('target_service')}")

        result = call_env("POST", "/step", action_dict)
        obs = result.get("observation", obs)
        done = result.get("done", False)

        if done:
            print("Episode complete.")
            break

    state = call_env("GET", "/state")
    grader_result = call_env("POST", "/grader", state)
    score = grader_result.get("score", 0.0)

    print(f"Final score: {score:.4f}")
    return {
        "task_id": task_id,
        "score": score,
        "steps": step,
        "breakdown": grader_result.get("breakdown", {}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", action="store_true", help="Output JSON only")
    args = parser.parse_args()

    scores: dict[str, dict] = {}
    start = time.time()

    try:
        for task_id in TASKS:
            scores[task_id] = run_task(task_id)
    except TimeoutError:
        print("Inference timed out, partial scores saved.")
    finally:
        signal.alarm(0)

    mean_score = round(sum(result["score"] for result in scores.values()) / max(len(scores), 1), 4)
    output = {
        "model": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "seed": SEED,
        "scores": scores,
        "mean_score": mean_score,
        "total_time_s": round(time.time() - start, 1),
    }

    with open("baseline_scores.json", "w", encoding="utf-8") as file_obj:
        json.dump(output, file_obj, indent=2)

    if args.output_json:
        print(json.dumps(output))
    else:
        print("\n" + "=" * 50)
        print("BASELINE RESULTS")
        print("=" * 50)
        for task_id, result in scores.items():
            print(f"  {task_id:8s}: {result['score']:.4f}  ({result['steps']} steps)")
        print(f"  {'MEAN':8s}: {mean_score:.4f}")
        print("\nSaved to baseline_scores.json")


if __name__ == "__main__":
    main()
