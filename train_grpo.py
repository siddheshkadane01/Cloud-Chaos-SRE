import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Any

import requests
import torch
import wandb
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from requests.adapters import HTTPAdapter
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from urllib3.util.retry import Retry

from trl import GRPOConfig, GRPOTrainer

ACTION_TYPES = {
    "CHECK_LOGS",
    "INSPECT_SERVICE",
    "DRAIN_TRAFFIC",
    "RESTART_SERVICE",
    "SCALE_UP",
    "SCALE_DOWN",
    "ROLLBACK",
    "UPDATE_CONFIG",
    "SILENCE_ALERT",
    "ACKNOWLEDGE_PAGERDUTY",
    "SEND_SLACK_MESSAGE",
    "RESOLVE_PAGERDUTY",
}

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

VALID_SERVICES = {
    "api-gateway",
    "auth-service",
    "user-service",
    "order-service",
    "db-proxy",
    "cache-service",
}


@dataclass
class EnvStepResult:
    observation: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_http_session(total_retries: int = 5, backoff_factor: float = 0.5) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        status=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[408, 429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def reset_env(session: requests.Session, env_url: str, timeout: float, seed: int | None = None) -> dict[str, Any]:
    response = session.post(
        f"{env_url.rstrip('/')}/reset",
        json={
            "task_id": "enterprise",
            "deterministic": False,
            "evaluation_mode": False,
            "seed": seed if seed is not None else random.randint(1, 10_000_000),
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Invalid /reset response format")
    return payload


def step_env(
    session: requests.Session,
    env_url: str,
    action: dict[str, Any],
    timeout: float,
) -> EnvStepResult:
    response = session.post(
        f"{env_url.rstrip('/')}/step",
        json=action,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Invalid /step response format")

    reward_payload = payload.get("reward", {})
    reward_value = 0.0
    if isinstance(reward_payload, dict):
        reward_value = float(reward_payload.get("step_reward", 0.0))
    elif isinstance(reward_payload, (float, int)):
        reward_value = float(reward_payload)

    observation = payload.get("observation", {})
    info = payload.get("info", {})

    return EnvStepResult(
        observation=observation if isinstance(observation, dict) else {},
        reward=reward_value,
        done=bool(payload.get("done", False)),
        info=info if isinstance(info, dict) else {},
    )


def extract_json_object(text: str) -> str | None:
    if not text:
        return None

    fenced_match = re.search(r"```(?:json)?\\s*(\{.*?\})\\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced_match:
        return fenced_match.group(1).strip()

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1].strip()
    return None


def fallback_action() -> dict[str, Any]:
    # Keep target_service to satisfy strict FastAPI schema and avoid 422 failures.
    return {"action_type": "CHECK_LOGS", "target_service": "user-service"}


def _parse_action_output_with_flag(text: str) -> tuple[dict[str, Any], bool]:
    json_blob = extract_json_object(text)
    if json_blob is None:
        return fallback_action(), True

    try:
        parsed = json.loads(json_blob)
    except json.JSONDecodeError:
        return fallback_action(), True

    if not isinstance(parsed, dict):
        return fallback_action(), True

    action_type = str(parsed.get("action_type", "CHECK_LOGS")).upper()
    target_service = str(parsed.get("target_service", "user-service"))

    if action_type not in ACTION_TYPES:
        action_type = "CHECK_LOGS"
    if target_service not in VALID_SERVICES:
        target_service = "user-service"

    action: dict[str, Any] = {
        "action_type": action_type,
        "target_service": target_service,
    }

    for optional_key in [
        "config_key",
        "config_value",
        "reason",
        "incident_id",
        "channel_name",
        "message_text",
        "params",
    ]:
        if optional_key in parsed:
            action[optional_key] = parsed[optional_key]

    return action, False


def parse_action_output(text: str) -> dict[str, Any]:
    action, _ = _parse_action_output_with_flag(text)
    return action


def _is_json_object_response(text: str) -> bool:
    json_blob = extract_json_object(text)
    if json_blob is None:
        return False
    try:
        parsed = json.loads(json_blob)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict)


def _validate_action_payload(payload: dict[str, Any]) -> bool:
    action_type = payload.get("action_type")
    target_service = payload.get("target_service")
    if action_type not in ACTION_TYPES:
        return False
    if target_service not in VALID_SERVICES:
        return False

    if action_type == "UPDATE_CONFIG":
        if payload.get("config_key") is None or payload.get("config_value") is None:
            return False

    if action_type in {"ACKNOWLEDGE_PAGERDUTY", "RESOLVE_PAGERDUTY"}:
        if not payload.get("incident_id") and not payload.get("params", {}).get("incident_id"):
            return False

    if action_type == "SEND_SLACK_MESSAGE":
        channel = payload.get("channel_name") or payload.get("params", {}).get("channel_name")
        message = payload.get("message_text") or payload.get("params", {}).get("message_text")
        if not channel or not message:
            return False

    return True


def _protocol_adherence_scores(actions: list[dict[str, Any]]) -> list[float]:
    scores: list[float] = []
    is_acknowledged = False
    is_notified = False
    infra_done = False
    is_resolved = False

    for payload in actions:
        action_type = payload.get("action_type")
        reward = 0.0

        if action_type in INFRA_ACTIONS and not is_acknowledged:
            reward -= 0.2

        if action_type == "ACKNOWLEDGE_PAGERDUTY":
            if not is_acknowledged and not is_notified and not infra_done and not is_resolved:
                is_acknowledged = True
                reward += 0.25
            else:
                reward -= 0.15
        elif action_type == "SEND_SLACK_MESSAGE":
            if is_acknowledged and not is_notified and not is_resolved:
                is_notified = True
                reward += 0.25
            else:
                reward -= 0.2
        elif action_type == "RESOLVE_PAGERDUTY":
            if is_acknowledged and is_notified and infra_done and not is_resolved:
                is_resolved = True
                reward += 0.3
            else:
                reward -= 0.3
        elif action_type in INFRA_ACTIONS:
            if is_acknowledged and is_notified and not is_resolved and not infra_done:
                infra_done = True
                reward += 0.2

        scores.append(round(max(-1.0, min(1.0, reward)), 4))

    return scores


def _extract_prompt_observation(prompt: str) -> dict[str, Any]:
    start = prompt.find("<|im_start|>user\n")
    end = prompt.find("<|im_end|>", start + 1)
    if start == -1 or end == -1:
        return {}
    payload = prompt[start + len("<|im_start|>user\n") : end].strip()
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _group_size(prompts: list[str], completions: list[Any]) -> int:
    if not prompts:
        return 1
    return max(1, len(completions) // len(prompts))


def _safe_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean_v = sum(values) / len(values)
    var = sum((v - mean_v) ** 2 for v in values) / len(values)
    return math.sqrt(var)


def _get_service_metrics_from_obs(obs: dict[str, Any], service: str) -> tuple[float, float, float]:
    # Returns health, latency, error_rate.
    default_health = 0.5
    default_latency = 1000.0
    default_error = 0.2

    services = obs.get("services", {})
    if isinstance(services, dict) and service in services:
        details = services.get(service, {})
        if isinstance(details, dict):
            return (
                float(details.get("health", default_health)),
                float(details.get("latency_ms", default_latency)),
                float(details.get("error_rate", default_error)),
            )

    metrics = obs.get("metrics", {})
    health_summary = obs.get("health_summary", {})
    health_map = health_summary.get("per_service", {}) if isinstance(health_summary, dict) else {}

    latency_map = metrics.get("latency_ms", {}) if isinstance(metrics, dict) else {}
    error_map = metrics.get("error_rate", {}) if isinstance(metrics, dict) else {}

    return (
        float(health_map.get(service, default_health)) if isinstance(health_map, dict) else default_health,
        float(latency_map.get(service, default_latency)) if isinstance(latency_map, dict) else default_latency,
        float(error_map.get(service, default_error)) if isinstance(error_map, dict) else default_error,
    )


def _count_open_alerts(obs: dict[str, Any]) -> int:
    alerts = obs.get("active_alerts")
    if not isinstance(alerts, list):
        alerts = obs.get("alerts")
    if not isinstance(alerts, list):
        return 0
    return sum(1 for alert in alerts if not isinstance(alert, dict) or not alert.get("silenced", False))


def make_format_validity_reward_function():
    def format_validity_reward(prompts: list[str], completions: list[Any], **_: Any) -> list[float]:
        rewards: list[float] = []
        for completion in completions:
            completion_text = _completion_to_text(completion)
            json_blob = extract_json_object(completion_text)
            if json_blob is None:
                rewards.append(-0.35)
                continue
            try:
                parsed = json.loads(json_blob)
            except json.JSONDecodeError:
                rewards.append(-0.25)
                continue

            if not isinstance(parsed, dict):
                rewards.append(-0.2)
                continue

            score = 0.08
            if "action_type" in parsed:
                score += 0.06
            if "target_service" in parsed:
                score += 0.06
            if parsed.get("reason"):
                score += 0.04
            rewards.append(round(max(-0.4, min(0.3, score)), 4))
        return rewards

    return format_validity_reward


def make_action_validity_reward_function():
    def action_validity_reward(prompts: list[str], completions: list[Any], **_: Any) -> list[float]:
        rewards: list[float] = []
        for completion in completions:
            completion_text = _completion_to_text(completion)
            payload = parse_action_output(completion_text)
            if not _validate_action_payload(payload):
                rewards.append(-0.45)
                continue

            score = 0.12
            action_type = payload.get("action_type")
            if action_type in {"CHECK_LOGS", "INSPECT_SERVICE"}:
                score += 0.05
            if action_type in {"RESTART_SERVICE", "SCALE_UP", "DRAIN_TRAFFIC", "ROLLBACK"}:
                score += 0.08
            if action_type == "UPDATE_CONFIG" and payload.get("config_key"):
                score += 0.06
            rewards.append(round(max(-0.5, min(0.35, score)), 4))
        return rewards

    return action_validity_reward


def make_protocol_adherence_reward_function():
    def protocol_adherence_reward(prompts: list[str], completions: list[Any], **_: Any) -> list[float]:
        rewards: list[float] = []
        group_n = _group_size(prompts, completions)

        for idx, completion in enumerate(completions):
            prompt_idx = min(len(prompts) - 1, idx // group_n) if prompts else 0
            prompt_obs = _extract_prompt_observation(prompts[prompt_idx]) if prompts else {}

            completion_text = _completion_to_text(completion)
            action = parse_action_output(completion_text)
            action_type = action.get("action_type")

            protocol = prompt_obs.get("protocol_status", {}) if isinstance(prompt_obs, dict) else {}
            is_ack = bool(protocol.get("is_acknowledged", False))
            is_notified = bool(protocol.get("is_team_notified", False))
            is_resolved = bool(protocol.get("is_resolved", False))

            score = 0.0
            if action_type in INFRA_ACTIONS and not is_ack:
                score -= 0.2
            if action_type == "ACKNOWLEDGE_PAGERDUTY":
                score += 0.2 if not is_ack else -0.15
            elif action_type == "SEND_SLACK_MESSAGE":
                if is_ack and not is_notified:
                    score += 0.2
                else:
                    score -= 0.18
            elif action_type == "RESOLVE_PAGERDUTY":
                if is_ack and is_notified and not is_resolved:
                    score += 0.25
                else:
                    score -= 0.28

            rewards.append(round(max(-0.5, min(0.35, score)), 4))

        # Repeated actions in the same prompt group get penalized to prevent collapse.
        for prompt_idx in range(len(prompts)):
            start = prompt_idx * group_n
            end = min(start + group_n, len(completions))
            if start >= end:
                continue
            group_actions = [parse_action_output(_completion_to_text(c)) for c in completions[start:end]]
            counts: dict[tuple[str, str], int] = {}
            for action in group_actions:
                key = (str(action.get("action_type")), str(action.get("target_service")))
                counts[key] = counts.get(key, 0) + 1

            for local_idx, action in enumerate(group_actions):
                key = (str(action.get("action_type")), str(action.get("target_service")))
                freq = counts.get(key, 1)
                if freq > 1:
                    rewards[start + local_idx] = round(rewards[start + local_idx] - 0.08 * (freq - 1), 4)

        return rewards

    return protocol_adherence_reward


def build_prompt(observation: dict[str, Any]) -> str:
    compact_obs = json.dumps(
        observation, separators=(",", ":"), ensure_ascii=True
    )
    return (
        "<|im_start|>system\n"
        "You are an expert on-call SRE agent responding to a "
        "production incident.\n"
        "You must respond with ONLY a single JSON object. "
        "No explanation, no markdown, no extra text.\n"
        "Required keys:\n"
        "  action_type: one of CHECK_LOGS, INSPECT_SERVICE, "
        "DRAIN_TRAFFIC, RESTART_SERVICE, SCALE_UP, SCALE_DOWN, "
        "ROLLBACK, UPDATE_CONFIG, SILENCE_ALERT, "
        "ACKNOWLEDGE_PAGERDUTY, SEND_SLACK_MESSAGE, "
        "RESOLVE_PAGERDUTY\n"
        "  target_service: one of api-gateway, auth-service, "
        "user-service, order-service, db-proxy, cache-service\n"
        "Example: {\"action_type\": \"CHECK_LOGS\", "
        "\"target_service\": \"api-gateway\"}"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{compact_obs}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def init_model(args: argparse.Namespace):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj","k_proj","v_proj","o_proj",
            "gate_proj","up_proj","down_proj",
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def _completion_to_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, list):
        # Chat format from TRL can be [{"role":"assistant","content":"..."}] per sample.
        chunks: list[str] = []
        for part in item:
            if isinstance(part, dict):
                chunks.append(str(part.get("content", "")))
            else:
                chunks.append(str(part))
        return "\n".join(chunks)
    if isinstance(item, dict):
        return str(item.get("content", ""))
    return str(item)


def make_env_reward_function(
    session: requests.Session,
    env_url: str,
    timeout: float,
    max_steps: int,
):
    def env_reward_func(prompts: list[str], completions: list[Any], **_: Any) -> list[float]:
        rewards: list[float] = []
        sampled_actions: list[dict[str, Any]] = []
        group_n = _group_size(prompts, completions)

        current_group = -1
        group_seed = None

        for idx, completion in enumerate(completions):
            prompt_idx = min(len(prompts) - 1, idx // group_n) if prompts else 0
            # Regenerate group_seed only when a new group starts
            if prompt_idx != current_group:
                current_group = prompt_idx
                group_seed = random.randint(1, 10_000_000)
            prompt_obs = _extract_prompt_observation(prompts[prompt_idx]) if prompts else {}
            try:
                reset_env(
                    session,
                    env_url,
                    timeout,
                    seed=group_seed
                )
            except Exception:
                rewards.append(-1.2)
                sampled_actions.append(fallback_action())
                continue

            completion_text = _completion_to_text(completion)
            action_payload, used_fallback = _parse_action_output_with_flag(completion_text)
            sampled_actions.append(action_payload)

            is_valid_action = _validate_action_payload(action_payload)

            try:
                step_result = step_env(session, env_url, action_payload, timeout)
                env_step_reward = step_result.reward
            except Exception:
                env_step_reward = -1.0
                step_result = EnvStepResult(observation={}, reward=env_step_reward, done=False, info={})

            target_service = str(action_payload.get("target_service", "user-service"))
            pre_health, pre_latency, pre_error = _get_service_metrics_from_obs(prompt_obs, target_service)
            post_health, post_latency, post_error = _get_service_metrics_from_obs(step_result.observation, target_service)

            pre_overall = pre_health
            if isinstance(prompt_obs.get("health_summary"), dict):
                pre_overall = float(prompt_obs["health_summary"].get("overall", pre_health))
            post_overall = float(
                step_result.observation.get("health_summary", {}).get("overall", post_health)
            )

            health_improvement = post_overall - pre_overall
            service_health_improvement = post_health - pre_health
            latency_improvement = (pre_latency - post_latency) / max(100.0, pre_latency)
            error_improvement = pre_error - post_error

            pre_open_alerts = _count_open_alerts(prompt_obs)
            post_open_alerts = _count_open_alerts(step_result.observation)
            if pre_open_alerts > 0:
                progress_alerts = (pre_open_alerts - post_open_alerts) / pre_open_alerts
            else:
                progress_alerts = 0.0
            task_progress = 0.5 * service_health_improvement + 0.3 * latency_improvement + 0.2 * progress_alerts

            invalid_penalty = 0.0
            if used_fallback or not is_valid_action:
                invalid_penalty -= 0.45

            protocol = prompt_obs.get("protocol_status", {}) if isinstance(prompt_obs, dict) else {}
            is_ack = bool(protocol.get("is_acknowledged", False))
            is_notified = bool(protocol.get("is_team_notified", False))
            ordering_penalty = 0.0
            action_type = action_payload.get("action_type")
            if action_type in INFRA_ACTIONS and not is_ack:
                ordering_penalty -= 0.12
            if action_type == "SEND_SLACK_MESSAGE" and not is_ack:
                ordering_penalty -= 0.18
            if action_type == "RESOLVE_PAGERDUTY" and (not is_ack or not is_notified):
                ordering_penalty -= 0.28

            shaped_reward = (
                env_step_reward * 1.0
                + health_improvement * 2.0
                + latency_improvement * 1.5
                + error_improvement * 1.5
                + task_progress * 2.0
                + invalid_penalty
                + ordering_penalty
            )

            # Entropy penalty: avoid repeated same action per group
            if idx > 0:
                prev_action = sampled_actions[-2]
                if prev_action.get("action_type") == action_type and prev_action.get("target_service") == action_payload.get("target_service"):
                    shaped_reward -= 0.1

            rewards.append(round(max(-1.5, min(1.5, shaped_reward)), 4))

        # Penalize repeated actions in each same-state group to maintain diversity.
        for prompt_idx in range(len(prompts)):
            start = prompt_idx * group_n
            end = min(start + group_n, len(sampled_actions))
            if start >= end:
                continue
            counts: dict[tuple[str, str], int] = {}
            for action in sampled_actions[start:end]:
                key = (str(action.get("action_type")), str(action.get("target_service")))
                counts[key] = counts.get(key, 0) + 1

            for local_idx, action in enumerate(sampled_actions[start:end]):
                key = (str(action.get("action_type")), str(action.get("target_service")))
                freq = counts.get(key, 1)
                if freq > 1:
                    rewards[start + local_idx] = round(rewards[start + local_idx] - 0.10 * (freq - 1), 4)

        if rewards:
            global_std = _safe_std(rewards)
            group_stds: list[float] = []
            for prompt_idx in range(len(prompts)):
                start = prompt_idx * group_n
                end = min(start + group_n, len(rewards))
                if end - start > 1:
                    group_stds.append(_safe_std(rewards[start:end]))

            wandb.log(
                {
                    "step_reward": float(sum(rewards) / len(rewards)),
                    "reward_std_local": global_std,
                    "group_reward_std_mean": float(sum(group_stds) / len(group_stds)) if group_stds else 0.0,
                    "invalid_action_rate": float(
                        sum(1 for action in sampled_actions if not _validate_action_payload(action)) / max(len(sampled_actions), 1)
                    ),
                }
            )

            # Mandatory variance debug: show rewards for several samples from same state.
            if len(prompts) > 0 and group_n >= 4:
                for prompt_idx in range(min(2, len(prompts))):
                    start = prompt_idx * group_n
                    end = min(start + group_n, len(rewards))
                    if end - start < 4:
                        continue
                    group_actions = sampled_actions[start:end]
                    group_rewards = rewards[start:end]
                    print(
                        "[variance-debug]",
                        {
                            "prompt_index": prompt_idx,
                            "actions": [
                                {
                                    "action_type": action.get("action_type"),
                                    "target_service": action.get("target_service"),
                                }
                                for action in group_actions
                            ],
                            "rewards": group_rewards,
                            "std": round(_safe_std(group_rewards), 6),
                        },
                    )
        return rewards

    return env_reward_func


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    # Avoid interactive wandb login prompts during local dry-runs.
    os.environ.setdefault("WANDB_MODE", "offline")

    wandb.init(
        project=os.getenv("WANDB_PROJECT", "openenv-enterprise-grpo"),
        name=os.getenv("WANDB_RUN_NAME", f"grpo-enterprise-{args.seed}"),
        config={
            "model_name": args.model_name,
            "env_url": args.env_url,
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "quantization": "4bit",
            "adapter": "lora",
            "algorithm": "grpo",
        },
    )

    session = build_http_session()
    model, tokenizer = init_model(args)

    # Force exploratory decoding for GRPO rollouts.
    model_any: Any = model
    generation_config = getattr(model_any, "generation_config", None)
    if generation_config is not None:
        generation_config.do_sample = True
        generation_config.temperature = args.temperature
        generation_config.top_p = args.top_p
        generation_config.pad_token_id = tokenizer.pad_token_id
        generation_config.eos_token_id = tokenizer.eos_token_id

    env_reward_fn = make_env_reward_function(
        session=session,
        env_url=args.env_url,
        timeout=args.request_timeout,
        max_steps=args.max_steps,
    )
    format_reward_fn = make_format_validity_reward_function()
    action_reward_fn = make_action_validity_reward_function()
    protocol_reward_fn = make_protocol_adherence_reward_function()

    obs_templates: list[dict[str, Any]] = []
    for _ in range(args.dataset_size):
        try:
            obs_templates.append(reset_env(session, args.env_url, args.request_timeout))
        except Exception:
            # Keep a tiny fallback set so training can still proceed if env reset is flaky.
            obs_templates.append(
                {
                    "services": {
                        "api-gateway": {"health": random.uniform(0.1, 0.6), "latency_ms": random.randint(300, 1800), "error_rate": random.uniform(0.02, 0.3)}
                    },
                    "alerts": [{"type": "DEGRADED", "service": "api-gateway"}],
                    "step": 0,
                }
            )
    random.shuffle(obs_templates)

    # Ensure num_generations is valid for GRPO constraints
    batch_size = max(1, args.per_device_train_batch_size)
    num_generations = min(args.num_generations, batch_size)

    # Make sure batch_size % num_generations == 0
    if batch_size % num_generations != 0:
        import math
        gcd_val = math.gcd(batch_size, num_generations)
        num_generations = gcd_val if gcd_val > 0 else 1

    print("[DEBUG] final num_generations:", num_generations)
    print("[DEBUG] batch_size:", batch_size)
    train_dataset = Dataset.from_dict(
        {"prompt": [build_prompt(obs) for obs in obs_templates]}
    )

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=max(1, args.per_device_train_batch_size),
        gradient_accumulation_steps=1,
        num_generations=num_generations,
        max_completion_length=args.max_new_tokens,
        max_prompt_length=args.max_prompt_length,
        num_train_epochs=args.epochs,
        report_to="wandb",
        logging_steps=1,
        bf16=False,
        fp16=True,
    )

    grpo_trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            env_reward_fn,
            format_reward_fn,
            action_reward_fn,
            protocol_reward_fn,
        ],
        args=grpo_config,
        train_dataset=train_dataset,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        grpo_trainer.train()
    except Exception as exc:
        wandb.log({"grpo_error": str(exc)})

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    wandb.finish()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GRPO training for enterprise OpenEnv workflow")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--env_url", type=str, default="http://localhost:7860")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=25)
    parser.add_argument("--output_dir", type=str, default="./artifacts/grpo-enterprise-llama3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--request_timeout", type=float, default=20.0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.92)
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--dataset_size", type=int, default=64)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    return parser


if __name__ == "__main__":
    cli_args = build_arg_parser().parse_args()
    train(cli_args)
