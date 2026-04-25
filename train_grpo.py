import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any

import requests
import torch
import wandb
from datasets import Dataset
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import unsloth
from unsloth import FastLanguageModel

from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer

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


def reset_env(session: requests.Session, env_url: str, timeout: float) -> dict[str, Any]:
    response = session.post(
        f"{env_url.rstrip('/')}/reset",
        json={"task_id": "enterprise"},
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


def make_format_validity_reward_function():
    def format_validity_reward(prompts: list[str], completions: list[Any], **_: Any) -> list[float]:
        rewards: list[float] = []
        for completion in completions:
            completion_text = _completion_to_text(completion)
            rewards.append(0.2 if _is_json_object_response(completion_text) else -0.2)
        return rewards

    return format_validity_reward


def make_action_validity_reward_function():
    def action_validity_reward(prompts: list[str], completions: list[Any], **_: Any) -> list[float]:
        rewards: list[float] = []
        for completion in completions:
            completion_text = _completion_to_text(completion)
            payload = parse_action_output(completion_text)
            rewards.append(0.3 if _validate_action_payload(payload) else -0.3)
        return rewards

    return action_validity_reward


def make_protocol_adherence_reward_function():
    def protocol_adherence_reward(prompts: list[str], completions: list[Any], **_: Any) -> list[float]:
        actions: list[dict[str, Any]] = []
        for completion in completions:
            completion_text = _completion_to_text(completion)
            actions.append(parse_action_output(completion_text))
        return _protocol_adherence_scores(actions)

    return protocol_adherence_reward


def build_prompt(observation: dict[str, Any]) -> str:
    compact_obs = json.dumps(observation, separators=(",", ":"), ensure_ascii=True)
    return (
        "You are an on-call SRE agent for an enterprise workflow incident. "
        "Return ONLY one JSON object and no markdown. "
        "Required keys: action_type, target_service. "
        "Allowed action_type values: CHECK_LOGS, INSPECT_SERVICE, DRAIN_TRAFFIC, RESTART_SERVICE, "
        "SCALE_UP, SCALE_DOWN, ROLLBACK, UPDATE_CONFIG, SILENCE_ALERT, "
        "ACKNOWLEDGE_PAGERDUTY, SEND_SLACK_MESSAGE, RESOLVE_PAGERDUTY.\\n"
        "Observation:\\n"
        f"{compact_obs}\\n"
        "Action JSON:"
    )


def init_unsloth_model(args: argparse.Namespace):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

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

        for completion in completions:
            try:
                reset_env(session, env_url, timeout)
            except Exception:
                rewards.append(-1.0)
                wandb.log(
                    {
                        "step_reward": -1.0,
                        "invalid_action_rate": 1.0,
                    }
                )
                continue

            completion_text = _completion_to_text(completion)
            action_payload, used_fallback = _parse_action_output_with_flag(completion_text)

            try:
                step_result = step_env(session, env_url, action_payload, timeout)
                step_reward = step_result.reward
            except Exception:
                step_reward = -1.0

            wandb.log(
                {
                    "step_reward": step_reward,
                    "invalid_action_rate": 1.0 if used_fallback else 0.0,
                }
            )
            rewards.append(step_reward)
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
    model, tokenizer = init_unsloth_model(args)

    env_reward_fn = make_env_reward_function(
        session=session,
        env_url=args.env_url,
        timeout=args.request_timeout,
        max_steps=args.max_steps,
    )
    format_reward_fn = make_format_validity_reward_function()
    action_reward_fn = make_action_validity_reward_function()
    protocol_reward_fn = make_protocol_adherence_reward_function()

    # Keep a minimal bootstrap dataset; we overwrite this each epoch.
    train_dataset = Dataset.from_dict({"prompt": ["Bootstrap prompt for GRPO trainer."]})

    grpo_config = GRPOConfig(
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.num_generations,
        gradient_accumulation_steps=1,
        num_generations=args.num_generations,
        max_completion_length=args.max_new_tokens,
        num_train_epochs=args.epochs,
        report_to="wandb",
        logging_steps=1,
        bf16=False,
        fp16=True,
        output_dir=args.output_dir,
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
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    return parser


if __name__ == "__main__":
    cli_args = build_arg_parser().parse_args()
    train(cli_args)
