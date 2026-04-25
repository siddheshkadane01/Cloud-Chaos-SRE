import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
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

TRAIN_TASKS = ("easy", "medium", "hard", "expert", "enterprise")
TASK_SAMPLING_WEIGHTS = {
    "easy": 3,
    "medium": 2,
    "hard": 2,
    "expert": 1,
    "enterprise": 1,
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


def reset_env(
    session: requests.Session,
    env_url: str,
    timeout: float,
    task_id: str = "enterprise",
    scenario_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"task_id": task_id}
    if scenario_id:
        payload["scenario_id"] = scenario_id
    response = session.post(
        f"{env_url.rstrip('/')}/reset",
        json=payload,
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


def _extract_observation_from_prompt(prompt: str) -> dict[str, Any]:
    marker = "<|im_start|>user\n"
    start = prompt.find(marker)
    if start == -1:
        return {}
    start += len(marker)
    end = prompt.find("<|im_end|>", start)
    if end == -1:
        return {}
    blob = prompt[start:end].strip()
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


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
                payload = json.loads(json_blob)
            except json.JSONDecodeError:
                rewards.append(-0.3)
                continue
            if not isinstance(payload, dict):
                rewards.append(-0.3)
                continue

            required = int("action_type" in payload) + int("target_service" in payload)
            extra_text_penalty = -0.05 if completion_text.strip() != json_blob.strip() else 0.0
            score = -0.1 + required * 0.12 + extra_text_penalty
            rewards.append(round(max(-1.0, min(1.0, score)), 4))
        return rewards

    return format_validity_reward


def make_action_validity_reward_function():
    def action_validity_reward(prompts: list[str], completions: list[Any], **_: Any) -> list[float]:
        rewards: list[float] = []
        for completion in completions:
            completion_text = _completion_to_text(completion)
            payload, used_fallback = _parse_action_output_with_flag(completion_text)
            if used_fallback:
                rewards.append(-0.3)
                continue

            base = 0.15 if _validate_action_payload(payload) else -0.2
            action_type = payload.get("action_type")
            if action_type == "UPDATE_CONFIG":
                base += 0.05
            elif action_type in ENTERPRISE_ACTIONS:
                base += 0.03
            rewards.append(round(max(-1.0, min(1.0, base)), 4))
        return rewards

    return action_validity_reward


def make_protocol_adherence_reward_function():
    def protocol_adherence_reward(prompts: list[str], completions: list[Any], **_: Any) -> list[float]:
        rewards: list[float] = []
        for idx, completion in enumerate(completions):
            prompt = prompts[idx] if idx < len(prompts) else ""
            obs = _extract_observation_from_prompt(prompt)
            task_id = str(obs.get("task_id", ""))

            completion_text = _completion_to_text(completion)
            payload, used_fallback = _parse_action_output_with_flag(completion_text)
            action_type = payload.get("action_type")

            if used_fallback:
                rewards.append(-0.15)
                continue

            if task_id != "enterprise":
                # Non-enterprise tasks should favor infra actions and avoid enterprise APIs.
                if action_type in INFRA_ACTIONS:
                    rewards.append(0.05)
                else:
                    rewards.append(-0.1)
                continue

            protocol = obs.get("protocol_status", {})
            is_ack = bool(protocol.get("is_acknowledged", False))
            is_notified = bool(protocol.get("is_team_notified", False))

            reward = 0.0
            if action_type in INFRA_ACTIONS and not is_ack:
                reward -= 0.2
            elif action_type == "ACKNOWLEDGE_PAGERDUTY":
                reward += 0.25 if not is_ack else -0.1
            elif action_type == "SEND_SLACK_MESSAGE":
                reward += 0.25 if is_ack and not is_notified else -0.2
            elif action_type == "RESOLVE_PAGERDUTY":
                reward += 0.3 if is_ack and is_notified else -0.3
            elif action_type in INFRA_ACTIONS:
                reward += 0.1 if is_ack and is_notified else 0.0

            rewards.append(round(max(-1.0, min(1.0, reward)), 4))
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

    target_modules = _resolve_lora_target_modules(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def _resolve_lora_target_modules(model: torch.nn.Module) -> list[str]:
    # Preference order: decoder-only chat models first, then GPT2-style fallback.
    candidate_groups = [
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ["c_attn", "c_proj", "c_fc"],
        ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        ["Wqkv", "out_proj", "fc1", "fc2"],
    ]

    module_names = {name.split(".")[-1] for name, _ in model.named_modules()}
    for group in candidate_groups:
        selected = [name for name in group if name in module_names]
        if selected:
            return selected

    # Final fallback: pick the most frequent linear-like leaf modules.
    linear_leaf_counts: dict[str, int] = {}
    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        cls = module.__class__.__name__.lower()
        if "linear" in cls or leaf.startswith("fc"):
            linear_leaf_counts[leaf] = linear_leaf_counts.get(leaf, 0) + 1

    if linear_leaf_counts:
        ranked = sorted(linear_leaf_counts.items(), key=lambda item: item[1], reverse=True)
        return [name for name, _ in ranked[:6]]

    raise RuntimeError("Could not resolve LoRA target modules for the selected model architecture")


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
):
    def env_reward_func(prompts: list[str], completions: list[Any], **kwargs: Any) -> list[float]:
        rewards: list[float] = []
        task_ids = kwargs.get("task_id")
        scenario_ids = kwargs.get("scenario_id")

        if not isinstance(task_ids, list):
            task_ids = [None] * len(completions)
        if not isinstance(scenario_ids, list):
            scenario_ids = [None] * len(completions)

        for idx, completion in enumerate(completions):
            prompt = prompts[idx] if idx < len(prompts) else ""
            prompt_obs = _extract_observation_from_prompt(prompt)
            task_id = task_ids[idx] if idx < len(task_ids) else None
            scenario_id = scenario_ids[idx] if idx < len(scenario_ids) else None

            resolved_task = str(task_id or prompt_obs.get("task_id") or "enterprise")
            if resolved_task not in TRAIN_TASKS:
                resolved_task = "enterprise"
            resolved_scenario = scenario_id or prompt_obs.get("scenario_id")

            try:
                reset_env(
                    session,
                    env_url,
                    timeout,
                    task_id=resolved_task,
                    scenario_id=resolved_scenario,
                )
            except Exception:
                rewards.append(-1.0)
                continue

            completion_text = _completion_to_text(completion)
            action_payload, used_fallback = _parse_action_output_with_flag(completion_text)
            step_result: EnvStepResult | None = None

            try:
                step_result = step_env(session, env_url, action_payload, timeout)
                step_reward = step_result.reward
            except Exception:
                step_reward = -1.0

            if used_fallback:
                step_reward -= 0.15
            if step_result is not None and step_result.info.get("action_valid") is False:
                step_reward -= 0.1
            rewards.append(round(max(-1.0, min(1.0, step_reward)), 4))
        return rewards

    return env_reward_func


def _estimate_service_health(service_state: dict[str, Any]) -> float:
    cpu = float(service_state.get("cpu_pct", 100.0))
    mem = float(service_state.get("mem_pct", 100.0))
    err = float(service_state.get("error_rate", 1.0))
    lat = float(service_state.get("latency_ms", 5000.0))

    cpu_score = max(0.0, min(1.0, (70.0 - cpu) / 30.0 + 1.0))
    mem_score = max(0.0, min(1.0, (80.0 - mem) / 20.0 + 1.0))
    err_score = max(0.0, min(1.0, 1.0 - err / 1.0))
    lat_score = max(0.0, min(1.0, (200.0 - lat) / 1800.0 + 1.0))
    return round((cpu_score + mem_score + err_score + lat_score) / 4.0, 4)


def _build_dataset_from_scenarios(max_scenarios_per_task: int) -> Dataset:
    prompts: list[str] = []
    task_ids: list[str] = []
    scenario_ids: list[str] = []
    root = Path("scenarios")

    for task_id in TRAIN_TASKS:
        scenario_paths = sorted((root / task_id).glob("*.json"))[:max_scenarios_per_task]
        for path in scenario_paths:
            scenario = json.loads(path.read_text())
            initial_state = scenario.get("initial_state", {})
            services: dict[str, dict[str, Any]] = {}
            for service_name, metrics in initial_state.items():
                if not isinstance(metrics, dict):
                    continue
                service_blob = dict(metrics)
                service_blob["health"] = _estimate_service_health(metrics)
                services[service_name] = service_blob

            gt = scenario.get("ground_truth", {})
            alerts = []
            root_cause = gt.get("root_cause_service")
            if root_cause:
                alerts.append({"type": "ROOT_CAUSE", "service": root_cause})
            secondary = gt.get("secondary_cause_service")
            if secondary:
                alerts.append({"type": "SECONDARY", "service": secondary})

            observation = {
                "task_id": task_id,
                "scenario_id": scenario.get("scenario_id", path.stem),
                "step": 0,
                "services": services,
                "alerts": alerts,
                "current_config": scenario.get("initial_config", {}),
                "incident_context": scenario.get("incident_context", {}),
            }

            observation["protocol_status"] = {
                "is_acknowledged": False,
                "is_team_notified": False,
                "is_resolved": False,
            }

            repeat_count = TASK_SAMPLING_WEIGHTS.get(task_id, 1)
            for _ in range(repeat_count):
                prompts.append(build_prompt(observation))
                task_ids.append(task_id)
                scenario_ids.append(observation["scenario_id"])

    if not prompts:
        raise RuntimeError("No training prompts could be built from scenarios/")

    return Dataset.from_dict(
        {
            "prompt": prompts,
            "task_id": task_ids,
            "scenario_id": scenario_ids,
        }
    )


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    # Avoid interactive wandb login prompts during local dry-runs.
    os.environ["WANDB_MODE"] = args.wandb_mode

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
            "num_generations": args.num_generations,
            "logging_steps": args.logging_steps,
            "wandb_mode": args.wandb_mode,
        },
    )

    session = build_http_session()
    model, tokenizer = init_model(args)

    env_reward_fn = make_env_reward_function(
        session=session,
        env_url=args.env_url,
        timeout=args.request_timeout,
    )
    format_reward_fn = make_format_validity_reward_function()
    action_reward_fn = make_action_validity_reward_function()
    protocol_reward_fn = make_protocol_adherence_reward_function()

    train_dataset = _build_dataset_from_scenarios(args.max_scenarios_per_task)

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.num_generations,
        gradient_accumulation_steps=1,
        num_generations=args.num_generations,
        max_completion_length=args.max_new_tokens,
        num_train_epochs=args.epochs,
        report_to="wandb",
        logging_steps=args.logging_steps,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        max_grad_norm=0.5,
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
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--max_scenarios_per_task", type=int, default=10)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--wandb_mode", type=str, default="offline", choices=["offline", "online", "disabled"])
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    return parser


if __name__ == "__main__":
    cli_args = build_arg_parser().parse_args()
    train(cli_args)
