import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any
import inspect

import requests
import torch
import wandb
from peft import LoraConfig
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig
from urllib3.util.retry import Retry

try:
    from trl import (
        AutoModelForCausalLMWithValueHead,
        PPOConfig,
        PPOTrainer,
        create_reference_model,
    )
except ImportError:
    from trl.experimental.ppo import (  # type: ignore
        AutoModelForCausalLMWithValueHead,
        PPOConfig,
        PPOTrainer,
    )
    from trl.models import create_reference_model  # type: ignore


def _has_step_api() -> bool:
    return hasattr(PPOTrainer, "step")

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

    # Handle markdown fences like ```json { ... } ```
    fenced_match = re.search(r"```(?:json)?\\s*(\{.*?\})\\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced_match:
        return fenced_match.group(1).strip()

    # Fallback: extract first balanced JSON object from free text.
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
    # Include target_service to satisfy strict API schema and avoid 422 errors.
    return {"action_type": "CHECK_LOGS", "target_service": "user-service"}


def parse_action_output(text: str) -> dict[str, Any]:
    json_blob = extract_json_object(text)
    if json_blob is None:
        return fallback_action()

    try:
        parsed = json.loads(json_blob)
    except json.JSONDecodeError:
        return fallback_action()

    if not isinstance(parsed, dict):
        return fallback_action()

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

    return action


def build_prompt(observation: dict[str, Any]) -> str:
    compact_obs = json.dumps(observation, separators=(",", ":"), ensure_ascii=True)
    return (
        "You are an on-call SRE agent for an enterprise workflow incident. "
        "Return ONLY one JSON object and no markdown. "
        "Required keys: action_type, target_service. "
        "Allowed action_type values: CHECK_LOGS, INSPECT_SERVICE, DRAIN_TRAFFIC, RESTART_SERVICE, "
        "SCALE_UP, SCALE_DOWN, ROLLBACK, UPDATE_CONFIG, SILENCE_ALERT, "
        "ACKNOWLEDGE_PAGERDUTY, SEND_SLACK_MESSAGE, RESOLVE_PAGERDUTY.\n"
        "Observation:\n"
        f"{compact_obs}\n"
        "Action JSON:"
    )


def init_model(model_name: str):
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs: dict[str, Any] = {
        "peft_config": peft_config,
        "device_map": "auto",
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    }

    if torch.cuda.is_available():
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name, **model_kwargs)
    ref_model = create_reference_model(model)
    return model, ref_model, tokenizer


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    # Avoid interactive wandb login prompts during local dry-runs.
    os.environ.setdefault("WANDB_MODE", "offline")

    wandb.init(
        project=os.getenv("WANDB_PROJECT", "openenv-enterprise-ppo"),
        name=os.getenv("WANDB_RUN_NAME", f"ppo-enterprise-{args.seed}"),
        config={
            "model_name": args.model_name,
            "env_url": args.env_url,
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "quantization": "8bit",
            "adapter": "lora",
        },
    )

    session = build_http_session()
    model, ref_model, tokenizer = init_model(args.model_name)

    ppo_trainer = None
    use_step_updates = _has_step_api()

    if use_step_updates:
        ppo_cfg_kwargs = {
            "learning_rate": args.learning_rate,
            "batch_size": 1,
            "mini_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "log_with": None,
        }

        # Keep compatibility with older/newer config signatures.
        ppo_sig = inspect.signature(PPOConfig.__init__)
        ppo_cfg_kwargs = {
            key: value for key, value in ppo_cfg_kwargs.items() if key in ppo_sig.parameters
        }

        ppo_config = PPOConfig(**ppo_cfg_kwargs)

        trainer_kwargs = {
            "config": ppo_config,
            "model": model,
            "ref_model": ref_model,
            "tokenizer": tokenizer,
        }

        trainer_sig = inspect.signature(PPOTrainer.__init__)
        trainer_kwargs = {
            key: value for key, value in trainer_kwargs.items() if key in trainer_sig.parameters
        }
        ppo_trainer = PPOTrainer(**trainer_kwargs)
    else:
        wandb.log(
            {
                "compat_warning": (
                    "Installed TRL exposes experimental PPOTrainer without step(). "
                    "Running environment wiring loop with no-op policy updates."
                )
            }
        )

    os.makedirs(args.output_dir, exist_ok=True)

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if ppo_trainer is not None and hasattr(ppo_trainer, "accelerator"):
        device = ppo_trainer.accelerator.device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_step = 0

    for epoch in tqdm(range(args.epochs), desc="PPO Epochs"):
        try:
            observation = reset_env(session, args.env_url, timeout=args.request_timeout)
        except Exception as exc:
            wandb.log({"epoch": epoch, "reset_error": str(exc)})
            time.sleep(1.0)
            continue

        episode_rewards: list[float] = []
        done = False

        for step_idx in range(args.max_steps):
            prompt = build_prompt(observation)
            query_tensor = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_prompt_length,
            ).input_ids[0].to(device)

            try:
                if ppo_trainer is not None and hasattr(ppo_trainer, "generate"):
                    response_tensor = ppo_trainer.generate(query_tensor, return_prompt=False, **generation_kwargs)
                else:
                    response_tensor = model.generate(query_tensor.unsqueeze(0), **generation_kwargs)[0]
                if response_tensor.dim() > 1:
                    response_tensor = response_tensor[0]
                response_text = tokenizer.decode(response_tensor, skip_special_tokens=True)
            except Exception as exc:
                wandb.log({"epoch": epoch, "step": step_idx, "generation_error": str(exc)})
                response_text = ""
                response_tensor = tokenizer(
                    json.dumps(fallback_action()),
                    return_tensors="pt",
                ).input_ids[0].to(device)

            action_payload = parse_action_output(response_text)

            try:
                step_result = step_env(
                    session,
                    args.env_url,
                    action_payload,
                    timeout=args.request_timeout,
                )
            except Exception as exc:
                wandb.log({"epoch": epoch, "step": step_idx, "step_error": str(exc)})
                step_result = EnvStepResult(
                    observation=observation,
                    reward=-1.0,
                    done=True,
                    info={"error": str(exc)},
                )

            reward_tensor = torch.tensor(step_result.reward, dtype=torch.float32, device=device)

            if use_step_updates and ppo_trainer is not None:
                try:
                    ppo_trainer.step([query_tensor], [response_tensor], [reward_tensor])
                except Exception as exc:
                    wandb.log({"epoch": epoch, "step": step_idx, "ppo_error": str(exc)})

            episode_rewards.append(step_result.reward)
            global_step += 1
            mean_reward = sum(episode_rewards) / len(episode_rewards)

            wandb.log(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "step_index": step_idx,
                    "step_reward": step_result.reward,
                    "mean_reward": mean_reward,
                    "episode_length": step_idx + 1,
                }
            )

            observation = step_result.observation
            done = step_result.done
            if done:
                break

        final_mean_reward = sum(episode_rewards) / max(1, len(episode_rewards))
        wandb.log(
            {
                "epoch": epoch,
                "mean_reward": final_mean_reward,
                "episode_length": len(episode_rewards),
                "episode_done": done,
            }
        )

    if ppo_trainer is not None and hasattr(ppo_trainer, "model"):
        ppo_trainer.model.save_pretrained(args.output_dir)
    else:
        model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    wandb.finish()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PPO training for enterprise OpenEnv workflow")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--env_url", type=str, default="http://localhost:7860")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=25)
    parser.add_argument("--output_dir", type=str, default="./artifacts/ppo-enterprise-llama3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--request_timeout", type=float, default=20.0)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    return parser


if __name__ == "__main__":
    cli_args = build_arg_parser().parse_args()
    train(cli_args)
