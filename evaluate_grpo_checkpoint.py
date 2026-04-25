import argparse
import json
from pathlib import Path
from typing import Any

import requests
import torch
from peft import PeftModel
from requests.adapters import HTTPAdapter
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from urllib3.util.retry import Retry

from train_grpo import build_prompt, parse_action_output

TRAIN_TASKS = ("easy", "medium", "hard", "expert", "enterprise")


def build_http_session(total_retries: int = 4, backoff_factor: float = 0.3) -> requests.Session:
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
    task_id: str,
    scenario_id: str,
    seed: int,
) -> dict[str, Any]:
    response = session.post(
        f"{env_url.rstrip('/')}/reset",
        json={
            "task_id": task_id,
            "scenario_id": scenario_id,
            "seed": seed,
            "deterministic": True,
            "evaluation_mode": True,
            "mode": "single_agent",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def step_env(
    session: requests.Session,
    env_url: str,
    action: dict[str, Any],
) -> tuple[dict[str, Any], float, bool]:
    response = session.post(
        f"{env_url.rstrip('/')}/step",
        json=action,
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        return {}, -1.0, True

    reward_payload = payload.get("reward", {})
    reward_value = 0.0
    if isinstance(reward_payload, dict):
        reward_value = float(reward_payload.get("step_reward", 0.0))
    elif isinstance(reward_payload, (float, int)):
        reward_value = float(reward_payload)

    obs = payload.get("observation", {})
    done = bool(payload.get("done", False))
    return (obs if isinstance(obs, dict) else {}), reward_value, done


def grade_current_episode(session: requests.Session, env_url: str) -> tuple[float, dict[str, Any]]:
    state_response = session.get(f"{env_url.rstrip('/')}/state", timeout=30)
    state_response.raise_for_status()
    episode_state = state_response.json()

    grade_response = session.post(
        f"{env_url.rstrip('/')}/grader",
        json=episode_state,
        timeout=30,
    )
    grade_response.raise_for_status()
    grade_payload = grade_response.json()
    score = float(grade_payload.get("score", 0.0))
    breakdown = grade_payload.get("breakdown", {})
    return score, breakdown if isinstance(breakdown, dict) else {}


def load_policy(
    model_name: str,
    adapter_path: str | None,
    load_in_4bit: bool,
):
    model_kwargs: dict[str, Any] = {"device_map": "auto"}
    if torch.cuda.is_available() and load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model.eval()
    return model, tokenizer


def list_scenarios(task_id: str, limit: int) -> list[str]:
    root = Path("scenarios") / task_id
    paths = sorted(root.glob("*.json"))[:limit]
    return [path.stem for path in paths]


def generate_action(
    model,
    tokenizer,
    observation: dict[str, Any],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    prompt = build_prompt(observation)
    tokenized = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

    if hasattr(model, "device"):
        device = model.device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_len = input_ids.shape[1]
    completion_ids = outputs[0][prompt_len:]
    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return parse_action_output(completion_text)


def evaluate_policy(
    *,
    policy_name: str,
    model,
    tokenizer,
    session: requests.Session,
    env_url: str,
    tasks: list[str],
    scenarios_per_task: int,
    max_steps: int,
    success_threshold: float,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    seed: int,
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []

    for task_id in tasks:
        scenario_ids = list_scenarios(task_id, scenarios_per_task)
        for scenario_id in scenario_ids:
            obs = reset_env(session, env_url, task_id, scenario_id, seed)
            rewards: list[float] = []
            done = False
            step_count = 0

            for _ in range(max_steps):
                action = generate_action(
                    model,
                    tokenizer,
                    obs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                )
                obs, step_reward, done = step_env(session, env_url, action)
                rewards.append(step_reward)
                step_count += 1
                if done:
                    break

            score, breakdown = grade_current_episode(session, env_url)
            episodes.append(
                {
                    "task_id": task_id,
                    "scenario_id": scenario_id,
                    "score": round(score, 4),
                    "success": bool(score >= success_threshold),
                    "steps": step_count,
                    "mean_step_reward": round(sum(rewards) / max(1, len(rewards)), 4),
                    "breakdown": breakdown,
                }
            )

    mean_score = sum(item["score"] for item in episodes) / max(1, len(episodes))
    success_rate = sum(1 for item in episodes if item["success"]) / max(1, len(episodes))
    mean_steps = sum(item["steps"] for item in episodes) / max(1, len(episodes))

    return {
        "policy": policy_name,
        "episodes": episodes,
        "summary": {
            "episode_count": len(episodes),
            "mean_score": round(mean_score, 4),
            "success_rate": round(success_rate, 4),
            "mean_steps": round(mean_steps, 4),
        },
    }


def print_summary(results: dict[str, Any]) -> None:
    summary = results.get("summary", {})
    print(
        f"[{results.get('policy')}] episodes={summary.get('episode_count')} "
        f"mean_score={summary.get('mean_score')} success_rate={summary.get('success_rate')} "
        f"mean_steps={summary.get('mean_steps')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GRPO policies on deterministic OpenEnv scenarios")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--env_url", type=str, default="http://localhost:7860")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--compare_adapter_path", type=str, default=None)
    parser.add_argument("--tasks", type=str, nargs="+", default=list(TRAIN_TASKS))
    parser.add_argument("--scenarios_per_task", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=25)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--success_threshold", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--output_json", type=str, default="eval_results.json")
    args = parser.parse_args()

    tasks = [task for task in args.tasks if task in TRAIN_TASKS]
    if not tasks:
        raise ValueError("No valid tasks provided")

    session = build_http_session()
    policies: list[dict[str, Any]] = []

    model, tokenizer = load_policy(args.model_name, args.adapter_path, args.load_in_4bit)
    base_name = "candidate"
    if args.adapter_path:
        base_name = f"adapter:{Path(args.adapter_path).name}"
    policies.append(
        evaluate_policy(
            policy_name=base_name,
            model=model,
            tokenizer=tokenizer,
            session=session,
            env_url=args.env_url,
            tasks=tasks,
            scenarios_per_task=args.scenarios_per_task,
            max_steps=args.max_steps,
            success_threshold=args.success_threshold,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        )
    )

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.compare_adapter_path:
        compare_model, compare_tokenizer = load_policy(
            args.model_name,
            args.compare_adapter_path,
            args.load_in_4bit,
        )
        policies.append(
            evaluate_policy(
                policy_name=f"adapter:{Path(args.compare_adapter_path).name}",
                model=compare_model,
                tokenizer=compare_tokenizer,
                session=session,
                env_url=args.env_url,
                tasks=tasks,
                scenarios_per_task=args.scenarios_per_task,
                max_steps=args.max_steps,
                success_threshold=args.success_threshold,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed,
            )
        )
        del compare_model
        del compare_tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output = {"policies": policies}
    if len(policies) == 2:
        before = policies[0]["summary"]
        after = policies[1]["summary"]
        output["delta"] = {
            "mean_score": round(after["mean_score"] - before["mean_score"], 4),
            "success_rate": round(after["success_rate"] - before["success_rate"], 4),
            "mean_steps": round(after["mean_steps"] - before["mean_steps"], 4),
        }

    for item in policies:
        print_summary(item)
    if "delta" in output:
        print(f"[delta] {json.dumps(output['delta'], sort_keys=True)}")

    Path(args.output_json).write_text(json.dumps(output, indent=2))
    print(f"Wrote evaluation report to {args.output_json}")


if __name__ == "__main__":
    main()
