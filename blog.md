# Beyond Code: Teaching LLMs to Follow Enterprise Protocol using GRPO and OpenEnv

**Team Powerhouse**  
5 min read

If a production server goes down at 3 AM and you ask a standard Instruct LLM to fix it, it will almost certainly do the exact wrong thing: it will try to blindly execute `RESTART_SERVICE`.

Modern LLMs are trained to be "helpful assistants." When presented with an error, their instinct is to immediately provide a technical fix. But in an enterprise Site Reliability Engineering (SRE) environment, taking technical action before acknowledging the PagerDuty ticket, diagnosing the root cause, and notifying stakeholders in Slack is a massive protocol violation. LLMs are great at writing code, but they are notoriously bad at navigating rigid business workflows.

For the April '26 OpenEnv Hackathon, my team set out to fix this. I did not just fine-tune a model to write better bash scripts. We built Site Reliability Server, a fully interactive incident response simulator, and used GRPO (Group Relative Policy Optimization) to train an agent to survive the chaos and respect the pager.

## The Problem Statement: Why Enterprise SRE Breaks LLMs

To train an agent to be an SRE, you cannot just feed it static text traces. It needs to act in a dynamic environment, make mistakes, and feel the consequences.

The core challenge is that real-world operations require a blend of diagnostic reasoning and state-machine compliance. A human SRE knows the unwritten rule:

**Acknowledge -> Notify -> Triage -> Remediate -> Resolve**

If an AI agent skips directly to "Remediate" while the incident is unacknowledged, escalations fire, managers are woken up unnecessarily, and the incident command structure breaks down.

Using the OpenEnv framework, we built a FastAPI-based simulator that mocks out a real enterprise infrastructure. The agent receives a complex state dictionary containing:

- `apps_state`: The health metrics of interconnected microservices (API Gateway, DB Proxy, Cache Service, etc.)
- `protocol_status`: The current, highly-rigid state of PagerDuty and Slack channels

The agent can take 12 different actions, ranging from `INSPECT_SERVICE` to `DRAIN_TRAFFIC`.

![Reward Curve](./reward_curve.png)

![Loss Curve](./loss_curve.png)

## Preventing Reward Hacking with Multiple Verifiers

In Reinforcement Learning, if you only give an AI one single monolithic reward signal, it will inevitably find a way to "hack" it (like resolving a ticket without actually fixing the server just to get the reward).

To prevent this, we leaned heavily into programmatic verifiers using TRL and Unsloth for lightning-fast training on an L40S GPU. Instead of one opaque reward function, we split our grading into three independent, strict verifiers:

- **Format Validity (+0.5 / -0.5):** Did the agent output valid, parseable JSON?
- **Action Validity (+0.5 / -0.5):** Did it select a real action and a valid target service?
- **Protocol Adherence (+1.0 / -1.0):** The crucial step. If the agent tries to restart a server while the PagerDuty ticket is unacknowledged, it receives a massive `-1.0` penalty.

Here is the exact GRPO verifier we wrote to enforce the business logic:

```python
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


def make_protocol_adherence_reward_function():
    def protocol_adherence_reward(prompts: list[str], completions: list[Any], **_: Any) -> list[float]:
        _ = prompts
        actions: list[dict[str, Any]] = []
        for completion in completions:
            completion_text = _completion_to_text(completion)
            actions.append(parse_action_output(completion_text))
        return _protocol_adherence_scores(actions)
    return protocol_adherence_reward
```

By tracking these independently on Weights & Biases (WandB), we could watch the model transition from immediate, reckless server restarts to carefully sequencing the Acknowledge -> Notify -> Resolve pipeline.

## The Twist: Natively Multi-Agent

As we finalized our single-agent training loop, we realized something critical: real incident response is never single-player. We decided to architect a fully backward-compatible layer to tackle this simultaneously. By passing `mode="multi_agent"` to our OpenEnv simulator, the environment instantly transforms into a 4-role Incident Command System (ICS):

- **Incident Commander:** Manages PagerDuty and delegates tasks
- **Investigator:** Reads logs and inspects services
- **Remediator:** Has the keys to restart, scale, and rollback
- **Comms Officer:** Manages Slack updates to stakeholders

Strict Role-Based Access Control (RBAC): agents must now pass their `actor_role` and optionally a `handoff_to` role in their action payload. If the Investigator tries to scale up a database, the environment blocks the action and issues a penalty. If an agent properly hands off control to the next role in the sequence, the team gets a `coordination_bonus`. We effectively built a system that grades not just actions, but collaboration quality.

## Results and Takeaways

Building RL environments is incredibly different from standard supervised fine-tuning.

- You have to start simple: if the task is too hard, the agent never gets a positive reward, and learning stalls entirely. Curriculum learning is a requirement, not a luxury.
- The "Groundhog Day" trap: we had to carefully ensure our trainer scored complete trajectories rather than resetting the server after every single token generation.
- Infrastructure matters: running a FastAPI OpenEnv simulator natively on a Hugging Face Space while pointing a powerful A10G Jupyter training cluster at it was the ultimate separation of concerns.

You can check out our live interactive environment, training codebase, and demo video below:

- **Hugging Face Space:** https://huggingface.co/spaces/siddheshkadane/Cloud-Chaos-SRE
- **GitHub Repo:** https://github.com/siddheshkadane01/Site-Reliability-Server
- **Demo Video:** Link
- **W&B Training Dashboard:** https://wandb.ai/strungpattern-pune-institute-of-computer-technology/openenv-enterprise-grpo/runs/kpwdjvm1

Built for the OpenEnv April 2026 Hackathon.
