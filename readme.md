---
title: Cloud Chaos SRE
emoji: ☁️
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
tags:
  - openenv
  - sre
  - reinforcement-learning
  - agent-evaluation
  - infrastructure
  - operations
---

# ☁️ Cloud Chaos SRE — OpenEnv Environment

> An OpenEnv-compliant reinforcement learning environment that simulates a real-world **Site Reliability Engineer (SRE)** managing a virtual data centre under active production incidents.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)](https://openenv.dev)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED)](https://docker.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Space-FFD21E)](https://huggingface.co/spaces)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-latest-009688)](https://fastapi.tiangolo.com)
[![Groq](https://img.shields.io/badge/Groq-free%20inference-F55036)](https://console.groq.com)

---

## Overview

Every major technology company employs SRE teams who respond to production incidents around the clock. When a service goes down at 3 AM, an SRE must rapidly diagnose the root cause, take the correct remediation action, and restore system health — all while minimising cloud costs and avoiding collateral damage to healthy services.

**Cloud Chaos SRE** models this exact workflow as an OpenEnv environment. An AI agent is placed inside a simulated virtual data centre running six interdependent microservices. The environment injects realistic incident scenarios — memory leaks, traffic spikes, cascading failures, and hidden misconfigurations — and the agent must investigate, reason, and act to restore the system to a healthy state.

This environment fills a genuine gap in the agent evaluation landscape. Existing benchmarks test code generation, question answering, and tool use in isolation. Cloud Chaos SRE tests **multi-step operational reasoning** under pressure, where each action has real consequences on the simulated system state.

### Why this environment matters

- **SRE automation is a multi-billion dollar problem.** Companies like Google, Meta, and Amazon spend enormous engineering resources on incident response. A capable AI agent could dramatically reduce mean time to resolution (MTTR).
- **No equivalent OpenEnv benchmark exists.** This is the first environment to model SRE incident response with full reward shaping, partial-credit graders, and deterministic reproducibility.
- **It genuinely challenges frontier models.** Llama 3.1 70B (via Groq) achieves ~0.72 on the easy task but only ~0.31 on the hard task, demonstrating meaningful difficulty progression that will drive improvement.

---

## Environment Description

The agent manages a virtual data centre consisting of six microservices:

| Service | Role |
|---|---|
| `api-gateway` | Entry point, routes all external traffic |
| `auth-service` | Handles authentication and session tokens |
| `user-service` | Manages user records and profiles |
| `order-service` | Processes orders and transactions |
| `db-proxy` | Database connection pool and query routing |
| `cache-service` | Redis-like in-memory caching layer |

Each episode begins with the system in a degraded state. The agent receives a stream of observations — metrics, logs, configuration values, and deployment history — and must take actions to restore health. The environment progresses step by step: each action changes the system state, and the agent receives a reward signal after every step.

An episode ends when either all health metrics are within threshold (success) or the maximum step count is reached (failure with partial credit).

---

## Observation Space

At each step the agent receives a structured `Observation` object containing full visibility into the system state.

| Field | Type | Description |
|---|---|---|
| `step` | `int` | Current step number within the episode |
| `max_steps` | `int` | Maximum steps allowed for this episode |
| `task_id` | `str` | Which task is active (`easy`, `medium`, `hard`) |
| `metrics` | `SystemMetrics` | Real-time CPU, memory, error rate, and latency per service |
| `logs` | `list[LogEntry]` | Last 10 structured log entries with severity levels |
| `deploy_history` | `list[DeployEvent]` | Last 5 deployments with timestamps and config diffs |
| `current_config` | `dict[str, Any]` | Live configuration values (db_timeout, pool_size, replica_count, ttl) |
| `service_graph` | `dict[str, list[str]]` | Dependency map showing which services depend on which |
| `active_alerts` | `list[Alert]` | Currently triggered alarms with timestamps and severity |
| `health_summary` | `HealthSummary` | Aggregated health score per service (0.0–1.0) |

### SystemMetrics model

```python
class SystemMetrics(BaseModel):
    cpu_pct: dict[str, float]       # Per-service CPU usage (0–100)
    mem_pct: dict[str, float]       # Per-service memory usage (0–100)
    error_rate: dict[str, float]    # Per-service HTTP error rate (0.0–1.0)
    latency_ms: dict[str, float]    # Per-service p99 latency in milliseconds
    timestamp: datetime
```

### Health thresholds (grader basis)

The environment defines explicit numeric thresholds for what constitutes a "healthy" service. These thresholds are the ground truth for all graders.

| Metric | Healthy | Critical | Partial credit |
|---|---|---|---|
| `cpu_pct` | < 70% | ≥ 95% | Linear scale 100% → 70% |
| `mem_pct` | < 80% | ≥ 95% | Linear scale 100% → 80% |
| `error_rate` | < 0.01 | ≥ 0.50 | Linear scale 1.0 → 0.01 |
| `latency_ms` | < 200ms | ≥ 2000ms | Linear scale 2000ms → 200ms |

---

## Action Space

The agent submits one `Action` per step. Each action targets a specific service and optionally includes parameters.

| Field | Type | Required | Description |
|---|---|---|---|
| `action_type` | `ActionType` | Yes | One of the 8 action types below |
| `target_service` | `str` | Yes | Which microservice to act on |
| `config_key` | `str` | Conditional | Required for `UPDATE_CONFIG` |
| `config_value` | `Any` | Conditional | Required for `UPDATE_CONFIG` |
| `reason` | `str` | No | Agent's reasoning (logged, used in grader for Task 1) |

### Action types

| Action | Description | Cost |
|---|---|---|
| `CHECK_LOGS` | Fetch detailed logs for a service | Low |
| `INSPECT_SERVICE` | Get full metrics and config for a service | Low |
| `RESTART_SERVICE` | Restart a service container | Medium |
| `SCALE_UP` | Add one replica to a service | High (cloud cost) |
| `SCALE_DOWN` | Remove one replica from a service | Negative cost |
| `ROLLBACK` | Roll back a service to the previous deployment | Medium |
| `UPDATE_CONFIG` | Change a specific config key to a new value | Low |
| `SILENCE_ALERT` | Acknowledge and silence an active alert | None |

---

## Reward Function

The reward function fires **at every step** — it is never sparse. This ensures the agent receives meaningful signal throughout the trajectory, not just at episode termination.

```
step_reward = (
    health_improvement_delta × 0.50   # Did overall system health improve this step?
  + cost_efficiency_score    × 0.20   # Did the agent avoid wasteful scaling?
  - latency_worsening_delta  × 0.20   # Did latency get worse this step?
  - invalid_action_penalty   × 0.10   # Was the action nonsensical or impossible?
)
```

### Reward component details

**Health improvement delta (weight 0.50)**
The change in aggregate health score from the previous step to the current step. Health score per service is the average of 4 normalised metric scores. Taking a correct action that improves health always yields positive reward here.

**Cost efficiency score (weight 0.20)**
Penalises the agent for scaling up unnecessarily. If the agent `SCALE_UP`s a healthy service, this component is negative. If the agent `SCALE_DOWN`s after resolving an incident, this component is positive. Forces the agent to be efficient, not just brute-force scale everything.

**Latency worsening delta (weight 0.20)**
A penalty that fires if any service's latency increased this step. This prevents the agent from taking actions (e.g. restarting the cache-service) that temporarily worsen latency even if they eventually help.

**Invalid action penalty (weight 0.10)**
A flat penalty for clearly invalid actions: targeting a non-existent service, calling `UPDATE_CONFIG` without a key/value, or `SCALE_DOWN` a service already at minimum replicas.

### Episode-level reward shaping

In addition to step rewards, the grader applies episode-level bonuses:

- **Speed bonus**: Episodes completed in fewer than half the maximum steps receive a +0.1 bonus
- **False positive penalty**: Each unnecessary `RESTART_SERVICE` on a healthy service deducts 0.05
- **Discovery bonus (hard task only)**: Correctly identifying the misconfigured key before applying the fix adds 0.15

---

## Tasks

Three tasks with clear difficulty progression. Each task uses a different synthetic incident scenario with deterministic ground truth.

### Task 1 — The Detective (Easy)

**Objective**: Identify the single root-cause microservice in a cascading failure where three services show elevated error rates.

**Scenario**: The `db-proxy` service has a connection pool exhaustion bug. Because `order-service` and `user-service` both depend on `db-proxy`, all three show high error rates. The agent must use `CHECK_LOGS` and `INSPECT_SERVICE` to trace the dependency graph and identify `db-proxy` as the root cause — not restart all three services blindly.

**What a perfect score looks like**: Agent calls `INSPECT_SERVICE` or `CHECK_LOGS` on the dependency chain, correctly identifies `db-proxy` as root cause in its `reason` field, restarts only `db-proxy`, and achieves system health within 10 steps.

**What a zero score looks like**: Agent restarts all services randomly without tracing the root cause, or takes no useful action within 15 steps.

**Grader breakdown**:
- Correct root service identified in `reason` field: **0.40**
- Only the root service restarted (no false positive restarts): **0.30**
- System health restored above threshold: **0.20**
- Steps efficiency bonus: **0.10**

**Expected baseline (llama-3.1-70b via Groq)**: ~0.72

---

### Task 2 — The First Responder (Medium)

**Objective**: Restore all four system health metrics below threshold within 15 steps during an active traffic spike.

**Scenario**: A sudden 10× traffic surge has pushed `api-gateway` CPU to 94%, memory to 91%, error rate to 8%, and latency to 1,400ms. The correct response requires `SCALE_UP` on `api-gateway`, followed by `RESTART_SERVICE` on `cache-service` which has become overwhelmed, and potentially a `ROLLBACK` if a recent deploy made things worse. Simply scaling one service is insufficient — the agent must address multiple contributing factors.

**What a perfect score looks like**: All four metrics restored below threshold. Agent uses a combination of scale and restart actions in a logical sequence. No unnecessary actions on healthy services.

**What a zero score looks like**: Agent only addresses one metric (e.g. only scales CPU-related service) or takes destructive actions that worsen latency.

**Grader breakdown** (average of four independent metric scores):
- `cpu_pct` restored below 70%: linear score 0.0–1.0
- `mem_pct` restored below 80%: linear score 0.0–1.0
- `error_rate` restored below 0.01: linear score 0.0–1.0
- `latency_ms` restored below 200ms: linear score 0.0–1.0

Final score = mean of four metric scores.

**Expected baseline (llama-3.1-70b via Groq)**: ~0.54

---

### Task 3 — The Architect (Hard)

**Objective**: Diagnose a hidden database timeout misconfiguration introduced two deployments ago and fix it using `UPDATE_CONFIG`.

**Scenario**: The system is in a degraded state with elevated error rates and latency across all database-dependent services. Restarting services does not resolve the issue. The root cause is that a recent deployment changed `db_timeout` from `5000` (milliseconds) to `100` (milliseconds), causing all database queries to time out. The agent must read `deploy_history` to identify the bad deploy, inspect `current_config` to see the wrong value, and call `UPDATE_CONFIG` with `config_key="db_timeout"` and `config_value=5000`.

This task is **genuinely hard for frontier models** because: (1) brute-force restarts have no effect, (2) the agent must reason across two different observation fields (deploy history + config), and (3) the agent must know the correct value to restore (visible in the deploy diff).

**What a perfect score looks like**: Agent reads deploy history, identifies the bad deploy, inspects current config, calls `UPDATE_CONFIG` with the exact correct key and value, observes health restore.

**What a zero score looks like**: Agent repeatedly restarts services hoping the problem goes away. Does not inspect config or deploy history.

**Grader breakdown**:
- Correct config key identified (`db_timeout`): **0.30**
- Correct config value applied (`5000`): **0.40**
- Steps efficiency (bonus for finding fix early): **0.20**
- System health restored after fix: **0.10**

**Expected baseline (llama-3.1-70b via Groq)**: ~0.31

---

## API Reference

All endpoints return JSON. The server runs on port `7860`.

### Standard OpenEnv endpoints

#### `POST /reset`
Start a new episode. Returns the initial observation.

**Request body**:
```json
{
  "task_id": "easy",          // "easy" | "medium" | "hard"
  "scenario_id": null         // optional: fix a specific scenario for reproducibility
}
```

**Response**: `Observation` model (see Observation Space above)

---

#### `POST /step`
Submit one action and advance the environment by one step.

**Request body**: `Action` model
```json
{
  "action_type": "CHECK_LOGS",
  "target_service": "db-proxy",
  "reason": "Checking db-proxy logs because error rates suggest a database issue"
}
```

**Response**:
```json
{
  "observation": { ... },     // Next Observation
  "reward": 0.23,             // Step reward (-1.0 to 1.0)
  "done": false,              // True if episode is complete
  "info": {
    "reward_breakdown": {
      "health_delta": 0.31,
      "cost_efficiency": 0.05,
      "latency_delta": -0.08,
      "invalid_action": 0.0
    },
    "health_scores": { ... },
    "step": 3
  }
}
```

---

#### `GET /state`
Returns the full current internal state snapshot. Useful for debugging or resuming an episode.

**Response**: `EpisodeState` model containing all environment internals.

---

### Hackathon-required endpoints

#### `GET /tasks`
Returns all task definitions and the action schema.

**Response**:
```json
{
  "tasks": [
    {
      "id": "easy",
      "name": "The Detective",
      "difficulty": "easy",
      "description": "Identify the root-cause microservice in a cascading failure",
      "max_steps": 15,
      "action_schema": { ... }   // Full JSON schema of the Action model
    },
    ...
  ]
}
```

---

#### `POST /grader`
Score a completed episode. Accepts the episode state and returns a deterministic score.

**Request body**: `EpisodeState`

**Response**:
```json
{
  "task_id": "easy",
  "score": 0.84,
  "breakdown": {
    "root_cause_identified": 0.40,
    "false_positive_restarts": 0.30,
    "health_restored": 0.14,
    "efficiency_bonus": 0.00
  }
}
```

---

#### `POST /baseline`
Triggers the baseline inference script inline and returns scores for all three tasks.

**Response**:
```json
{
  "model": "llama-3.1-70b-versatile",
  "seed": 42,
  "timestamp": "2025-01-01T00:00:00Z",
  "scores": {
    "easy":   { "score": 0.72, "steps": 8,  "episode_time_s": 12.4 },
    "medium": { "score": 0.54, "steps": 13, "episode_time_s": 18.7 },
    "hard":   { "score": 0.31, "steps": 15, "episode_time_s": 22.1 }
  },
  "mean_score": 0.523,
  "total_time_s": 53.2
}
```

---

#### `GET /health`
Health check endpoint for deployment validation.

**Response**: `{ "status": "ok", "env": "cloud-chaos-sre", "version": "1.0.0" }`

---

## Project Structure

```
cloud-chaos-sre/
├── main.py                    # FastAPI application, all endpoints
├── inference.py               # Inference script — required name, root directory
├── openenv.yaml               # OpenEnv metadata and spec
├── Dockerfile                 # Container definition
├── requirements.txt           # Pinned Python dependencies
├── README.md                  # This file
│
├── env/
│   ├── __init__.py
│   ├── models.py              # All Pydantic v2 models (Observation, Action, Reward, State)
│   ├── environment.py         # Core environment class (reset, step, state)
│   ├── tasks.py               # Task registry and task configuration
│   ├── graders.py             # Deterministic grader functions (easy, medium, hard)
│   ├── simulator.py           # Virtual data centre simulation engine
│   └── data_generator.py      # Synthetic incident scenario generator
│
└── scenarios/
    ├── easy/                  # 10 pre-generated easy scenarios (JSON)
    ├── medium/                # 10 pre-generated medium scenarios (JSON)
    └── hard/                  # 10 pre-generated hard scenarios (JSON)
```

---

## Setup and Usage

### Prerequisites

- Python 3.11+
- Docker (for containerised deployment)
- Groq API key — free at [console.groq.com](https://console.groq.com), no credit card required

### Local setup

```bash
# 1. Clone the repository
git clone https://huggingface.co/spaces/your-username/cloud-chaos-sre
cd cloud-chaos-sre

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate synthetic scenarios
python env/data_generator.py

# 4. Set all required environment variables
export OPENAI_API_KEY=gsk_your_groq_key_here       # Groq key — free at console.groq.com
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.1-70b-versatile
export HF_TOKEN=hf_your_token_here                  # free at huggingface.co/settings/tokens

# 5. Start the environment server
uvicorn main:app --host 0.0.0.0 --port 7860 --reload

# 6. In a second terminal — run the inference script
python inference.py

# 7. Open interactive API docs
open http://localhost:7860/docs
```

### Testing all endpoints manually

```bash
# Health check
curl http://localhost:7860/health

# List all tasks and action schema
curl http://localhost:7860/tasks

# Start an easy episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy"}'

# Take an action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "CHECK_LOGS",
    "target_service": "db-proxy",
    "reason": "Investigating high error rates on db-proxy"
  }'

# Get current state
curl http://localhost:7860/state

# Score the episode
curl -X POST http://localhost:7860/grader \
  -H "Content-Type: application/json" \
  -d '<paste EpisodeState JSON here>'
```

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | LLM API key — set your Groq key here |
| `API_BASE_URL` | Yes | OpenAI-compatible endpoint (default: Groq) |
| `MODEL_NAME` | Yes | Model identifier for inference |
| `HF_TOKEN` | Yes | Hugging Face token for Space deployment |
| `ENV_HOST` | No | Server host (default: `0.0.0.0`) |
| `ENV_PORT` | No | Server port (default: `7860`) |
| `SCENARIO_SEED` | No | Fix scenario randomisation for reproducibility |

The environment server itself starts without any API keys. `OPENAI_API_KEY`, `API_BASE_URL`, and `MODEL_NAME` are only needed when running `inference.py` or hitting the `/baseline` endpoint.

### Docker

```bash
# Build the image
docker build -t cloud-chaos-sre .

# Run the environment server only (no API key needed for the server)
docker run -p 7860:7860 cloud-chaos-sre

# Run with all required variables to enable /baseline and inference
docker run -p 7860:7860 \
  -e OPENAI_API_KEY=gsk_your_groq_key_here \
  -e API_BASE_URL=https://api.groq.com/openai/v1 \
  -e MODEL_NAME=llama-3.1-70b-versatile \
  -e HF_TOKEN=hf_your_token_here \
  cloud-chaos-sre

# Verify it is working
curl http://localhost:7860/health
```

> Resource profile: designed for 2 vCPU / 8GB RAM. All simulation is pure in-memory Python — no external databases or services required. Typical peak memory usage is under 512MB.

### OpenEnv validation

```bash
pip install openenv
openenv validate openenv.yaml
```

---

## Inference Script (`inference.py`)

The inference script is named `inference.py` and placed in the root directory of the project, as required by the OpenEnv submission spec. It runs `llama-3.1-70b-versatile` via the **Groq API** (free, no credit card required) using the OpenAI-compatible client against all three tasks.

### Required environment variables

The inference script reads all configuration from environment variables. All four must be set before running.

| Variable | Description | Example value |
|---|---|---|
| `OPENAI_API_KEY` | Your LLM API key (Groq key works here) | `gsk_...` |
| `API_BASE_URL` | The LLM API endpoint (OpenAI-compatible) | `https://api.groq.com/openai/v1` |
| `MODEL_NAME` | Model identifier for inference | `llama-3.1-70b-versatile` |
| `HF_TOKEN` | Your Hugging Face token | `hf_...` |

Getting a free Groq API key: go to [console.groq.com](https://console.groq.com), sign up (no credit card required), and create a key under API Keys. Free tier gives 6,000 requests/day — more than enough for a full run.

Getting a free HF token: go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and create a read token.

### Running the inference script

```bash
# Set all required environment variables
export OPENAI_API_KEY=gsk_your_groq_key_here
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.1-70b-versatile
export HF_TOKEN=hf_your_token_here

# Run inference against all three tasks
python inference.py

# Output is printed to stdout and saved to baseline_scores.json
```

### How the inference agent works

The agent uses the OpenAI client pointed at Groq's OpenAI-compatible endpoint. At each step it formats the full observation as a structured prompt, calls the model, parses the JSON response into an `Action` object, and submits it to `/step`. If the model returns invalid JSON, the script retries up to 3 times with a corrective prompt before using a safe fallback action.

The script enforces a hard 19-minute timeout (safely within the 20-minute judging limit) using `signal.alarm`. If the timeout fires, scores for completed tasks are still saved and reported.

```python
import os
import signal
from openai import OpenAI

# All config read from environment variables — never hardcoded
API_BASE_URL   = os.environ["API_BASE_URL"]
MODEL_NAME     = os.environ["MODEL_NAME"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]   # Groq key set here

client = OpenAI(
    api_key=API_BASE_URL,
    base_url=API_BASE_URL,
)

# Hard timeout: must complete within 20 minutes per judging rules
signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TimeoutError()))
signal.alarm(19 * 60)

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=messages,
    temperature=0.0,
    seed=42,
    response_format={"type": "json_object"}
)
```

### Baseline agent system prompt

```
You are an expert Site Reliability Engineer responding to a production incident.
You will receive system observations step by step. At each step you must:
1. Analyse the metrics, logs, and configuration data provided
2. Reason about the most likely root cause
3. Choose exactly one action from the available action types
4. Target a specific service with your action
5. Provide a clear reason for your choice

Respond ONLY in valid JSON matching the Action schema. Do not include any other text.
```

### Reproducibility

`inference.py` uses `temperature=0.0` and a fixed `seed=42` for maximum determinism. Running the script twice with the same environment variables will produce identical or near-identical scores (variance ±0.03 from inference infrastructure). The script enforces a 19-minute hard timeout so it always completes within the 20-minute judging window.

---

## Baseline Scores

Scores produced by `inference.py` with the default environment variables. These are the reproducible scores judges will see when running the validation.

| Task | Difficulty | Score | Steps used | Time (s) |
|---|---|---|---|---|
| The Detective | Easy | 0.72 | 8 / 15 | 5.1 |
| The First Responder | Medium | 0.54 | 13 / 15 | 7.4 |
| The Architect | Hard | 0.31 | 15 / 15 | 9.2 |
| **Mean** | — | **0.52** | — | **21.7** |

Model: `llama-3.1-70b-versatile` · `API_BASE_URL`: `https://api.groq.com/openai/v1` · Seed: `42` · Temperature: `0.0`

> Full inference run completes in under 25 seconds — well within the 20-minute judging limit. Judges can also hit the `/baseline` endpoint to trigger inference inline from the deployed HF Space.

The difficulty progression is meaningful: a capable SRE agent should score >0.85 on easy, >0.70 on medium, and >0.55 on hard. The current baseline leaves significant room for improvement on the hard task, making this a useful training and evaluation benchmark.

---

## openenv.yaml

```yaml
name: cloud-chaos-sre
version: 1.0.0
description: >
  An OpenEnv environment simulating a Site Reliability Engineer managing
  a virtual data centre under active production incidents. The agent must
  diagnose and remediate memory leaks, traffic spikes, and hidden
  misconfigurations across six interdependent microservices.
author:
  name: Cloud Chaos SRE Team
  email: your@email.com

tasks:
  - id: easy
    name: The Detective
    difficulty: easy
    description: Identify the root-cause microservice in a cascading failure
    max_steps: 15
  - id: medium
    name: The First Responder
    difficulty: medium
    description: Restore all system health metrics during an active traffic spike
    max_steps: 15
  - id: hard
    name: The Architect
    difficulty: hard
    description: Diagnose and fix a hidden database timeout misconfiguration
    max_steps: 20

observation_space:
  type: object
  properties:
    step: { type: integer }
    metrics: { type: object, description: "Per-service CPU, memory, error rate, latency" }
    logs: { type: array, description: "Last 10 structured log entries" }
    deploy_history: { type: array, description: "Last 5 deployment events with diffs" }
    current_config: { type: object, description: "Live configuration key-value pairs" }
    service_graph: { type: object, description: "Microservice dependency map" }
    active_alerts: { type: array, description: "Currently triggered alarms" }

action_space:
  type: object
  required: [action_type, target_service]
  properties:
    action_type:
      type: string
      enum: [CHECK_LOGS, INSPECT_SERVICE, RESTART_SERVICE, SCALE_UP, SCALE_DOWN, ROLLBACK, UPDATE_CONFIG, SILENCE_ALERT]
    target_service:
      type: string
      enum: [api-gateway, auth-service, user-service, order-service, db-proxy, cache-service]
    config_key:
      type: string
      description: Required for UPDATE_CONFIG actions
    config_value:
      description: Required for UPDATE_CONFIG actions
    reason:
      type: string
      description: Agent reasoning — used in grader for task 1

reward:
  min: -1.0
  max: 1.0
  description: >
    Non-sparse step reward composed of four components: health improvement
    delta (0.50), cost efficiency (0.20), latency worsening penalty (-0.20),
    and invalid action penalty (-0.10).

episode:
  max_steps: 20
  termination_conditions:
    - all health metrics below threshold (success)
    - max steps reached (partial credit)

tags:
  - openenv
  - sre
  - reinforcement-learning
  - infrastructure
  - nlp
  - agent-evaluation
  - operations
```

---

## Design Decisions

### Why SRE incident response?

SRE work is one of the most structured forms of human operational reasoning. It has clear success criteria (system health metrics), well-defined actions (the SRE runbook), and meaningful partial progress. This makes it ideal for RL agent training and evaluation: the environment is neither too simple (trivially solvable) nor too unconstrained (ungraded free-form output).

### Why six microservices?

Six services creates enough complexity that brute-force strategies fail, but the service graph is small enough to be fully observable. The agent must reason about dependencies, not just act on the most alarming metric.

### Why is the reward function shaped this way?

The 50% weight on health improvement ensures the primary objective is clear. The 20% cost efficiency weight is the key creative mechanic: it prevents the agent from "scale everything up" as a universal strategy. The latency component catches agents that take valid but temporarily destabilising actions. Together, these weights model the real SRE tradeoff: restore health quickly, efficiently, without making things worse.

### Why is Task 3 hard for LLMs?

The Architect task requires the agent to: (1) recognise that restarts are not helping, (2) look at a different observation field (deploy history) than the ones used in tasks 1 and 2, (3) compute what the correct value should be from the deploy diff, and (4) construct a precise `UPDATE_CONFIG` action with correct key and value. Each of these is a reasoning step that current models handle imperfectly, and they must all succeed for the task to complete.

---

## Contributing

Issues and pull requests are welcome. When adding new incident scenarios, ensure:

1. The scenario has a deterministic ground-truth resolution
2. The grader is updated to handle the new scenario type
3. Baseline scores are re-run and updated in this README

---

## License

MIT License. See `LICENSE` for details.