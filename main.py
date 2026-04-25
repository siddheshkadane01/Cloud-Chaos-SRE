import json
import sys
import threading
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

from env.data_generator import generate_all_scenarios
from env.environment import SREEnvironment
from env.models import Action, EpisodeState, Observation, Reward
from env.tasks import TASKS as TASK_LIST

env = SREEnvironment(deterministic=True, evaluation_mode=True)


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: Literal["easy", "medium", "hard", "expert", "enterprise"] = Field(default="easy")
    scenario_id: str | None = Field(default=None)
    seed: int | None = Field(default=None)
    deterministic: bool | None = Field(default=None)
    evaluation_mode: bool | None = Field(default=None)
    mode: Literal["single_agent", "multi_agent"] = Field(default="single_agent")

# In-memory leaderboard — tracks best score per task across all sessions.
_leaderboard_lock = threading.Lock()
_leaderboard: dict[str, list[dict]] = defaultdict(list)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ = app
    scenarios_root = Path("scenarios")
    required_tasks = ["easy", "medium", "hard", "expert", "enterprise"]
    should_generate = False

    for task in required_tasks:
        task_dir = scenarios_root / task
        if not task_dir.exists() or not any(task_dir.glob("*.json")):
            should_generate = True
            break

    if should_generate:
        generate_all_scenarios()
    yield


app = FastAPI(
    title="Site Reliability Server",
    description="OpenEnv environment simulating SRE incident response across 6 interdependent microservices.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static landing page
_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ------------------------------------------------------------------
# Landing page
# ------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root():
    index = _STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"env": "site-reliability-server", "version": "1.1.0", "docs": "/docs"}


# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "env": "site-reliability-server", "version": "1.1.0"}


# ------------------------------------------------------------------
# Standard OpenEnv endpoints
# ------------------------------------------------------------------

@app.post("/reset")
def reset(body: ResetRequest | None = None):
    body = body or ResetRequest()
    obs = env.reset(
        task_id=body.task_id,
        scenario_id=body.scenario_id,
        seed=body.seed,
        deterministic=body.deterministic,
        evaluation_mode=body.evaluation_mode,
        mode=body.mode,
    )
    return obs.model_dump()


@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    return env.state().model_dump()


# ------------------------------------------------------------------
# Tasks
# ------------------------------------------------------------------

@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {
                **task,
                "action_schema": Action.model_json_schema(),
                "observation_schema": Observation.model_json_schema(),
                "reward_schema": Reward.model_json_schema(),
                "deterministic_default": True,
                "evaluation_mode_default": True,
            }
            for task in TASK_LIST
        ]
    }


# ------------------------------------------------------------------
# Grader
# ------------------------------------------------------------------

@app.post("/grader")
def grader(episode_state: EpisodeState):
    env._state = episode_state
    score, breakdown = env.grade()

    # Record on leaderboard
    with _leaderboard_lock:
        _leaderboard[episode_state.task_id].append(
            {
                "score": score,
                "steps": episode_state.step,
                "scenario_id": episode_state.scenario_id,
            }
        )

    return {"task_id": episode_state.task_id, "score": score, "breakdown": breakdown}


# ------------------------------------------------------------------
# Leaderboard
# ------------------------------------------------------------------

@app.get("/metrics")
def metrics():
    """
    Returns the best-ever score and mean score per task across all grader calls
    in this server session. Useful for real-time evaluation dashboards.
    """
    result: dict[str, dict] = {}
    with _leaderboard_lock:
        for task_id, entries in _leaderboard.items():
            if not entries:
                continue
            scores = [e["score"] for e in entries]
            result[task_id] = {
                "runs": len(entries),
                "best_score": round(max(scores), 4),
                "mean_score": round(sum(scores) / len(scores), 4),
                "last_score": round(scores[-1], 4),
            }
    return {"leaderboard": result, "total_runs": sum(len(v) for v in _leaderboard.values())}


# ------------------------------------------------------------------
# Baseline trigger
# ------------------------------------------------------------------

@app.post("/baseline")
def baseline():
    import subprocess
    import time

    inference_path = Path(__file__).with_name("inference.py")
    scores_path = Path(__file__).with_name("baseline_scores.json")
    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(inference_path)],
            capture_output=True,
            text=True,
            timeout=19 * 60,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "error": "inference_timeout",
            "message": "inference.py exceeded 19-minute timeout",
            "stdout_tail": (exc.stdout or "")[-1200:],
            "stderr_tail": (exc.stderr or "")[-1200:],
            "total_time_s": round(time.time() - start, 1),
        }

    elapsed = round(time.time() - start, 1)

    if result.returncode != 0:
        return {
            "ok": False,
            "error": "inference_failed",
            "returncode": result.returncode,
            "stdout_tail": (result.stdout or "")[-1200:],
            "stderr_tail": (result.stderr or "")[-1200:],
            "total_time_s": elapsed,
        }

    try:
        loaded = json.loads(scores_path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "ok": False,
            "error": "invalid_inference_artifact",
            "message": "Could not parse baseline_scores.json produced by inference.py",
            "stdout_tail": (result.stdout or "")[-1200:],
            "stderr_tail": (result.stderr or "")[-1200:],
            "total_time_s": elapsed,
        }

    if not isinstance(loaded, dict):
        loaded = {"result": loaded}

    scores: dict[str, object] = dict(loaded)
    scores["ok"] = True
    scores["total_time_s"] = elapsed
    return scores
