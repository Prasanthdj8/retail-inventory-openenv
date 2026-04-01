"""
FastAPI server for the Retail Inventory & Expiry Management OpenEnv.

Endpoints
---------
POST /reset   → Observation
POST /step    → {observation, reward, done, info}
GET  /state   → full internal state dict
GET  /tasks   → list of tasks
GET  /health  → liveness check
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Flat imports — all files are in the same directory
from env import RetailInventoryEnv
from models import Action, ActionType

app = FastAPI(
    title       = "Retail Inventory & Expiry Management — OpenEnv",
    description = "OpenEnv environment for retail inventory management with perishable goods.",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

_envs: Dict[str, RetailInventoryEnv] = {}
DEFAULT_SEED = int(os.getenv("ENV_SEED", "42"))


class ResetRequest(BaseModel):
    task: str = "easy"
    seed: Optional[int] = None


class StepRequest(BaseModel):
    task        : str           = "easy"
    action_type : str           = "do_nothing"
    product_id  : Optional[str] = None
    discount_pct: float         = 0.0
    reorder_qty : int           = 0


@app.get("/health")
def health():
    return {"status": "ok", "service": "retail-inventory-openenv"}


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    task = req.task
    if task not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail=f"Unknown task: {task}. Choose easy/medium/hard.")
    seed = req.seed if req.seed is not None else DEFAULT_SEED
    env  = RetailInventoryEnv(task=task, seed=seed)
    _envs[task] = env
    obs = env.reset()
    return obs.model_dump()


@app.post("/step")
def step(req: StepRequest):
    task = req.task
    env  = _envs.get(task)
    if env is None:
        raise HTTPException(status_code=400,
            detail=f"No active episode for task '{task}'. Call /reset first.")
    try:
        action_type = ActionType(req.action_type)
    except ValueError:
        raise HTTPException(status_code=400,
            detail=f"Unknown action_type: {req.action_type}. "
                   f"Choose: {[a.value for a in ActionType]}")
    action = Action(
        action_type  = action_type,
        product_id   = req.product_id,
        discount_pct = req.discount_pct,
        reorder_qty  = req.reorder_qty,
    )
    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "observation": obs.model_dump(),
        "reward"     : reward.model_dump(),
        "done"       : done,
        "info"       : info,
    }


@app.get("/state")
def state(task: str = "easy"):
    env = _envs.get(task)
    if env is None:
        raise HTTPException(status_code=400,
            detail=f"No active episode for task '{task}'. Call /reset first.")
    return env.state()


@app.get("/tasks")
def tasks():
    from env import TASK_CONFIGS
    from graders import EasyGrader, MediumGrader, HardGrader
    theo = 1000.0
    graders = {
        "easy"  : EasyGrader(theo),
        "medium": MediumGrader(theo),
        "hard"  : HardGrader(theo),
    }
    return {
        task_id: {**cfg, "grader": graders[task_id].info()}
        for task_id, cfg in TASK_CONFIGS.items()
    }
