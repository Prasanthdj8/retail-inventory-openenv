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

import math
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
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


def _safe_score(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return 0.5
    return float(max(1e-6, min(1 - 1e-6, x)))

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
    return {"status": "healthy", "service": "retail-inventory-openenv"}


@app.get("/metadata")
def metadata():
    return {
        "name"       : "retail-inventory-expiry",
        "description": (
            "Simulates a store manager making real-time inventory decisions "
            "for perishable products. The agent decides when to discount, "
            "reorder, or remove items nearing expiry."
        ),
        "version"    : "1.0.0",
    }


@app.get("/schema")
def schema():
    return {
        "action": {
            "type": "object",
            "fields": {
                "action_type" : {"type": "string", "enum": ["discount","reorder","remove","do_nothing"]},
                "product_id"  : {"type": "string"},
                "discount_pct": {"type": "float", "minimum": 0.0, "maximum": 80.0},
                "reorder_qty" : {"type": "integer", "minimum": 0},
            }
        },
        "observation": {
            "type": "object",
            "fields": {
                "day"                  : {"type": "integer"},
                "total_days"           : {"type": "integer"},
                "products"             : {"type": "array"},
                "daily_revenue"        : {"type": "float"},
                "daily_waste_cost"     : {"type": "float"},
                "cumulative_revenue"   : {"type": "float"},
                "cumulative_waste_cost": {"type": "float"},
                "stockout_events"      : {"type": "integer"},
                "budget_remaining"     : {"type": "float"},
            }
        },
        "state": {
            "type": "object",
            "fields": {
                "task"           : {"type": "string"},
                "day"            : {"type": "integer"},
                "done"           : {"type": "boolean"},
                "budget"         : {"type": "float"},
                "episode_score"  : {"type": "float"},
            }
        },
    }


@app.post("/mcp")
async def mcp(request: Request):
    body = await request.json()
    method = body.get("method", "")
    req_id = body.get("id", 1)

    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities"   : {"tools": {}},
                "serverInfo"     : {"name": "retail-inventory-openenv", "version": "1.0.0"},
            }
        }
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {"tools": [
                {"name": "reset", "description": "Reset the environment", "inputSchema": {"type": "object"}},
                {"name": "step",  "description": "Take a step",          "inputSchema": {"type": "object"}},
                {"name": "state", "description": "Get current state",    "inputSchema": {"type": "object"}},
            ]}
        }
    else:
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    task = req.task
    task_map = {"easy_single_product":"easy","medium_multi_product":"medium","hard_full_store":"hard"}
    task = task_map.get(task, task)
    if task not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail=f"Unknown task: {task}. Choose easy/medium/hard.")
    seed = req.seed if req.seed is not None else DEFAULT_SEED
    env  = RetailInventoryEnv(task=task, seed=seed)
    _envs[task] = env
    obs = env.reset()
    result = obs.model_dump()
    result["episode_score"] = 0.5
    return result


@app.post("/step")
def step(req: StepRequest):
    task = req.task
    task_map = {"easy_single_product":"easy","medium_multi_product":"medium","hard_full_store":"hard"}
    task = task_map.get(task, task)
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
    r = reward.model_dump()
    r["total"] = _safe_score(r["total"])
    info["episode_score"] = _safe_score(info.get("episode_score", 0.5))
    return {
        "observation": obs.model_dump(),
        "reward"     : r,
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


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    task = "easy"
    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type")
            payload  = data.get("data", {})

            if msg_type == "reset":
                t = payload.get("task", payload.get("task_id", "easy"))
                task_map = {"easy_single_product":"easy","medium_multi_product":"medium","hard_full_store":"hard"}
                task = task_map.get(t, t)
                if task not in ("easy","medium","hard"):
                    task = "easy"
                seed = payload.get("seed", DEFAULT_SEED)
                env  = RetailInventoryEnv(task=task, seed=seed)
                _envs[task] = env
                obs = env.reset()
                await ws.send_json({"type": "reset", "data": {
                    "observation": obs.model_dump(),
                    "reward"     : {"total": 0.5},
                    "done"       : False,
                    "info"       : {"episode_score": 0.5, "task_id": task},
                }})

            elif msg_type == "step":
                env = _envs.get(task)
                if env is None:
                    await ws.send_json({"type": "error", "data": {"message": "No active episode", "code": "NO_EPISODE"}})
                    continue
                action_type = payload.get("action_type", payload.get("type", "do_nothing"))
                task_map = {"easy_single_product":"easy","medium_multi_product":"medium","hard_full_store":"hard"}
                t2 = payload.get("task", task)
                task = task_map.get(t2, t2) if t2 in task_map else task
                try:
                    action_type_enum = ActionType(action_type)
                except ValueError:
                    action_type_enum = ActionType.DO_NOTHING
                action = Action(
                    action_type  = action_type_enum,
                    product_id   = payload.get("product_id"),
                    discount_pct = float(payload.get("discount_pct", 0.0)),
                    reorder_qty  = int(payload.get("reorder_qty", 0)),
                )
                obs, reward, done, info = env.step(action)
                r = reward.model_dump()
                r["total"] = _safe_score(r["total"])
                info["episode_score"] = _safe_score(info.get("episode_score", 0.5))
                await ws.send_json({"type": "step", "data": {
                    "observation": obs.model_dump(),
                    "reward"     : r,
                    "done"       : done,
                    "info"       : info,
                }})

            elif msg_type == "state":
                env = _envs.get(task)
                if env:
                    await ws.send_json({"type": "state", "data": env.state()})
                else:
                    await ws.send_json({"type": "state", "data": {}})

            elif msg_type == "close":
                break

    except WebSocketDisconnect:
        pass
