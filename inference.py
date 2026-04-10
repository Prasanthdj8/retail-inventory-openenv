"""
inference.py - Baseline inference script for Retail Inventory & Expiry Management OpenEnv

STDOUT FORMAT (matches sample spec exactly):
    [START] task=<task> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

  - reward in [STEP]  : :.2f  (0.00 and 1.00 are valid per sample spec)
  - score  in [END]   : :.3f
  - rewards in [END]  : :.2f  comma-separated
  - [END] always emitted in finally block even on exception
"""

from __future__ import annotations

import json
import math
import os
import textwrap
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from openai import OpenAI

API_BASE_URL            = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY                 = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME              = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
ENV_HOST                = os.getenv("ENV_HOST", "http://localhost:7860").rstrip("/")
TEMPERATURE             = 0.0
MAX_TOKENS              = 300
FALLBACK_ACTION         = {"action_type": "do_nothing"}
TASKS                   = ["easy", "medium", "hard"]
SUCCESS_SCORE_THRESHOLD = 0.1


def _safe_score(v: Any) -> float:
    """Clamp episode score to [0.001, 0.999] — grader contract."""
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return 0.5
    return float(max(0.001, min(0.999, float(v))))


def _safe_reward(v: Any) -> float:
    """Normalise raw RL reward to [0.0, 1.0] for [STEP] logging."""
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return 0.0
    return float(max(0.0, min(1.0, float(v))))


# ---------------------------------------------------------------------------
# Structured loggers — match sample format exactly
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Env HTTP client
# ---------------------------------------------------------------------------

class RetailEnvClient:
    def __init__(self, host: str, task: str, seed: int = 42):
        self.host = host
        self.task = task
        self.seed = seed

    def _post(self, endpoint: str, payload: Dict) -> Dict:
        data = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            f"{self.host}{endpoint}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            print(f"[DEBUG] HTTP {e.code} on {endpoint}: {body[:300]}", flush=True)
            raise
        except Exception as e:
            print(f"[DEBUG] Request error on {endpoint}: {e}", flush=True)
            raise

    def reset(self) -> Dict:
        return self._post("/reset", {"task": self.task, "seed": self.seed})

    def step(self, action: Dict) -> Dict:
        payload = {
            "task"        : self.task,
            "action_type" : action.get("action_type", "do_nothing"),
            "product_id"  : action.get("product_id"),
            "discount_pct": float(action.get("discount_pct", 0.0)),
            "reorder_qty" : int(action.get("reorder_qty", 0)),
        }
        return self._post("/step", payload)

    def close(self):
        pass


# ---------------------------------------------------------------------------
# LLM prompting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert retail store manager AI managing perishable inventory.

Each day decide ONE action:
  - discount   : Apply discount % to boost sales (use when nearing expiry)
  - reorder    : Order more units (use when stock is low)
  - remove     : Remove expired products (days_to_expiry == 0)
  - do_nothing : Take no action

Respond with ONLY valid JSON:
{
  "action_type": "discount" | "reorder" | "remove" | "do_nothing",
  "product_id": "P01",
  "discount_pct": 20.0,
  "reorder_qty": 10
}

Rules:
- do_nothing: set product_id to null
- discount_pct: 0 to 80
- reorder_qty: positive integer
- Act on ONE product per step
- Priority: remove expired > discount near-expiry > reorder low stock
""").strip()


def build_user_prompt(obs: Dict, step: int) -> str:
    lines = [
        f"Day {obs['day']}/{obs['total_days']} | "
        f"Budget: {obs['budget_remaining']:.2f} | "
        f"Revenue: {obs['cumulative_revenue']:.2f} | "
        f"Waste: {obs['cumulative_waste_cost']:.2f}",
        "", "Inventory:",
    ]
    for p in obs["products"]:
        dte  = p["days_to_expiry"]
        flag = " EXPIRED" if dte == 0 else (" NEAR EXPIRY" if dte <= 2 else "")
        lines.append(
            f"  {p['product_id']} | {p['name']} | "
            f"Stock:{p['stock']} | Price:{p['price']:.2f} | "
            f"Expiry:{dte}d{flag} | Demand:{p['demand_estimate']:.1f}"
        )
    lines += ["", "Choose ONE action. Respond with ONLY the JSON object."]
    return "\n".join(lines)


def call_llm(client, obs: Dict, step: int) -> Dict:
    if client is None:
        return FALLBACK_ACTION
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_prompt(obs, step)},
            ],
        )
        text = (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", flush=True)
        return FALLBACK_ACTION

    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    try:
        action = json.loads(text)
        if "action_type" not in action:
            raise ValueError("Missing action_type")
        return action
    except Exception as exc:
        print(f"[DEBUG] Parse error: {exc}", flush=True)
        return FALLBACK_ACTION


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(client, env: RetailEnvClient, task: str) -> float:
    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.5
    success:     bool        = False
    info:        Dict        = {}

    log_start(task=task, env="retail-inventory-expiry", model=MODEL_NAME)

    try:
        obs  = env.reset()
        done = False

        for step in range(1, 10000):
            if done:
                break

            action = call_llm(client, obs, step)
            result = env.step(action)

            obs    = result["observation"]
            done   = result["done"]
            info   = result["info"]
            reward = result["reward"]

            # Normalise raw RL reward to [0, 1] for [STEP] line
            raw_r       = reward.get("total", 0.0)
            step_reward = _safe_reward(raw_r)
            rewards.append(step_reward)
            steps_taken = step

            log_step(
                step   = step,
                action = action.get("action_type", "do_nothing"),
                reward = step_reward,
                done   = done,
                error  = None,
            )

        # Use env's graded episode_score for [END] — already in [0.001, 0.999]
        raw_score = info.get("episode_score", sum(rewards) / len(rewards) if rewards else 0.5)
        score     = _safe_score(raw_score)
        success   = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60, flush=True)
    print("  Retail Inventory & Expiry Management - Baseline Inference", flush=True)
    print("=" * 60, flush=True)
    print(f"  Model    : {MODEL_NAME}", flush=True)
    print(f"  API Base : {API_BASE_URL}", flush=True)
    print(f"  Env Host : {ENV_HOST}", flush=True)
    print("=" * 60, flush=True)

    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as exc:
        print(f"[DEBUG] OpenAI client init failed: {exc}", flush=True)
        client = None

    scores     = {}
    start_time = time.time()

    for task in TASKS:
        print(f"\n[Task: {task.upper()}]", flush=True)
        env = RetailEnvClient(host=ENV_HOST, task=task, seed=42)
        scores[task] = run_episode(client, env, task)

    elapsed = time.time() - start_time
    avg     = _safe_score(sum(scores.values()) / max(len(scores), 1))

    print("\n" + "=" * 60, flush=True)
    print("  BASELINE SCORES", flush=True)
    print("=" * 60, flush=True)
    for task, s in scores.items():
        print(f"  {task:<8} : {s:.3f}", flush=True)
    print(f"  {'average':<8} : {avg:.3f}", flush=True)
    print(f"  Elapsed  : {elapsed:.1f}s", flush=True)
    print("=" * 60, flush=True)

    result = {"scores": scores, "average": float(avg), "elapsed_seconds": elapsed}
    print(f"\nJSON_SCORES: {json.dumps(result)}", flush=True)


if __name__ == "__main__":
    main()
