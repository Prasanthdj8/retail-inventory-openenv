"""
inference.py — Baseline inference script for Retail Inventory & Expiry Management OpenEnv
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import time
import urllib.request
from typing import Any, Dict

from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy-key")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
ENV_HOST     = os.getenv("ENV_HOST", "http://localhost:7860").rstrip("/")

TEMPERATURE     = 0.0
MAX_TOKENS      = 300
FALLBACK_ACTION = {"action_type": "do_nothing"}
TASKS           = ["easy", "medium", "hard"]


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
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def reset(self) -> Dict:
        return self._post("/reset", {"task": self.task, "seed": self.seed})

    def step(self, action: Dict) -> Dict:
        return self._post("/step", {"task": self.task, **action})

    def close(self):
        pass


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
        print(f"    [LLM error] {exc} - using fallback", flush=True)
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
        print(f"    [Parse error] {exc} - using fallback", flush=True)
        return FALLBACK_ACTION


def run_episode(client, env: RetailEnvClient, task: str) -> float:
    obs           = env.reset()
    done          = False
    step          = 0
    episode_score = 0.0

    # Required structured output — START
    print(f"[START] task={task}", flush=True)

    while not done:
        step  += 1
        action = call_llm(client, obs, step)
        result = env.step(action)
        obs    = result["observation"]
        done   = result["done"]
        info   = result["info"]
        reward = result["reward"]

        # Required structured output — STEP
        print(f"[STEP] step={step} reward={reward['total']:.4f}", flush=True)

        if done:
            episode_score = info.get("episode_score", 0.0)

    # Required structured output — END
    print(f"[END] task={task} score={episode_score:.4f} steps={step}", flush=True)

    return episode_score


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
        print(f"WARNING: OpenAI client init failed: {exc}", flush=True)
        client = None

    scores     = {}
    start_time = time.time()

    for task in TASKS:
        print(f"\n[Task: {task.upper()}]", flush=True)
        env = RetailEnvClient(host=ENV_HOST, task=task, seed=42)
        try:
            scores[task] = run_episode(client, env, task)
        except Exception as exc:
            print(f"  ERROR on task '{task}': {exc}", flush=True)
            print(f"[START] task={task}", flush=True)
            print(f"[STEP] step=1 reward=0.0", flush=True)
            print(f"[END] task={task} score=0.0 steps=1", flush=True)
            scores[task] = 0.001
        finally:
            env.close()

    elapsed = time.time() - start_time
    avg     = sum(scores.values()) / max(len(scores), 1)

    print("\n" + "=" * 60, flush=True)
    print("  BASELINE SCORES", flush=True)
    print("=" * 60, flush=True)
    for task, score in scores.items():
        bar = "X" * int(score * 20)
        print(f"  {task:<8} : {score:.4f}  |{bar:<20}|", flush=True)
    print(f"  {'average':<8} : {avg:.4f}", flush=True)
    print(f"  Elapsed  : {elapsed:.1f}s", flush=True)
    print("=" * 60, flush=True)

    result = {"scores": scores, "average": avg, "elapsed_seconds": elapsed}
    print(f"\nJSON_SCORES: {json.dumps(result)}", flush=True)


if __name__ == "__main__":
    main()
