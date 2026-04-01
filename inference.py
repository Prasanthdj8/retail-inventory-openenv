"""
inference.py — Baseline inference script for Retail Inventory & Expiry Management OpenEnv
==========================================================================================
MANDATORY REQUIREMENTS (per hackathon spec):
  - Named inference.py and placed in the project root
  - Uses OpenAI client for all LLM calls
  - Reads credentials from environment variables:
      API_BASE_URL  → LLM endpoint  (default: https://router.huggingface.co/v1)
      MODEL_NAME    → Model to use
      HF_TOKEN      → HuggingFace / API key
  - Produces reproducible baseline scores on all 3 tasks
  - Must complete in < 20 minutes on 2 vCPU / 8 GB RAM

Usage:
    python inference.py

Environment variables:
    API_BASE_URL   (required) LLM API base URL
    MODEL_NAME     (required) model identifier
    HF_TOKEN       (required) API key
    ENV_HOST       (optional) OpenEnv server host (default: http://localhost:7860)
    MAX_STEPS      (optional) max steps per episode (default: uses task episode length)
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "")
ENV_HOST     = os.getenv("ENV_HOST", "http://localhost:7860").rstrip("/")

TEMPERATURE  = 0.0        # Deterministic for reproducibility
MAX_TOKENS   = 300
FALLBACK_ACTION = {"action_type": "do_nothing"}

TASKS = ["easy", "medium", "hard"]


# ---------------------------------------------------------------------------
# OpenEnv HTTP client
# ---------------------------------------------------------------------------

class RetailEnvClient:
    """Thin HTTP wrapper around the OpenEnv FastAPI server."""

    def __init__(self, host: str, task: str, seed: int = 42):
        self.host = host
        self.task = task
        self.seed = seed
        self._http = httpx.Client(timeout=30.0)

    def reset(self) -> Dict[str, Any]:
        r = self._http.post(
            f"{self.host}/reset",
            json={"task": self.task, "seed": self.seed},
        )
        r.raise_for_status()
        return r.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"task": self.task, **action}
        r = self._http.post(f"{self.host}/step", json=payload)
        r.raise_for_status()
        return r.json()

    def state(self) -> Dict[str, Any]:
        r = self._http.get(f"{self.host}/state", params={"task": self.task})
        r.raise_for_status()
        return r.json()

    def close(self):
        self._http.close()


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert retail store manager AI. Your job is to manage perishable
inventory to maximise sales revenue and minimise food waste.

Each day you observe the store's inventory and must decide ONE action:
  - discount   : Apply a discount % to a product to boost sales (use when nearing expiry)
  - reorder    : Order more units of a product (use when stock is low but expiry is far)
  - remove     : Remove expired products from shelf (use when days_to_expiry == 0)
  - do_nothing : Take no action

Respond with ONLY valid JSON matching this schema (no explanation, no markdown):
{
  "action_type": "discount" | "reorder" | "remove" | "do_nothing",
  "product_id": "P01",
  "discount_pct": 20.0,
  "reorder_qty": 10
}

Rules:
- If action_type is "do_nothing", omit product_id or set it to null.
- discount_pct must be between 0 and 80.
- reorder_qty must be a positive integer.
- Only act on ONE product per step.
- Prioritise removing expired items, then discounting near-expiry items.
""").strip()


def build_user_prompt(obs: Dict[str, Any], step: int) -> str:
    day        = obs["day"]
    total_days = obs["total_days"]
    budget     = obs["budget_remaining"]
    cum_rev    = obs["cumulative_revenue"]
    cum_waste  = obs["cumulative_waste_cost"]
    products   = obs["products"]

    lines = [
        f"Day {day}/{total_days} | Budget: £{budget:.2f} | "
        f"Revenue so far: £{cum_rev:.2f} | Waste so far: £{cum_waste:.2f}",
        "",
        "Current inventory:",
    ]

    for p in products:
        dte  = p["days_to_expiry"]
        flag = ""
        if dte == 0:
            flag = " ⚠ EXPIRED"
        elif dte <= 2:
            flag = " ⚠ NEAR EXPIRY"
        lines.append(
            f"  {p['product_id']} | {p['name']:<22} | "
            f"Stock: {p['stock']:>3} | "
            f"Price: £{p['price']:.2f} | "
            f"Days to expiry: {dte:>3}{flag} | "
            f"Est. demand/day: {p['demand_estimate']:.1f}"
        )

    lines += [
        "",
        "Choose the single best action for this time step.",
        "Respond with ONLY the JSON object.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM call + action parsing
# ---------------------------------------------------------------------------

def call_llm(client: OpenAI, obs: Dict[str, Any], step: int) -> Dict[str, Any]:
    """Ask the LLM what action to take and parse the JSON response."""
    user_prompt = build_user_prompt(obs, step)

    try:
        completion = client.chat.completions.create(
            model       = MODEL_NAME,
            temperature = TEMPERATURE,
            max_tokens  = MAX_TOKENS,
            stream      = False,
            messages    = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
        )
        text = (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"    [LLM error] {exc} — using fallback action")
        return FALLBACK_ACTION

    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    try:
        action = json.loads(text)
        # Validate required field
        if "action_type" not in action:
            raise ValueError("Missing action_type")
        return action
    except Exception as exc:
        print(f"    [Parse error] {exc} | raw: {text[:80]} — using fallback")
        return FALLBACK_ACTION


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    client   : OpenAI,
    env      : RetailEnvClient,
    task     : str,
    verbose  : bool = True,
) -> float:
    """Run one full episode and return the final score."""
    obs  = env.reset()
    done = False
    step = 0
    episode_score = 0.0

    if verbose:
        print(f"  Episode started | {len(obs['products'])} products | "
              f"{obs['total_days']} days | Budget: £{obs['budget_remaining']:.2f}")

    while not done:
        step += 1
        action = call_llm(client, obs, step)

        # Inject task into action payload
        result = env.step(action)

        obs    = result["observation"]
        reward = result["reward"]
        done   = result["done"]
        info   = result["info"]

        if verbose and step % 5 == 0:
            print(
                f"    Day {info['day']:>3} | "
                f"reward: {reward['total']:+.2f} | "
                f"cum_rev: £{obs['cumulative_revenue']:.2f} | "
                f"cum_waste: £{obs['cumulative_waste_cost']:.2f}"
            )

        if done:
            episode_score = info.get("episode_score", 0.0)

    if verbose:
        print(f"  Episode done | Final score: {episode_score:.4f}")

    return episode_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Validate required env vars
    missing = []
    if not API_KEY:
        missing.append("HF_TOKEN (or API_KEY)")
    if not MODEL_NAME:
        missing.append("MODEL_NAME")
    if missing:
        print(f"ERROR: Missing required environment variables: {', '.join(missing)}")
        print("Set them before running:\n"
              "  export HF_TOKEN=your_key\n"
              "  export MODEL_NAME=your_model\n"
              "  export API_BASE_URL=https://router.huggingface.co/v1")
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print("=" * 60)
    print("  Retail Inventory & Expiry Management — Baseline Inference")
    print("=" * 60)
    print(f"  Model      : {MODEL_NAME}")
    print(f"  API Base   : {API_BASE_URL}")
    print(f"  Env Host   : {ENV_HOST}")
    print("=" * 60)

    scores: Dict[str, float] = {}
    start_time = time.time()

    for task in TASKS:
        print(f"\n[Task: {task.upper()}]")
        env = RetailEnvClient(host=ENV_HOST, task=task, seed=42)
        try:
            score = run_episode(client, env, task, verbose=True)
            scores[task] = score
        except Exception as exc:
            print(f"  ERROR running task '{task}': {exc}")
            scores[task] = 0.0
        finally:
            env.close()

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("  BASELINE SCORES")
    print("=" * 60)
    for task, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"  {task:<8} : {score:.4f}  |{bar:<20}|")
    avg = sum(scores.values()) / len(scores)
    print(f"  {'average':<8} : {avg:.4f}")
    print(f"  Elapsed time : {elapsed:.1f}s")
    print("=" * 60)

    # Machine-readable output for automated validators
    result = {"scores": scores, "average": avg, "elapsed_seconds": elapsed}
    print(f"\nJSON_SCORES: {json.dumps(result)}")


if __name__ == "__main__":
    main()
