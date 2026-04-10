"""
Retail Inventory & Expiry Management — OpenEnv Environment

Implements the full OpenEnv interface:
    reset()  -> Observation
    step()   -> (Observation, Reward, done, info)
    state()  -> dict (full internal state)
"""

from __future__ import annotations
import math

import copy
import random
from typing import Any, Dict, List, Optional, Tuple

from graders import (
    EasyGrader,
    EpisodeHistory,
    HardGrader,
    MediumGrader,
    StepRecord,
)
from models import (
    Action,
    ActionType,
    Observation,
    Product,
    Reward,
)
from simulation import (
    apply_action,
    build_product_catalogue,
    estimate_demand,
    resolve_day,
)

# ---------------------------------------------------------------------------
# Safe clamping — strictly (0, 1), same bounds as inference.py
# 1e-4 is far enough from the boundary that :.6f never rounds to 0.000000
# ---------------------------------------------------------------------------
_LO = 0.001
_HI = 0.999


def _safe(v: Any) -> float:
    """Return a float strictly in (_LO, _HI), safe against None / NaN / inf."""
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return 0.5
    return float(max(_LO, min(_HI, float(v))))


# ---------------------------------------------------------------------------
# Task configurations
# ---------------------------------------------------------------------------

TASK_CONFIGS = {
    "easy": {
        "num_products"   : 1,
        "episode_length" : 7,
        "initial_budget" : 50.0,
        "start_month"    : 6,
        "description"    : "1 product · 7 days · minimise waste",
    },
    "medium": {
        "num_products"   : 5,
        "episode_length" : 30,
        "initial_budget" : 300.0,
        "start_month"    : 11,
        "description"    : "5 products · 30 days · balance waste & stockouts",
    },
    "hard": {
        "num_products"   : 20,
        "episode_length" : 90,
        "initial_budget" : 2000.0,
        "start_month"    : 10,
        "description"    : "20 products · 90 days · maximise profit + sustainability",
    },
}


class RetailInventoryEnv:
    """
    OpenEnv-compliant environment for retail inventory & expiry management.

    Parameters
    ----------
    task  : "easy" | "medium" | "hard"
    seed  : Random seed for reproducibility (default 42).
    """

    name        = "retail-inventory-expiry"
    version     = "1.0.0"
    description = (
        "Simulates a store manager making real-time inventory decisions "
        "for perishable products: when to discount, reorder, or remove items "
        "nearing expiry. Features realistic seasonal demand patterns "
        "(Christmas peak, summer produce surge, Lent fish demand) on top of "
        "day-of-week patterns and discount elasticity."
    )

    def __init__(self, task: str = "easy", seed: int = 42):
        if task not in TASK_CONFIGS:
            raise ValueError(f"task must be one of {list(TASK_CONFIGS)}")

        self.task    = task
        self.seed    = seed
        self.config  = TASK_CONFIGS[task]
        self._rng    = random.Random(seed)

        self._products      : List[Product] = []
        self._day           : int   = 0
        self._done          : bool  = False
        self._budget        : float = 0.0
        self._cum_revenue   : float = 0.0
        self._cum_waste     : float = 0.0
        self._stockout_total: int   = 0
        self._history       : EpisodeHistory = EpisodeHistory()
        self._grader        = None

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        self._rng = random.Random(self.seed)

        cfg = self.config
        self._products = build_product_catalogue(
            num_products   = cfg["num_products"],
            episode_length = cfg["episode_length"],
            rng            = self._rng,
            start_month    = cfg["start_month"],
        )
        self._day            = 1
        self._done           = False
        self._budget         = cfg["initial_budget"]
        self._cum_revenue    = 0.0
        self._cum_waste      = 0.0
        self._stockout_total = 0
        self._history        = EpisodeHistory(
            total_days     = cfg["episode_length"],
            initial_budget = cfg["initial_budget"],
        )

        theo_max = self._theoretical_max_revenue()
        if self.task == "easy":
            self._grader = EasyGrader(theo_max)
        elif self.task == "medium":
            self._grader = MediumGrader(theo_max, cfg["num_products"])
        else:
            self._grader = HardGrader(
                theo_max,
                cfg["num_products"],
                total_possible_expirations=cfg["num_products"] * cfg["episode_length"],
            )

        return self._build_observation(0.0, 0.0, 0)

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        reward = Reward()

        if action.action_type != ActionType.DO_NOTHING and action.product_id:
            pid_map = {p.product_id: i for i, p in enumerate(self._products)}
            if action.product_id in pid_map:
                idx = pid_map[action.product_id]
                updated_p, r_delta, self._budget = apply_action(
                    self._products[idx], action, self._day, self._budget
                )
                self._products[idx] = updated_p
                reward.reorder_cost  += r_delta if action.action_type == ActionType.REORDER else 0.0
                reward.waste_penalty += r_delta if action.action_type == ActionType.REMOVE  else 0.0

        self._products, daily_rev, daily_waste, stockouts = resolve_day(
            self._products, self._day, self._rng,
            start_month=self.config["start_month"],
        )

        self._cum_revenue    += daily_rev
        self._cum_waste      += daily_waste
        self._stockout_total += stockouts

        reward.sales_revenue    = daily_rev
        reward.waste_penalty   -= daily_waste
        reward.stockout_penalty = -5.0 * stockouts
        reward.compute_total()

        expired_today = sum(
            1 for p in self._products
            if p.days_to_expiry(self._day) == 0 and p.expiry_day != -1
        )
        self._history.records.append(StepRecord(
            day              = self._day,
            revenue          = daily_rev,
            waste_cost       = daily_waste,
            stockout_events  = stockouts,
            budget_remaining = self._budget,
            products_expired = expired_today,
            total_stock      = sum(p.stock for p in self._products),
        ))

        self._day += 1
        self._done = self._day > self.config["episode_length"]

        obs           = self._build_observation(daily_rev, daily_waste, stockouts, reward.total)
        running_score = self.current_score()

        info: Dict[str, Any] = {
            "day"          : self._day - 1,
            "done"         : self._done,
            "budget"       : self._budget,
            "cum_revenue"  : self._cum_revenue,
            "cum_waste"    : self._cum_waste,
            "episode_score": running_score,   # already safe via current_score()
            "task_id"      : self._grader.task_id if self._grader else self.task,
        }

        return obs, reward, self._done, info

    def state(self) -> Dict[str, Any]:
        return {
            "task"           : self.task,
            "day"            : self._day,
            "done"           : self._done,
            "budget"         : self._budget,
            "cum_revenue"    : self._cum_revenue,
            "cum_waste"      : self._cum_waste,
            "stockout_total" : self._stockout_total,
            "start_month"    : self.config["start_month"],
            "products"       : [p.model_dump() for p in self._products],
            "episode_length" : self.config["episode_length"],
            "history_length" : len(self._history.records),
            "episode_score"  : self.current_score(),
            "task_id"        : self._grader.task_id if self._grader else self.task,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        daily_revenue: float,
        daily_waste: float,
        stockouts: int,
        reward_total: float = 0.5,
    ) -> Observation:
        product_states = [
            p.to_product_state(
                self._day,
                estimate_demand(p, self._day, self.config["start_month"]),
            )
            for p in self._products
        ]
        # reward embedded in observation must also be strictly (0, 1)
        clamped_reward = round(_safe(reward_total), 6)
        return Observation(
            day                   = self._day,
            total_days            = self.config["episode_length"],
            products              = product_states,
            daily_revenue         = round(daily_revenue, 4),
            daily_waste_cost      = round(daily_waste, 4),
            cumulative_revenue    = round(self._cum_revenue, 4),
            cumulative_waste_cost = round(self._cum_waste, 4),
            stockout_events       = stockouts,
            budget_remaining      = round(self._budget, 4),
            reward                = clamped_reward,
        )

    def _theoretical_max_revenue(self) -> float:
        total = 0.0
        days  = self.config["episode_length"]
        for p in self._products:
            total += p.stock * p.price
            total += p.base_demand * days * p.price
        return max(total, 1.0)

    def get_task_info(self) -> Dict[str, Any]:
        return self._grader.info() if self._grader else {}

    def current_score(self) -> float:
        """Return the running episode score, always strictly in (_LO, _HI)."""
        if not self._grader or not self._history.records:
            return 0.5
        raw = self._grader.score(self._history)
        return _safe(raw)
