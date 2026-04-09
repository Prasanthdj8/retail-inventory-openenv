"""
Task graders for the Retail Inventory & Expiry Management OpenEnv.

Each grader receives the full episode history and returns a score in [0.0, 1.0].

Tasks
-----
task_easy   : Single product, 7-day horizon — minimise waste.
task_medium : 5 products, 30-day horizon — balance waste & stockouts.
task_hard   : 20 products, 90-day horizon — maximise profit + sustainability.
"""

from __future__ import annotations
import math

from dataclasses import dataclass, field
from typing import Dict, List



def safe_score(raw: float) -> float:
    """Return a score strictly in (0, 1), safe against NaN/inf."""
    if raw is None or math.isnan(raw) or math.isinf(raw):
        return 0.5
    return float(max(1e-6, min(1 - 1e-6, raw)))

# ---------------------------------------------------------------------------
# Episode history record
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    """One row of episode history recorded after each step."""
    day               : int
    revenue           : float
    waste_cost        : float
    stockout_events   : int
    budget_remaining  : float
    products_expired  : int   = 0   # products that hit expiry this day
    total_stock       : int   = 0


@dataclass
class EpisodeHistory:
    """Accumulated across all steps of an episode."""
    records           : List[StepRecord] = field(default_factory=list)
    total_days        : int = 0
    initial_budget    : float = 0.0

    # Aggregate helpers
    @property
    def total_revenue(self) -> float:
        return sum(r.revenue for r in self.records)

    @property
    def total_waste(self) -> float:
        return sum(r.waste_cost for r in self.records)

    @property
    def total_stockouts(self) -> int:
        return sum(r.stockout_events for r in self.records)

    @property
    def total_expired_products(self) -> int:
        return sum(r.products_expired for r in self.records)


# ---------------------------------------------------------------------------
# Grader base
# ---------------------------------------------------------------------------

class BaseGrader:
    task_id     : str = ""
    description : str = ""
    difficulty  : str = ""

    def score(self, history: EpisodeHistory) -> float:
        raise NotImplementedError

    def info(self) -> Dict:
        return {
            "task_id"    : self.task_id,
            "description": self.description,
            "difficulty" : self.difficulty,
        }


# ---------------------------------------------------------------------------
# Task 1 — Easy
# ---------------------------------------------------------------------------

class EasyGrader(BaseGrader):
    """
    Single product, 7-day horizon.

    Score = fraction of total potential revenue actually earned,
            penalised by waste ratio.

    Score breakdown:
        revenue_ratio  = actual_revenue / theoretical_max_revenue  [0, 1]
        waste_ratio    = total_waste / (total_waste + total_revenue + 1e-9) [0, 1]
        score          = revenue_ratio * (1 - waste_ratio)
    """
    task_id     = "easy_single_product"
    description = "Manage 1 perishable product over 7 days. Maximise sales, minimise waste."
    difficulty  = "easy"

    def __init__(self, theoretical_max_revenue: float):
        self.theoretical_max = max(theoretical_max_revenue, 1.0)

    def score(self, history: EpisodeHistory) -> float:
        rev   = history.total_revenue
        waste = history.total_waste

        revenue_ratio = min(rev / self.theoretical_max, 1.0)
        waste_ratio   = waste / (waste + rev + 1e-9)

        raw = revenue_ratio * (1.0 - waste_ratio)
        return safe_score(raw)


# ---------------------------------------------------------------------------
# Task 2 — Medium
# ---------------------------------------------------------------------------

class MediumGrader(BaseGrader):
    """
    5 products, 30-day horizon.

    Score is a weighted combination of:
        - Revenue efficiency  (40 %)
        - Waste minimisation  (35 %)
        - Stockout avoidance  (25 %)

    waste_score    = 1 - clamp(waste / (revenue + waste + 1), 0, 1)
    stockout_score = 1 - clamp(stockouts / (total_days * num_products), 0, 1)
    revenue_score  = clamp(revenue / theoretical_max, 0, 1)
    """
    task_id     = "medium_multi_product"
    description = (
        "Manage 5 perishable products over 30 days. "
        "Balance revenue, waste reduction, and stockout avoidance."
    )
    difficulty  = "medium"

    def __init__(self, theoretical_max_revenue: float, num_products: int = 5):
        self.theoretical_max = max(theoretical_max_revenue, 1.0)
        self.num_products    = num_products

    def score(self, history: EpisodeHistory) -> float:
        rev       = history.total_revenue
        waste     = history.total_waste
        stockouts = history.total_stockouts
        days      = max(history.total_days, 1)

        revenue_score  = min(rev / self.theoretical_max, 1.0)
        waste_score    = 1.0 - min(waste / (rev + waste + 1e-9), 1.0)
        max_stockouts  = days * self.num_products
        stockout_score = 1.0 - min(stockouts / max(max_stockouts, 1), 1.0)

        raw = (
            0.40 * revenue_score
            + 0.35 * waste_score
            + 0.25 * stockout_score
        )
        return safe_score(raw)


# ---------------------------------------------------------------------------
# Task 3 — Hard
# ---------------------------------------------------------------------------

class HardGrader(BaseGrader):
    """
    20 products, 90-day horizon.

    Score is a weighted combination of:
        - Profit margin efficiency (35 %)
        - Waste minimisation       (30 %)
        - Stockout avoidance       (20 %)
        - Sustainability bonus     (15 %)
            → sustainability = 1 - (expired_products / total_possible_expirations)

    Profit margin = (revenue - waste - reorder_costs) / theoretical_max_revenue
    """
    task_id     = "hard_full_store"
    description = (
        "Manage 20 perishable products over 90 days. "
        "Maximise profit margin, minimise waste, avoid stockouts, "
        "and earn sustainability bonus for near-zero food waste."
    )
    difficulty  = "hard"

    def __init__(
        self,
        theoretical_max_revenue: float,
        num_products: int = 20,
        total_possible_expirations: int = 1,
    ):
        self.theoretical_max              = max(theoretical_max_revenue, 1.0)
        self.num_products                 = num_products
        self.total_possible_expirations   = max(total_possible_expirations, 1)

    def score(self, history: EpisodeHistory) -> float:
        rev       = history.total_revenue
        waste     = history.total_waste
        stockouts = history.total_stockouts
        expired   = history.total_expired_products
        days      = max(history.total_days, 1)

        profit_score = min(
            max((rev - waste) / self.theoretical_max, 0.0),
            1.0,
        )
        waste_score = 1.0 - min(waste / (rev + waste + 1e-9), 1.0)

        max_stockouts  = days * self.num_products
        stockout_score = 1.0 - min(stockouts / max(max_stockouts, 1), 1.0)

        sustainability = 1.0 - min(
            expired / self.total_possible_expirations, 1.0
        )

        raw = (
            0.35 * profit_score
            + 0.30 * waste_score
            + 0.20 * stockout_score
            + 0.15 * sustainability
        )
        return safe_score(raw)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: Dict[str, type] = {
    "easy"  : EasyGrader,
    "medium": MediumGrader,
    "hard"  : HardGrader,
}
