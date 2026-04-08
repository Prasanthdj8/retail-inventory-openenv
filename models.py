"""
Typed Pydantic models for the Retail Inventory & Expiry Management OpenEnv.
Defines Observation, Action, Reward, and supporting data structures.
"""

from __future__ import annotations
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    DISCOUNT   = "discount"    # Apply a percentage discount to a product
    REORDER    = "reorder"     # Place a restock order for a product
    REMOVE     = "remove"      # Remove expired / unsellable product from shelf
    DO_NOTHING = "do_nothing"  # Take no action this step


class Action(BaseModel):
    """
    One action taken by the agent per time-step.

    Fields
    ------
    action_type : ActionType
        What kind of action to take.
    product_id : str
        Which product to act on.  Required for all actions except DO_NOTHING.
    discount_pct : float
        Discount percentage (0–80).  Only used when action_type == DISCOUNT.
    reorder_qty : int
        Number of units to reorder.  Only used when action_type == REORDER.
    """
    action_type : ActionType        = Field(..., description="Type of action")
    product_id  : Optional[str]     = Field(None, description="Target product ID")
    discount_pct: float             = Field(0.0, ge=0.0, le=80.0,
                                            description="Discount percentage (0-80)")
    reorder_qty : int               = Field(0,   ge=0,
                                            description="Units to reorder")


# ---------------------------------------------------------------------------
# Product snapshot (inside observation)
# ---------------------------------------------------------------------------

class ProductState(BaseModel):
    """Snapshot of a single product visible to the agent."""
    product_id      : str
    name            : str
    category        : str
    stock           : int           = Field(..., ge=0)
    price           : float         = Field(..., gt=0)
    cost            : float         = Field(..., gt=0)
    days_to_expiry  : int           = Field(..., description="Days until expiry; -1 = non-perishable")
    current_discount: float         = Field(0.0, ge=0.0, le=80.0)
    demand_estimate : float         = Field(..., description="Estimated daily demand (units)")
    is_expired      : bool          = False


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """
    Everything the agent can see at each time-step.

    Fields
    ------
    day              : Current simulation day (1-indexed).
    total_days       : Episode length in days.
    products         : List of product snapshots.
    daily_revenue    : Revenue earned on the current day.
    daily_waste_cost : Cost of items wasted (expired / removed) today.
    cumulative_revenue    : Total revenue so far this episode.
    cumulative_waste_cost : Total waste cost so far this episode.
    stockout_events  : Number of times any product ran out of stock today.
    budget_remaining : Remaining budget for reorder operations.
    """
    day                  : int
    total_days           : int
    products             : List[ProductState]
    daily_revenue        : float = 0.0
    daily_waste_cost     : float = 0.0
    cumulative_revenue   : float = 0.0
    cumulative_waste_cost: float = 0.0
    stockout_events      : int   = 0
    budget_remaining     : float = 0.0
    reward               : float = 0.001  # clamped step reward for OpenEnv standard


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    """
    Decomposed reward signal returned after each step.

    Components
    ----------
    sales_revenue   : +ve reward for units sold.
    waste_penalty   : -ve penalty for expired / removed units.
    stockout_penalty: -ve penalty for stockout events.
    reorder_cost    : -ve cost of placing a reorder.
    total           : Sum of all components.
    """
    sales_revenue   : float = 0.0
    waste_penalty   : float = 0.0
    stockout_penalty: float = 0.0
    reorder_cost    : float = 0.0
    total           : float = 0.0

    def compute_total(self) -> "Reward":
        self.total = (
            self.sales_revenue
            + self.waste_penalty
            + self.stockout_penalty
            + self.reorder_cost
        )
        return self


# ---------------------------------------------------------------------------
# Internal full product state (not exposed directly to agent)
# ---------------------------------------------------------------------------

class Product(BaseModel):
    """Full internal product state used by the simulation engine."""
    product_id      : str
    name            : str
    category        : str
    stock           : int
    price           : float
    cost            : float
    expiry_day      : int           # Absolute day when product expires (-1 = never)
    base_demand     : float         # Mean daily demand at full price
    current_discount: float = 0.0
    reorder_lead    : int   = 1     # Days until reorder arrives
    pending_reorder : int   = 0     # Units in transit

    def days_to_expiry(self, current_day: int) -> int:
        if self.expiry_day == -1:
            return -1
        return max(0, self.expiry_day - current_day)

    def to_product_state(self, current_day: int, demand_estimate: float) -> ProductState:
        dte = self.days_to_expiry(current_day)
        return ProductState(
            product_id      = self.product_id,
            name            = self.name,
            category        = self.category,
            stock           = self.stock,
            price           = self.price,
            cost            = self.cost,
            days_to_expiry  = dte,
            current_discount= self.current_discount,
            demand_estimate = round(demand_estimate, 2),
            is_expired      = (dte == 0 and self.expiry_day != -1),
        )
