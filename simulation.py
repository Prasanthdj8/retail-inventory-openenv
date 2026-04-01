"""
Simulation engine for the Retail Inventory & Expiry Management environment.

Handles:
- Stochastic demand generation with:
    * Day-of-week patterns (weekend uplift)
    * Monthly seasonal patterns (Christmas peak, summer produce surge etc.)
    * Category-specific seasonal boosts (strawberries in summer, dairy in winter)
    * Expiry urgency decay
    * Discount elasticity
- Daily sales resolution
- Expiry checking and waste accounting
- Reorder fulfilment
"""

from __future__ import annotations

import copy
import random
from typing import Dict, List, Tuple

from models import Action, ActionType, Product, Reward


# ---------------------------------------------------------------------------
# Catalogue helpers
# ---------------------------------------------------------------------------

def build_product_catalogue(
    num_products: int,
    episode_length: int,
    rng: random.Random,
    start_month: int = 1,
) -> List[Product]:
    templates = [
        ("Fresh Milk",        "dairy",    1.20, 0.70, 12.0, 7),
        ("Greek Yoghurt",     "dairy",    0.90, 0.50,  8.0, 10),
        ("Cheddar Cheese",    "dairy",    2.50, 1.40,  5.0, 21),
        ("Sliced Bread",      "bakery",   1.10, 0.55, 15.0, 5),
        ("Croissants x4",     "bakery",   1.80, 0.90,  6.0, 3),
        ("Chicken Breast",    "meat",     3.50, 2.10,  7.0, 4),
        ("Pork Sausages",     "meat",     2.20, 1.30,  6.0, 6),
        ("Salmon Fillet",     "fish",     4.00, 2.60,  4.0, 3),
        ("Baby Spinach",      "produce",  1.00, 0.55, 10.0, 5),
        ("Strawberries 400g", "produce",  1.50, 0.85,  9.0, 4),
        ("Orange Juice 1L",   "drinks",   1.40, 0.75,  8.0, 14),
        ("Hummus",            "deli",     1.20, 0.65,  5.0, 12),
        ("Ready Meal Pasta",  "ready",    2.80, 1.60,  6.0, 5),
        ("Eggs x12",          "dairy",    1.80, 1.00, 11.0, 21),
        ("Butter 250g",       "dairy",    1.50, 0.90,  7.0, 30),
        ("Broccoli",          "produce",  0.80, 0.45,  8.0, 6),
        ("Mixed Salad",       "produce",  1.30, 0.70,  7.0, 4),
        ("Cod Fillets",       "fish",     3.80, 2.40,  3.0, 3),
        ("Lamb Chops",        "meat",     5.50, 3.50,  3.0, 5),
        ("Cream Cheese",      "dairy",    1.60, 0.90,  4.0, 14),
    ]

    num_products = min(num_products, len(templates))
    chosen = rng.sample(templates, num_products)
    products: List[Product] = []

    for idx, (name, category, price, cost, demand, shelf_life) in enumerate(chosen):
        pid = f"P{idx+1:02d}"
        init_stock = int(demand * rng.uniform(2.0, 5.0))
        expiry_day = rng.randint(
            max(2, shelf_life // 2),
            min(shelf_life, episode_length - 1),
        )
        noisy_demand = demand * rng.uniform(0.8, 1.2)
        products.append(Product(
            product_id  = pid,
            name        = name,
            category    = category,
            stock       = init_stock,
            price       = round(price * rng.uniform(0.9, 1.1), 2),
            cost        = round(cost, 2),
            expiry_day  = expiry_day,
            base_demand = round(noisy_demand, 2),
        ))

    return products


# ---------------------------------------------------------------------------
# Seasonal demand model
# ---------------------------------------------------------------------------

# Monthly base multipliers (Jan=0 ... Dec=11)
MONTHLY_MULTIPLIERS = [
    0.85,  # Jan  - post-Christmas slump, dry January
    0.88,  # Feb  - quiet, Valentine minor boost
    0.92,  # Mar  - spring picking up, Lent fish demand
    0.95,  # Apr  - Easter boost, spring produce
    1.00,  # May  - baseline
    1.05,  # Jun  - summer starts, BBQ season
    1.10,  # Jul  - peak summer
    1.08,  # Aug  - late summer
    0.98,  # Sep  - back to routine
    1.02,  # Oct  - autumn comfort food, Halloween
    1.15,  # Nov  - pre-Christmas stockpiling
    1.30,  # Dec  - Christmas peak
]

# Category-specific seasonal boosts {month_index: multiplier}
CATEGORY_SEASONAL = {
    "produce": {
        5: 1.20, 6: 1.35, 7: 1.35, 8: 1.25,
        9: 1.10, 10: 1.05,
        0: 0.80, 1: 0.80,
    },
    "fish": {
        2: 1.25,
        5: 1.15, 6: 1.20, 7: 1.15,
        11: 1.20,
    },
    "meat": {
        5: 1.25, 6: 1.30, 7: 1.25,
        11: 1.35,
        0: 0.90,
    },
    "dairy": {
        11: 1.20, 10: 1.10,
        6: 0.90, 7: 0.90,
    },
    "drinks": {
        5: 1.30, 6: 1.40, 7: 1.40, 8: 1.25,
        11: 1.25,
        0: 0.80, 1: 0.80,
    },
    "bakery": {
        11: 1.30,
        0: 0.85,
    },
    "deli": {
        11: 1.25,
        5: 1.15, 6: 1.15,
    },
}

# Day-of-week multipliers (Mon=0 ... Sun=6)
DOW_MULTIPLIERS = [0.85, 0.85, 0.90, 0.95, 1.15, 1.25, 1.20]


def get_seasonal_factor(product: Product, current_day: int, start_month: int = 1) -> float:
    """Return the combined seasonal multiplier for a product on a given day."""
    month_idx = ((start_month - 1) + (current_day - 1) // 30) % 12

    base = MONTHLY_MULTIPLIERS[month_idx]
    category_boosts = CATEGORY_SEASONAL.get(product.category, {})
    category_boost  = category_boosts.get(month_idx, 1.0)

    # Product-level boosts
    product_boost = 1.0
    if "Strawberr" in product.name:
        if month_idx in (5, 6, 7):   product_boost = 1.50
        elif month_idx in (0, 1, 11): product_boost = 0.50
    elif "Lamb" in product.name:
        if month_idx in (2, 3):  product_boost = 1.40
        elif month_idx == 11:    product_boost = 1.30
    elif "Salmon" in product.name or "Cod" in product.name:
        if month_idx == 2:       product_boost = 1.35
    elif "Orange Juice" in product.name:
        if month_idx == 0:       product_boost = 1.30

    return base * category_boost * product_boost


def compute_demand(
    product: Product,
    current_day: int,
    rng: random.Random,
    start_month: int = 1,
) -> float:
    """
    Return units sold this day for *product*.

    Demand drivers:
    1. Base demand (product mean)
    2. Monthly seasonal pattern
    3. Category-specific seasonal boost
    4. Product-specific seasonal boost
    5. Day-of-week pattern
    6. Discount elasticity (each 10% off -> +15% demand)
    7. Expiry urgency decay
    8. Gaussian noise (sigma = 20%)
    """
    if product.stock == 0:
        return 0.0

    dte = product.days_to_expiry(current_day)
    if dte == -1:       expiry_factor = 1.0
    elif dte <= 1:      expiry_factor = 0.3
    elif dte <= 2:      expiry_factor = 0.6
    else:               expiry_factor = 1.0

    discount_factor = 1.0 + (product.current_discount / 10.0) * 0.15
    dow_factor      = DOW_MULTIPLIERS[(current_day - 1) % 7]
    seasonal_factor = get_seasonal_factor(product, current_day, start_month)

    adjusted = (
        product.base_demand
        * expiry_factor
        * discount_factor
        * dow_factor
        * seasonal_factor
    )

    noisy = rng.gauss(adjusted, adjusted * 0.2)
    return max(0.0, min(float(product.stock), noisy))


def estimate_demand(product: Product, current_day: int, start_month: int = 1) -> float:
    """Deterministic demand estimate shown in Observation (no noise)."""
    dte = product.days_to_expiry(current_day)
    expiry_factor   = 1.0 if dte == -1 else (0.3 if dte <= 1 else 0.6 if dte <= 2 else 1.0)
    discount_factor = 1.0 + (product.current_discount / 10.0) * 0.15
    dow_factor      = DOW_MULTIPLIERS[(current_day - 1) % 7]
    seasonal_factor = get_seasonal_factor(product, current_day, start_month)
    return round(
        product.base_demand * expiry_factor * discount_factor * dow_factor * seasonal_factor, 2
    )


# ---------------------------------------------------------------------------
# Step logic
# ---------------------------------------------------------------------------

def apply_action(
    product: Product,
    action: Action,
    current_day: int,
    budget: float,
) -> Tuple[Product, float, float]:
    p = copy.deepcopy(product)
    reward_delta = 0.0

    if action.action_type == ActionType.DISCOUNT:
        p.current_discount = action.discount_pct

    elif action.action_type == ActionType.REORDER:
        reorder_cost = p.cost * action.reorder_qty
        if reorder_cost <= budget:
            p.pending_reorder += action.reorder_qty
            budget -= reorder_cost
            reward_delta -= reorder_cost

    elif action.action_type == ActionType.REMOVE:
        removed = p.stock
        p.stock = 0
        reward_delta -= p.cost * removed * 0.5

    return p, reward_delta, budget


def resolve_day(
    products: List[Product],
    current_day: int,
    rng: random.Random,
    start_month: int = 1,
) -> Tuple[List[Product], float, float, int]:
    """Simulate one day of trading after actions have been applied."""
    updated: List[Product] = []
    total_revenue = 0.0
    total_waste   = 0.0
    stockouts     = 0

    for p in products:
        p = copy.deepcopy(p)

        p.stock += p.pending_reorder
        p.pending_reorder = 0

        units_float = compute_demand(p, current_day, rng, start_month)
        units_sold  = min(int(units_float), p.stock)
        effective_price = p.price * (1 - p.current_discount / 100.0)
        total_revenue  += units_sold * effective_price
        p.stock        -= units_sold

        if units_float > p.stock + units_sold:
            stockouts += 1

        dte = p.days_to_expiry(current_day)
        if dte == 0 and p.expiry_day != -1 and p.stock > 0:
            total_waste += p.cost * p.stock
            p.stock = 0

        p.current_discount = 0.0
        updated.append(p)

    return updated, total_revenue, total_waste, stockouts
