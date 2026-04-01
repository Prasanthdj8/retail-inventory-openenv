---
title: Retail Inventory & Expiry Management
emoji: 🛒
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - retail
  - inventory
  - reinforcement-learning
  - real-world
---

# Retail Inventory & Expiry Management — OpenEnv

An OpenEnv environment that simulates a store manager making real-time inventory
decisions for perishable products. The AI agent must decide **when to discount,
reorder, or remove items** nearing expiry, balancing sales revenue, food waste
reduction, and stockout avoidance.

---

## Why this environment?

Retailers, supermarkets, and pharmacies lose billions annually to expired stock and
inefficient ordering. This environment models the daily decision-making challenge
of a store manager — a genuine, high-impact task for which RL agents can provide
real value.

---

## Environment overview

| Property        | Value |
|-----------------|-------|
| Framework       | OpenEnv |
| Action space    | Discrete (4 types) + continuous parameters |
| Observation     | Product states, demand estimates, budget, day |
| Reward          | Dense (daily revenue − waste penalty − stockout penalty) |
| Episode lengths | 7 / 30 / 90 days |
| Tasks           | Easy → Medium → Hard |

---

## Action space

| Action       | Parameters                          | Description |
|--------------|-------------------------------------|-------------|
| `discount`   | `product_id`, `discount_pct` (0–80) | Apply a price discount to boost demand |
| `reorder`    | `product_id`, `reorder_qty`         | Order more units (deducted from budget) |
| `remove`     | `product_id`                        | Remove expired stock from shelf |
| `do_nothing` | —                                   | Take no action this step |

---

## Observation space

Each observation contains:

- `day` / `total_days` — current day and episode length
- `products` — list of product snapshots:
  - `product_id`, `name`, `category`
  - `stock` — units currently on shelf
  - `price`, `cost` — current price and unit cost
  - `days_to_expiry` — days until expiry (-1 = non-perishable)
  - `current_discount` — active discount percentage
  - `demand_estimate` — deterministic daily demand estimate
  - `is_expired` — whether product has already expired
- `daily_revenue`, `daily_waste_cost` — today's financials
- `cumulative_revenue`, `cumulative_waste_cost` — episode totals
- `stockout_events` — products that ran out today
- `budget_remaining` — reorder budget left

---

## Reward function

```
reward = sales_revenue − waste_penalty − stockout_penalty − reorder_cost
```

- **`sales_revenue`**: units sold × effective price (positive signal every day)
- **`waste_penalty`**: −cost × expired units (penalises letting items expire)
- **`stockout_penalty`**: −5.0 per stockout event (penalises running out)
- **`reorder_cost`**: −cost × reorder_qty (immediate reorder expenditure)

The reward is **dense** — partial progress is rewarded every day, not just at episode end.

---

## Tasks

### Easy — Single Product Management
- **Products**: 1 perishable product
- **Horizon**: 7 days
- **Goal**: Sell as much as possible before expiry; avoid waste
- **Grader**: `score = revenue_ratio × (1 − waste_ratio)` ∈ [0, 1]

### Medium — Multi-Product Balance
- **Products**: 5 perishable products
- **Horizon**: 30 days
- **Goal**: Balance revenue, waste reduction, and stockout avoidance
- **Grader**: weighted combination (revenue 40%, waste 35%, stockouts 25%)

### Hard — Full Store Optimisation
- **Products**: 20 perishable products across multiple categories
- **Horizon**: 90 days
- **Goal**: Maximise profit margin + sustainability bonus for near-zero waste
- **Grader**: weighted combination (profit 35%, waste 30%, stockouts 20%, sustainability 15%)

All graders return deterministic, reproducible scores in [0.0, 1.0].

---

## API endpoints

| Method | Endpoint   | Description |
|--------|------------|-------------|
| POST   | `/reset`   | Start/restart episode. Body: `{"task": "easy", "seed": 42}` |
| POST   | `/step`    | Take one action. Body: `{"task": "easy", "action_type": "discount", ...}` |
| GET    | `/state`   | Full internal state. Query: `?task=easy` |
| GET    | `/tasks`   | List all tasks + grader info |
| GET    | `/health`  | Liveness check |

---

## Setup & Usage

```bash
pip install -r requirements.txt
python main.py
```

```bash
docker build -t retail-inventory-openenv .
docker run -p 7860:7860 retail-inventory-openenv
```

---

## Baseline inference

```bash
export HF_TOKEN=your_token
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
python inference.py
```

---

## License

MIT
