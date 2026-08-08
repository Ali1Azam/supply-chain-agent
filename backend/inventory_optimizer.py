"""
Inventory Optimization (Methodology step 3).

Implements the classic formulas an ops team would recognize, driven by the
statistical demand estimates coming out of forecasting.py:

  Safety Stock (SS) = z * sigma_demand * sqrt(lead_time_days)
  Reorder Point (ROP) = avg_daily_demand * lead_time_days + SS
  Economic Order Qty (EOQ) = sqrt( (2 * D * S) / H )
      D = annual demand, S = ordering cost per PO, H = annual holding cost / unit

Also exposes a chain-of-thought style explanation string, since the agent
uses this reasoning when it decides to cut a purchase order.
"""
import math
from dataclasses import dataclass
from datetime import date, timedelta

from backend.config import DEFAULT_SERVICE_LEVEL_Z, DEFAULT_ORDERING_COST, DEFAULT_HOLDING_COST_PCT


@dataclass
class InventoryPlan:
    avg_daily_demand: float
    demand_std: float
    lead_time_days: int
    safety_stock: int
    reorder_point: int
    economic_order_qty: int
    current_stock: int
    predicted_stockout_date: "date | None"
    days_of_cover: float
    reasoning: str


def compute_safety_stock(demand_std: float, lead_time_days: int, z: float = DEFAULT_SERVICE_LEVEL_Z) -> int:
    return max(0, round(z * demand_std * math.sqrt(max(lead_time_days, 1))))


def compute_reorder_point(avg_daily_demand: float, lead_time_days: int, safety_stock: int) -> int:
    return max(0, round(avg_daily_demand * lead_time_days + safety_stock))


def compute_eoq(annual_demand: float, ordering_cost: float = DEFAULT_ORDERING_COST,
                 unit_cost: float = 1.0, holding_cost_pct: float = DEFAULT_HOLDING_COST_PCT) -> int:
    holding_cost = max(unit_cost * holding_cost_pct, 0.01)
    if annual_demand <= 0:
        return 0
    eoq = math.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
    return max(1, round(eoq))


def predict_stockout_date(current_stock: int, forecast_df) -> "date | None":
    """
    forecast_df: DataFrame with columns ['ds', 'yhat'] (daily forecast, future only,
    sorted ascending). Walks the forecast forward, depleting stock by yhat units/day,
    and returns the first date stock would hit zero, or None if it doesn't within
    the forecast horizon.
    """
    remaining = current_stock
    for _, row in forecast_df.iterrows():
        remaining -= max(0, row["yhat"])
        if remaining <= 0:
            return row["ds"].date() if hasattr(row["ds"], "date") else row["ds"]
    return None


def build_inventory_plan(current_stock: int, avg_daily_demand: float, demand_std: float,
                          lead_time_days: int, unit_cost: float, forecast_df=None,
                          z: float = DEFAULT_SERVICE_LEVEL_Z) -> InventoryPlan:
    safety_stock = compute_safety_stock(demand_std, lead_time_days, z)
    rop = compute_reorder_point(avg_daily_demand, lead_time_days, safety_stock)
    annual_demand = avg_daily_demand * 365
    eoq = compute_eoq(annual_demand, unit_cost=unit_cost)

    days_of_cover = round(current_stock / avg_daily_demand, 1) if avg_daily_demand > 0 else float("inf")

    stockout_date = None
    if forecast_df is not None and len(forecast_df) > 0:
        stockout_date = predict_stockout_date(current_stock, forecast_df)

    reasoning = (
        f"Current stock is {current_stock} units. Forecasted average daily demand is "
        f"{avg_daily_demand:.1f} units (σ={demand_std:.1f}). Supplier lead time is "
        f"{lead_time_days} days, so safety stock at the target service level is "
        f"{safety_stock} units, giving a reorder point of {rop} units. "
        f"At current depletion the product has ~{days_of_cover} days of cover"
        + (f", with a predicted stockout on {stockout_date}." if stockout_date else ".")
    )

    return InventoryPlan(
        avg_daily_demand=avg_daily_demand,
        demand_std=demand_std,
        lead_time_days=lead_time_days,
        safety_stock=safety_stock,
        reorder_point=rop,
        economic_order_qty=eoq,
        current_stock=current_stock,
        predicted_stockout_date=stockout_date,
        days_of_cover=days_of_cover,
        reasoning=reasoning,
    )
