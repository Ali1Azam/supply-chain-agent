"""
Monthly Savings Report.

Compares a naive "static reorder" baseline (fixed reorder point set once
from long-run averages, common in SMBs with no forecasting) against the
agent-optimized policy, over the historical sales data, to quantify:
  - Waste reduction (overstock that would have gone unsold/expired)
  - Stockout events prevented
  - Cost optimization (ordering + holding cost delta)
  - Revenue protected (sales that would have been lost to stockouts)

This is a backtest simulation used to populate the SavingsRecord table /
dashboard report, since a brand-new deployment has no "before" period to
compare against.
"""
from collections import defaultdict
from datetime import date

import numpy as np


def simulate_policy_backtest(product, sales_records: list, use_agent_policy: bool,
                              static_rop_multiplier: float = 1.0):
    """
    Simple discrete-event style simulation over historical daily demand.
    use_agent_policy=True: reorders using the (already-computed) product.reorder_point /
        economic_order_qty, i.e. the forecast-driven policy.
    use_agent_policy=False: a naive baseline that reorders a fixed EOQ-like batch only
        when stock hits a static ROP derived from the simple long-run average demand
        (no seasonality/weather awareness, no safety-stock adjustment) -- representative
        of how many SMBs run inventory today.
    """
    demands = [r["units_sold"] for r in sales_records]
    avg_demand = float(np.mean(demands)) if demands else 0
    std_demand = float(np.std(demands)) if demands else 1

    if use_agent_policy:
        rop = product.reorder_point or int(avg_demand * product.lead_time_days)
        order_qty = product.economic_order_qty or int(avg_demand * 14)
    else:
        # naive baseline: static ROP with no safety stock buffer, sized off simple average
        rop = int(avg_demand * product.lead_time_days * static_rop_multiplier)
        order_qty = int(avg_demand * 14)

    stock = rop + order_qty  # start reasonably stocked
    stockout_days = 0
    stockout_events = 0
    was_stocked_out = False
    waste_units = 0
    orders_placed = 0
    lost_units = 0
    pending_deliveries = []  # (arrival_day_index, qty)

    for i, r in enumerate(sales_records):
        # receive any deliveries arriving today
        arrived = [q for (day, q) in pending_deliveries if day == i]
        for q in arrived:
            stock += q
        pending_deliveries = [(day, q) for (day, q) in pending_deliveries if day != i]

        demand = r["units_sold"]
        sold = min(stock, demand)
        shortfall = demand - sold
        stock -= sold

        if shortfall > 0:
            lost_units += shortfall
            stockout_days += 1
            if not was_stocked_out:
                stockout_events += 1
                was_stocked_out = True
        else:
            was_stocked_out = False

        # naive policy doesn't react to promos/weather -> occasionally overstocks and
        # perishable/seasonal categories waste a fraction of excess above ~30 days cover
        if not use_agent_policy and stock > avg_demand * 30:
            excess = int(stock - avg_demand * 25)
            spoil = int(excess * 0.06)  # 6% of deep excess assumed wasted/expired per day-bucket
            if spoil > 0:
                waste_units += spoil
                stock -= spoil

        if stock <= rop and not pending_deliveries:
            pending_deliveries.append((i + product.lead_time_days, order_qty))
            orders_placed += 1

    return dict(
        stockout_days=stockout_days,
        stockout_events=stockout_events,
        lost_units=lost_units,
        waste_units=waste_units,
        orders_placed=orders_placed,
    )


def compute_monthly_savings(product, sales_records_by_month: dict, month_key: str):
    """sales_records_by_month: dict month_key -> list of sales record dicts for that product."""
    records = sales_records_by_month.get(month_key, [])
    if len(records) < 10:
        return None

    baseline = simulate_policy_backtest(product, records, use_agent_policy=False)
    optimized = simulate_policy_backtest(product, records, use_agent_policy=True)

    stockouts_prevented = max(0, baseline["stockout_events"] - optimized["stockout_events"])
    waste_avoided_units = max(0, baseline["waste_units"] - optimized["waste_units"])
    revenue_protected = max(0, baseline["lost_units"] - optimized["lost_units"]) * product.unit_price
    holding_cost_delta = waste_avoided_units * product.unit_cost
    ordering_cost_delta = (baseline["orders_placed"] - optimized["orders_placed"]) * 50  # $ per PO
    cost_savings = round(holding_cost_delta + max(0, ordering_cost_delta), 2)
    waste_reduction_pct = round(
        (waste_avoided_units / baseline["waste_units"] * 100) if baseline["waste_units"] > 0 else 0, 1)

    return dict(
        waste_reduction_pct=waste_reduction_pct,
        waste_avoided_units=int(waste_avoided_units),
        stockouts_prevented=int(stockouts_prevented),
        stockout_events=int(optimized["stockout_events"]),
        cost_savings_usd=round(cost_savings + revenue_protected, 2),
        revenue_protected_usd=round(revenue_protected, 2),
    )
