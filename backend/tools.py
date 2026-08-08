"""
LangChain tool definitions (Methodology step 6: "Agent Orchestration").

Each tool wraps one capability the agent can invoke: querying inventory,
running a demand forecast, generating a purchase order, or searching risk
intelligence. These are registered with LangChain's @tool decorator so they
can be bound to a real LLM-driven ReAct agent (see agent_orchestrator.py)
when an API key is configured, while also being directly callable by the
deterministic fallback orchestrator.
"""
from langchain_core.tools import tool
from sqlalchemy.orm import Session

from backend import models
from backend.forecasting import forecast_demand
from backend.inventory_optimizer import build_inventory_plan
from backend.po_generator import choose_best_supplier, generate_draft_po
from backend import risk_agent as risk_mod


def make_tools(db: Session):
    """Factory that closes over a DB session so tools can be bound per-request."""

    @tool
    def query_inventory(sku: str) -> str:
        """Look up the current stock level, reorder point, and safety stock for a product SKU."""
        p = db.query(models.Product).filter(models.Product.sku == sku).first()
        if not p:
            return f"No product found with SKU {sku}"
        return (f"{p.sku} ({p.name}): current_stock={p.current_stock}, "
                f"reorder_point={p.reorder_point}, safety_stock={p.safety_stock}, "
                f"lead_time_days={p.lead_time_days}")

    @tool
    def run_forecast(sku: str, horizon_days: int = 30) -> str:
        """Run a demand forecast for a product SKU over the given horizon in days,
        returning average predicted daily demand and a predicted stockout date if any."""
        p = db.query(models.Product).filter(models.Product.sku == sku).first()
        if not p:
            return f"No product found with SKU {sku}"
        records = [dict(date=s.date, units_sold=s.units_sold, promotion_flag=s.promotion_flag,
                         temperature_c=s.temperature_c, is_holiday_event=s.is_holiday_event,
                         competitor_price_index=s.competitor_price_index)
                   for s in p.sales]
        result = forecast_demand(records, periods=horizon_days, weather_sensitive=p.weather_sensitive)
        plan = build_inventory_plan(p.current_stock, result.avg_daily_demand, result.demand_std,
                                     p.lead_time_days, p.unit_cost, forecast_df=result.future_only)
        return plan.reasoning

    @tool
    def generate_po(sku: str) -> str:
        """Generate a draft purchase order for a product SKU if it is at or below its
        reorder point, selecting the best available supplier."""
        p = db.query(models.Product).filter(models.Product.sku == sku).first()
        if not p:
            return f"No product found with SKU {sku}"
        if p.current_stock > p.reorder_point:
            return f"{sku} is above its reorder point ({p.current_stock} > {p.reorder_point}); no PO needed."
        suppliers = db.query(models.Supplier).all()
        best = choose_best_supplier(suppliers, p.unit_cost)
        records = [dict(date=s.date, units_sold=s.units_sold, promotion_flag=s.promotion_flag,
                         temperature_c=s.temperature_c, is_holiday_event=s.is_holiday_event,
                         competitor_price_index=s.competitor_price_index) for s in p.sales]
        result = forecast_demand(records, periods=30, weather_sensitive=p.weather_sensitive)
        plan = build_inventory_plan(p.current_stock, result.avg_daily_demand, result.demand_std,
                                     p.lead_time_days, p.unit_cost, forecast_df=result.future_only)
        draft = generate_draft_po(p, plan, best)
        return (f"Draft PO created: {draft.quantity} units of {sku} from {draft.supplier_name} "
                f"for ${draft.total_cost:,.2f}, expected delivery {draft.expected_delivery}.")

    @tool
    def search_risk_intelligence(supplier_name: str) -> str:
        """Search current risk intelligence (weather, port congestion, reliability history)
        for a given supplier name and return a summary of any active risks."""
        s = db.query(models.Supplier).filter(models.Supplier.name == supplier_name).first()
        if not s:
            return f"No supplier found named {supplier_name}"
        alerts = []
        for fn in (risk_mod.get_port_congestion_risk, risk_mod.get_supplier_reliability_risk):
            r = fn(s)
            if r:
                alerts.append(r.description)
        w = risk_mod.get_weather_risk(s.location)
        if w:
            alerts.append(w.description)
        return " | ".join(alerts) if alerts else f"No active risks detected for {supplier_name}."

    return [query_inventory, run_forecast, generate_po, search_risk_intelligence]
