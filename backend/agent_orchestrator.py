"""
Agent Orchestration (Methodology step 6).

Runs the end-to-end agentic cycle: for every active product, forecast demand,
recompute its inventory plan (ROP/EOQ/safety stock), generate a draft PO if
needed, and check supplier risk. Every decision is logged as a
Thought -> Action -> Observation step so the reasoning is auditable, in the
spirit of a LangChain ReAct agent.

Tool-calling is implemented via backend/tools.py (LangChain @tool functions).
When an Anthropic or OpenAI API key is configured, a real LLM is used to
narrate/summarize the day's run in natural language; the underlying
decisions themselves are always made deterministically by the tools so the
system is auditable and doesn't depend on an LLM being available.
"""
import logging
from datetime import date, datetime

from sqlalchemy.orm import Session

from backend import models
from backend.forecasting import forecast_demand
from backend.inventory_optimizer import build_inventory_plan
from backend.po_generator import choose_best_supplier, generate_draft_po
from backend import risk_agent as risk_mod
from backend.config import ANTHROPIC_API_KEY, OPENAI_API_KEY, LLM_PROVIDER

logger = logging.getLogger(__name__)


def _sales_records_for(product: models.Product):
    return [dict(date=s.date, units_sold=s.units_sold, promotion_flag=s.promotion_flag,
                 temperature_c=s.temperature_c, is_holiday_event=s.is_holiday_event,
                 competitor_price_index=s.competitor_price_index) for s in product.sales]


def evaluate_product(db: Session, product: models.Product, trace: list, horizon_days: int = 30):
    trace.append({"step": "Thought", "detail": f"Evaluating {product.sku} ({product.name}). "
                  "Need current demand forecast to size safety stock and reorder point."})

    records = _sales_records_for(product)
    if len(records) < 30:
        trace.append({"step": "Observation", "detail": f"Not enough history for {product.sku}, skipping."})
        return None, None

    trace.append({"step": "Action", "detail": f"run_forecast(sku='{product.sku}', horizon_days={horizon_days})"})
    result = forecast_demand(records, periods=horizon_days, weather_sensitive=product.weather_sensitive)
    trace.append({"step": "Observation",
                  "detail": f"Forecast engine={result.engine}, avg_daily_demand={result.avg_daily_demand:.1f}, "
                            f"std={result.demand_std:.1f}"})

    plan = build_inventory_plan(product.current_stock, result.avg_daily_demand, result.demand_std,
                                 product.lead_time_days, product.unit_cost, forecast_df=result.future_only)
    trace.append({"step": "Thought", "detail": plan.reasoning})

    # persist updated plan fields
    product.safety_stock = plan.safety_stock
    product.reorder_point = plan.reorder_point
    product.economic_order_qty = plan.economic_order_qty
    db.add(product)

    # demand spike risk check
    spike = risk_mod.get_demand_spike_risk(product.sku, product.name, result.future_only, result.avg_daily_demand)
    if spike:
        _persist_alert(db, spike, product_id=product.id)
        trace.append({"step": "Observation", "detail": f"Risk flagged: {spike.description}"})

    return result, plan


def maybe_generate_po(db: Session, product: models.Product, plan, trace: list):
    if plan is None:
        return None
    if product.current_stock > plan.reorder_point:
        trace.append({"step": "Observation",
                      "detail": f"{product.sku} stock ({product.current_stock}) is above reorder point "
                                f"({plan.reorder_point}); no PO needed."})
        return None

    trace.append({"step": "Thought", "detail": f"{product.sku} is at/below reorder point -> draft a PO."})
    trace.append({"step": "Action", "detail": f"generate_po(sku='{product.sku}')"})

    suppliers = db.query(models.Supplier).all()
    best = choose_best_supplier(suppliers, product.unit_cost)
    if not best:
        trace.append({"step": "Observation", "detail": "No suppliers available."})
        return None

    draft = generate_draft_po(product, plan, best)

    po = models.PurchaseOrder(
        product_id=product.id,
        supplier_id=best.id,
        quantity=draft.quantity,
        unit_cost=draft.unit_cost,
        total_cost=draft.total_cost,
        status="pending_approval",
        reason=draft.reason,
        expected_delivery=draft.expected_delivery,
    )
    db.add(po)
    trace.append({"step": "Observation",
                  "detail": f"Draft PO for {draft.quantity} units of {product.sku} from "
                            f"{draft.supplier_name} (${draft.total_cost:,.2f}), pending human approval."})
    return po


def _persist_alert(db: Session, alert: risk_mod.RiskAlert, product_id=None, supplier_id=None):
    row = models.RiskAlert(
        supplier_id=supplier_id,
        product_id=product_id,
        risk_type=alert.risk_type,
        severity=alert.severity,
        probability=alert.probability,
        description=alert.description,
        recommendation=alert.recommendation,
    )
    db.add(row)
    return row


def evaluate_supplier_risk(db: Session, supplier: models.Supplier, trace: list):
    trace.append({"step": "Thought", "detail": f"Checking risk intelligence for supplier {supplier.name}."})
    trace.append({"step": "Action", "detail": f"search_risk_intelligence(supplier_name='{supplier.name}')"})

    found = []
    for fn in (risk_mod.get_port_congestion_risk, risk_mod.get_supplier_reliability_risk):
        r = fn(supplier)
        if r:
            _persist_alert(db, r, supplier_id=supplier.id)
            found.append(r.description)

    w = risk_mod.get_weather_risk(supplier.location)
    if w:
        _persist_alert(db, w, supplier_id=supplier.id)
        found.append(w.description)

    trace.append({"step": "Observation",
                  "detail": "; ".join(found) if found else f"No active risks for {supplier.name}."})


def run_daily_agent_cycle(db: Session) -> dict:
    """Runs the full agent cycle across all active products and suppliers.
    Returns a summary dict including the full reasoning trace."""
    trace = []
    trace.append({"step": "Thought",
                  "detail": "Starting daily supply chain review: forecast demand, refresh inventory "
                            "plans, flag risks, and draft POs where needed."})

    products = db.query(models.Product).filter(models.Product.active == True).all()  # noqa: E712
    pos_created = []
    for p in products:
        result, plan = evaluate_product(db, p, trace)
        po = maybe_generate_po(db, p, plan, trace)
        if po:
            pos_created.append(po)

    suppliers = db.query(models.Supplier).all()
    for s in suppliers:
        evaluate_supplier_risk(db, s, trace)

    db.commit()

    trace.append({"step": "Thought",
                  "detail": f"Cycle complete: {len(pos_created)} PO(s) drafted, "
                            f"{len(products)} products evaluated, {len(suppliers)} suppliers checked."})

    summary = {
        "products_evaluated": len(products),
        "suppliers_checked": len(suppliers),
        "purchase_orders_drafted": len(pos_created),
        "trace": trace,
        "ran_at": datetime.utcnow().isoformat(),
    }

    narrative = _try_llm_narrative(summary)
    if narrative:
        summary["llm_narrative"] = narrative

    return summary


def _try_llm_narrative(summary: dict) -> "str | None":
    """Optional: if an LLM API key is configured, ask it to write a short natural-language
    executive summary of the day's agent run. Purely additive -- never blocks the core
    deterministic pipeline above, and fails silently if no key/model is available."""
    provider = LLM_PROVIDER
    try:
        if provider in ("auto", "anthropic") and ANTHROPIC_API_KEY:
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model="claude-sonnet-4-6", api_key=ANTHROPIC_API_KEY, max_tokens=300)
        elif provider in ("auto", "openai") and OPENAI_API_KEY:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, max_tokens=300)
        else:
            return None

        bullet_points = "\n".join(f"- {t['step']}: {t['detail']}" for t in summary["trace"][-12:])
        prompt = (
            "You are a supply chain operations analyst. Write a concise 3-4 sentence "
            "executive summary of today's automated inventory review for a small business "
            "owner, based on this agent run log:\n\n" + bullet_points
        )
        resp = llm.invoke(prompt)
        return getattr(resp, "content", str(resp))
    except Exception as e:  # pragma: no cover
        logger.info(f"LLM narrative unavailable, skipping ({e})")
        return None
