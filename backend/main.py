"""
FastAPI backend for the AI-Powered Supply Chain Optimization Agent.

Run with:
    uvicorn backend.main:app --reload --port 8000

Endpoints map directly to the "Expected Output" in the brief:
  GET  /products                    -> real-time inventory dashboard data
  GET  /products/{sku}/forecast     -> demand forecasting graph data with confidence intervals
  GET  /purchase-orders             -> list of (auto-generated) POs
  POST /purchase-orders/{id}/approve / reject
  GET  /risks                       -> risk alerts
  GET  /reports/monthly-savings     -> monthly savings report
  POST /agent/run                   -> trigger one full agent cycle (forecast + PO + risk)
"""
from datetime import date, datetime
from typing import List, Optional

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from backend.database import Base, engine, get_db
from backend import models, schemas
from backend.forecasting import forecast_demand
from backend.inventory_optimizer import build_inventory_plan
from backend.agent_orchestrator import run_daily_agent_cycle
from backend.savings_report import compute_monthly_savings

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Supply Chain Optimization Agent API",
    description="Agentic inventory monitoring, demand forecasting, PO generation, and risk intelligence.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


# --------------------------------------------------------------------------
# Inventory dashboard
# --------------------------------------------------------------------------
@app.get("/products", response_model=List[schemas.ProductOut])
def list_products(category: Optional[str] = None, db: Session = Depends(get_db)):
    q = db.query(models.Product).filter(models.Product.active == True)  # noqa: E712
    if category:
        q = q.filter(models.Product.category == category)
    return q.all()


@app.get("/products/{sku}", response_model=schemas.ProductOut)
def get_product(sku: str, db: Session = Depends(get_db)):
    p = db.query(models.Product).filter(models.Product.sku == sku).first()
    if not p:
        raise HTTPException(404, "Product not found")
    return p


@app.get("/suppliers", response_model=List[schemas.SupplierOut])
def list_suppliers(db: Session = Depends(get_db)):
    return db.query(models.Supplier).all()


# --------------------------------------------------------------------------
# Forecasting
# --------------------------------------------------------------------------
@app.get("/products/{sku}/forecast", response_model=schemas.ForecastOut)
def get_forecast(sku: str, horizon_days: int = 30, db: Session = Depends(get_db)):
    p = db.query(models.Product).filter(models.Product.sku == sku).first()
    if not p:
        raise HTTPException(404, "Product not found")

    records = [dict(date=s.date, units_sold=s.units_sold, promotion_flag=s.promotion_flag,
                     temperature_c=s.temperature_c, is_holiday_event=s.is_holiday_event,
                     competitor_price_index=s.competitor_price_index) for s in p.sales]
    if len(records) < 30:
        raise HTTPException(400, "Not enough sales history to forecast")

    result = forecast_demand(records, periods=horizon_days, weather_sensitive=p.weather_sensitive)
    plan = build_inventory_plan(p.current_stock, result.avg_daily_demand, result.demand_std,
                                 p.lead_time_days, p.unit_cost, forecast_df=result.future_only)

    actuals = {r["date"]: r["units_sold"] for r in records}
    points = []
    for _, row in result.forecast.iterrows():
        d = row["ds"].date()
        points.append(schemas.ForecastPoint(
            date=d, yhat=round(float(row["yhat"]), 2),
            yhat_lower=round(float(row["yhat_lower"]), 2),
            yhat_upper=round(float(row["yhat_upper"]), 2),
            actual=actuals.get(d),
        ))

    return schemas.ForecastOut(
        sku=sku, engine=result.engine, avg_daily_demand=round(result.avg_daily_demand, 2),
        demand_std=round(result.demand_std, 2),
        predicted_stockout_date=plan.predicted_stockout_date, points=points,
    )


# --------------------------------------------------------------------------
# Purchase Orders
# --------------------------------------------------------------------------
@app.get("/purchase-orders", response_model=List[schemas.PurchaseOrderOut])
def list_purchase_orders(status: Optional[str] = None, db: Session = Depends(get_db)):
    q = db.query(models.PurchaseOrder).order_by(models.PurchaseOrder.created_at.desc())
    if status:
        q = q.filter(models.PurchaseOrder.status == status)
    return q.all()


@app.post("/purchase-orders/{po_id}/approve", response_model=schemas.PurchaseOrderOut)
def approve_po(po_id: int, db: Session = Depends(get_db)):
    po = db.query(models.PurchaseOrder).get(po_id)
    if not po:
        raise HTTPException(404, "PO not found")
    po.status = "approved"
    po.approved_at = datetime.utcnow()
    db.commit()
    db.refresh(po)
    return po


@app.post("/purchase-orders/{po_id}/reject", response_model=schemas.PurchaseOrderOut)
def reject_po(po_id: int, db: Session = Depends(get_db)):
    po = db.query(models.PurchaseOrder).get(po_id)
    if not po:
        raise HTTPException(404, "PO not found")
    po.status = "rejected"
    db.commit()
    db.refresh(po)
    return po


# --------------------------------------------------------------------------
# Risk alerts
# --------------------------------------------------------------------------
@app.get("/risks", response_model=List[schemas.RiskAlertOut])
def list_risks(resolved: Optional[bool] = None, db: Session = Depends(get_db)):
    q = db.query(models.RiskAlert).order_by(models.RiskAlert.created_at.desc())
    if resolved is not None:
        q = q.filter(models.RiskAlert.resolved == resolved)
    return q.limit(200).all()


@app.post("/risks/{risk_id}/resolve", response_model=schemas.RiskAlertOut)
def resolve_risk(risk_id: int, db: Session = Depends(get_db)):
    r = db.query(models.RiskAlert).get(risk_id)
    if not r:
        raise HTTPException(404, "Risk alert not found")
    r.resolved = True
    db.commit()
    db.refresh(r)
    return r


# --------------------------------------------------------------------------
# Savings report
# --------------------------------------------------------------------------
@app.get("/reports/monthly-savings", response_model=List[schemas.SavingsRecordOut])
def monthly_savings(db: Session = Depends(get_db)):
    records = db.query(models.SavingsRecord).order_by(models.SavingsRecord.month).all()
    return records


# --------------------------------------------------------------------------
# Agent orchestration
# --------------------------------------------------------------------------
@app.post("/agent/run", response_model=schemas.AgentRunOut)
def run_agent(db: Session = Depends(get_db)):
    """Triggers one full ReAct-style agent cycle: forecast every active product,
    refresh inventory plans, draft POs where needed, and check supplier risk."""
    summary = run_daily_agent_cycle(db)
    return summary
