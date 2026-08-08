from datetime import date, datetime
from typing import Optional, List

from pydantic import BaseModel


class SupplierOut(BaseModel):
    id: int
    name: str
    location: str
    port_dependent: bool
    avg_lead_time_days: int
    on_time_pct: float
    reliability_score: float
    price_index: float

    class Config:
        from_attributes = True


class ProductOut(BaseModel):
    id: int
    sku: str
    name: str
    category: str
    unit_cost: float
    unit_price: float
    current_stock: int
    lead_time_days: int
    safety_stock: int
    reorder_point: int
    economic_order_qty: int
    supplier_id: Optional[int]

    class Config:
        from_attributes = True


class ForecastPoint(BaseModel):
    date: date
    yhat: float
    yhat_lower: float
    yhat_upper: float
    actual: Optional[float] = None


class ForecastOut(BaseModel):
    sku: str
    engine: str
    avg_daily_demand: float
    demand_std: float
    predicted_stockout_date: Optional[date]
    points: List[ForecastPoint]


class PurchaseOrderOut(BaseModel):
    id: int
    product_id: int
    supplier_id: int
    quantity: int
    unit_cost: float
    total_cost: float
    status: str
    reason: Optional[str]
    created_at: datetime
    expected_delivery: Optional[date]

    class Config:
        from_attributes = True


class RiskAlertOut(BaseModel):
    id: int
    supplier_id: Optional[int]
    product_id: Optional[int]
    risk_type: str
    severity: str
    probability: float
    description: str
    recommendation: str
    created_at: datetime
    resolved: bool

    class Config:
        from_attributes = True


class SavingsRecordOut(BaseModel):
    month: str
    waste_reduction_pct: float
    waste_avoided_units: int
    stockouts_prevented: int
    stockout_events: int
    cost_savings_usd: float
    revenue_protected_usd: float

    class Config:
        from_attributes = True


class AgentRunOut(BaseModel):
    products_evaluated: int
    suppliers_checked: int
    purchase_orders_drafted: int
    ran_at: str
    trace: list
    llm_narrative: Optional[str] = None
