"""
SQLAlchemy ORM models.

Schema overview
----------------
Supplier          -- vendors we can order from, with reliability stats
Product           -- SKUs we stock, linked to a primary supplier
SalesRecord       -- daily historical POS/demand data + external regressors
InventorySnapshot -- point-in-time stock levels (simulates ERP feed)
PurchaseOrder     -- auto-generated and human-approved POs
RiskAlert         -- supply-chain risk events raised by the risk agent
SavingsRecord     -- monthly rollup used for the savings report
"""
from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, Date
)
from sqlalchemy.orm import relationship

from backend.database import Base


class Supplier(Base):
    __tablename__ = "suppliers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    location = Column(String, nullable=False)
    port_dependent = Column(Boolean, default=False)  # ships via ocean freight / major port
    avg_lead_time_days = Column(Integer, default=7)
    on_time_pct = Column(Float, default=0.9)          # historical on-time delivery rate
    reliability_score = Column(Float, default=0.9)     # composite 0-1 score
    price_index = Column(Float, default=1.0)           # relative price competitiveness (1.0 = market avg)

    products = relationship("Product", back_populates="supplier")
    purchase_orders = relationship("PurchaseOrder", back_populates="supplier")


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    sku = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    category = Column(String, index=True)
    unit_cost = Column(Float, nullable=False)     # cost we pay supplier
    unit_price = Column(Float, nullable=False)    # price we sell at
    current_stock = Column(Integer, default=0)
    lead_time_days = Column(Integer, default=7)
    safety_stock = Column(Integer, default=0)
    reorder_point = Column(Integer, default=0)
    economic_order_qty = Column(Integer, default=0)
    weather_sensitive = Column(Boolean, default=False)  # demand correlates with weather
    seasonal = Column(Boolean, default=True)
    active = Column(Boolean, default=True)

    supplier_id = Column(Integer, ForeignKey("suppliers.id"))
    supplier = relationship("Supplier", back_populates="products")

    sales = relationship("SalesRecord", back_populates="product")
    purchase_orders = relationship("PurchaseOrder", back_populates="product")


class SalesRecord(Base):
    __tablename__ = "sales_records"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), index=True)
    date = Column(Date, index=True, nullable=False)
    units_sold = Column(Integer, nullable=False)
    promotion_flag = Column(Boolean, default=False)
    temperature_c = Column(Float)
    is_holiday_event = Column(Boolean, default=False)
    competitor_price_index = Column(Float, default=1.0)
    stock_on_hand = Column(Integer)  # end-of-day stock (for backtesting stockouts)

    product = relationship("Product", back_populates="sales")


class PurchaseOrder(Base):
    __tablename__ = "purchase_orders"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"))
    supplier_id = Column(Integer, ForeignKey("suppliers.id"))
    quantity = Column(Integer, nullable=False)
    unit_cost = Column(Float, nullable=False)
    total_cost = Column(Float, nullable=False)
    status = Column(String, default="pending_approval")  # pending_approval, approved, sent, received, rejected
    reason = Column(Text)  # chain-of-thought explanation from the agent
    created_at = Column(DateTime, default=datetime.utcnow)
    expected_delivery = Column(Date)
    approved_at = Column(DateTime, nullable=True)
    received_at = Column(DateTime, nullable=True)

    product = relationship("Product", back_populates="purchase_orders")
    supplier = relationship("Supplier", back_populates="purchase_orders")


class RiskAlert(Base):
    __tablename__ = "risk_alerts"

    id = Column(Integer, primary_key=True, index=True)
    supplier_id = Column(Integer, ForeignKey("suppliers.id"), nullable=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=True)
    risk_type = Column(String)   # weather, port_congestion, supplier_reliability, demand_spike, price_volatility
    severity = Column(String)    # low, medium, high, critical
    probability = Column(Float)  # 0-1
    description = Column(Text)
    recommendation = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved = Column(Boolean, default=False)


class SavingsRecord(Base):
    __tablename__ = "savings_records"

    id = Column(Integer, primary_key=True, index=True)
    month = Column(String, index=True)  # "YYYY-MM"
    waste_reduction_pct = Column(Float)
    waste_avoided_units = Column(Integer)
    stockouts_prevented = Column(Integer)
    stockout_events = Column(Integer)
    cost_savings_usd = Column(Float)
    revenue_protected_usd = Column(Float)
    total_pos_generated = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
