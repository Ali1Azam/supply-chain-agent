"""
Seeds the database with suppliers, products, and 2 years of synthetic daily
sales history, then runs one agent cycle to populate initial inventory
plans, purchase orders, and risk alerts, and backfills the monthly savings
report.

Usage:
    python -m scripts.seed_db
"""
import sys
import os
from datetime import date, timedelta
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import Base, engine, SessionLocal
from backend import models
from backend.synthetic_data import product_master, supplier_master, generate_sales_history
from backend.agent_orchestrator import run_daily_agent_cycle
from backend.savings_report import compute_monthly_savings


def reset_db():
    print("Dropping and recreating all tables...")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


def seed():
    reset_db()
    db = SessionLocal()

    print("Seeding suppliers...")
    suppliers = []
    for s in supplier_master():
        row = models.Supplier(**s)
        db.add(row)
        suppliers.append(row)
    db.commit()
    for s in suppliers:
        db.refresh(s)

    print("Seeding products...")
    products = []
    for i, (sku, name, category, unit_cost, unit_price, base, wamp, yamp, wsens, trend, noise) in \
            enumerate(product_master()):
        supplier = suppliers[i % len(suppliers)]
        p = models.Product(
            sku=sku, name=name, category=category, unit_cost=unit_cost, unit_price=unit_price,
            current_stock=int(base * 12), lead_time_days=supplier.avg_lead_time_days,
            weather_sensitive=wsens, seasonal=(yamp > 0.2), active=True,
            supplier_id=supplier.id,
        )
        db.add(p)
        products.append(p)
    db.commit()
    for p in products:
        db.refresh(p)

    print("Generating ~2 years of synthetic sales history (this simulates the ERP/POS feed)...")
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=730)
    history = generate_sales_history(start, end)

    sku_to_product = {p.sku: p for p in products}
    sales_by_month = defaultdict(lambda: defaultdict(list))  # sku -> month -> records

    for sku, records in history.items():
        product = sku_to_product[sku]
        batch = []
        for r in records:
            batch.append(models.SalesRecord(
                product_id=product.id, date=r["date"], units_sold=r["units_sold"],
                promotion_flag=r["promotion_flag"], temperature_c=r["temperature_c"],
                is_holiday_event=r["is_holiday_event"],
                competitor_price_index=r["competitor_price_index"],
                stock_on_hand=r["stock_on_hand"],
            ))
            month_key = r["date"].strftime("%Y-%m")
            sales_by_month[sku][month_key].append(r)
        db.bulk_save_objects(batch)
        # set current_stock to the last simulated day's stock so the demo starts "live"
        product.current_stock = records[-1]["stock_on_hand"]
        db.add(product)
    db.commit()
    print(f"Inserted sales history for {len(history)} products, "
          f"{sum(len(v) for v in history.values())} rows.")

    print("Running initial agent cycle (forecast + inventory plan + risk + PO)...")
    summary = run_daily_agent_cycle(db)
    print(f"  -> {summary['purchase_orders_drafted']} purchase order(s) drafted, "
          f"{summary['products_evaluated']} products evaluated.")

    print("Backfilling monthly savings report (backtest simulation)...")
    months = sorted({m for months in sales_by_month.values() for m in months})[:-1]  # drop partial current month
    for month_key in months:
        agg = dict(waste_reduction_pct=[], waste_avoided_units=0, stockouts_prevented=0,
                   stockout_events=0, cost_savings_usd=0.0, revenue_protected_usd=0.0, orders=0)
        any_data = False
        for p in products:
            month_records = sales_by_month[p.sku].get(month_key, [])
            if len(month_records) < 10:
                continue
            result = compute_monthly_savings(p, sales_by_month[p.sku], month_key)
            if not result:
                continue
            any_data = True
            agg["waste_reduction_pct"].append(result["waste_reduction_pct"])
            agg["waste_avoided_units"] += result["waste_avoided_units"]
            agg["stockouts_prevented"] += result["stockouts_prevented"]
            agg["stockout_events"] += result["stockout_events"]
            agg["cost_savings_usd"] += result["cost_savings_usd"]
            agg["revenue_protected_usd"] += result["revenue_protected_usd"]

        if not any_data:
            continue

        rec = models.SavingsRecord(
            month=month_key,
            waste_reduction_pct=round(sum(agg["waste_reduction_pct"]) / max(len(agg["waste_reduction_pct"]), 1), 1),
            waste_avoided_units=agg["waste_avoided_units"],
            stockouts_prevented=agg["stockouts_prevented"],
            stockout_events=agg["stockout_events"],
            cost_savings_usd=round(agg["cost_savings_usd"], 2),
            revenue_protected_usd=round(agg["revenue_protected_usd"], 2),
            total_pos_generated=0,
        )
        db.add(rec)
    db.commit()

    total_po = db.query(models.PurchaseOrder).count()
    total_risk = db.query(models.RiskAlert).count()
    total_savings_months = db.query(models.SavingsRecord).count()
    print(f"Done. {len(products)} products, {len(suppliers)} suppliers, "
          f"{total_po} POs, {total_risk} risk alerts, {total_savings_months} monthly savings records.")
    db.close()


if __name__ == "__main__":
    seed()
