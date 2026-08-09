"""
Streamlit dashboard for the AI-Powered Supply Chain Optimization Agent.

Run with:
    streamlit run dashboard/app.py

Talks to the FastAPI backend over HTTP (set API_BASE_URL in .env if not
running on localhost:8000).
"""
import os
import sys
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# When deployed on Streamlit Community Cloud, config is set via st.secrets
# (Settings -> Secrets) rather than a local .env file. Bridge it into the
# environment before backend.config reads it, so the same code works both
# locally (.env) and when hosted.
try:
    for _key, _val in st.secrets.items():
        os.environ.setdefault(_key, str(_val))
except Exception:
    pass  # no secrets.toml present (e.g. running locally) -- fine, .env covers it

from backend.config import API_BASE_URL  # noqa: E402

st.set_page_config(page_title="Supply Chain Optimization Agent", page_icon="📦", layout="wide")

API = API_BASE_URL


# --------------------------------------------------------------------------
# API helpers
# --------------------------------------------------------------------------
@st.cache_data(ttl=30)
def get_products():
    r = requests.get(f"{API}/products", timeout=15)
    r.raise_for_status()
    return pd.DataFrame(r.json())


@st.cache_data(ttl=30)
def get_suppliers():
    r = requests.get(f"{API}/suppliers", timeout=15)
    r.raise_for_status()
    return pd.DataFrame(r.json())


@st.cache_data(ttl=15)
def get_forecast(sku: str, horizon_days: int = 30):
    r = requests.get(f"{API}/products/{sku}/forecast", params={"horizon_days": horizon_days}, timeout=60)
    r.raise_for_status()
    return r.json()


def get_purchase_orders(status=None):
    params = {"status": status} if status else {}
    r = requests.get(f"{API}/purchase-orders", params=params, timeout=15)
    r.raise_for_status()
    return pd.DataFrame(r.json())


def get_risks():
    r = requests.get(f"{API}/risks", timeout=15)
    r.raise_for_status()
    return pd.DataFrame(r.json())


@st.cache_data(ttl=60)
def get_savings():
    r = requests.get(f"{API}/reports/monthly-savings", timeout=15)
    r.raise_for_status()
    return pd.DataFrame(r.json())


def approve_po(po_id):
    requests.post(f"{API}/purchase-orders/{po_id}/approve", timeout=15)


def reject_po(po_id):
    requests.post(f"{API}/purchase-orders/{po_id}/reject", timeout=15)


def resolve_risk(risk_id):
    requests.post(f"{API}/risks/{risk_id}/resolve", timeout=15)


def run_agent():
    r = requests.post(f"{API}/agent/run", timeout=280)
    r.raise_for_status()
    return r.json()


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("📦 Supply Chain Agent")
st.sidebar.caption("AI-powered inventory monitoring & optimization")

try:
    requests.get(f"{API}/health", timeout=3).raise_for_status()
    st.sidebar.success(f"Connected to API\n{API}")
except Exception:
    st.sidebar.error(
        f"Cannot reach API at {API}. Start it with:\n\nuvicorn backend.main:app --reload"
    )
    st.stop()

page = st.sidebar.radio(
    "View",
    ["Inventory Dashboard", "Demand Forecasting", "Purchase Orders", "Risk Alerts", "Savings Report", "Run Agent"],
)

st.sidebar.divider()
if st.sidebar.button("🔄 Refresh cached data"):
    st.cache_data.clear()
    st.rerun()

products_df = get_products()
suppliers_df = get_suppliers()


# --------------------------------------------------------------------------
# Page: Inventory Dashboard
# --------------------------------------------------------------------------
if page == "Inventory Dashboard":
    st.title("Real-Time Inventory Dashboard")
    st.caption("Stock levels, reorder points, and predicted stockout risk across all SKUs.")

    col1, col2, col3, col4 = st.columns(4)
    at_risk = products_df[products_df["current_stock"] <= products_df["reorder_point"]]
    col1.metric("Total SKUs", len(products_df))
    col2.metric("At/Below Reorder Point", len(at_risk), delta=None,
                delta_color="inverse" if len(at_risk) else "normal")
    col3.metric("Total Inventory Value",
                f"${(products_df['current_stock'] * products_df['unit_cost']).sum():,.0f}")
    col4.metric("Categories", products_df["category"].nunique())

    st.divider()

    categories = ["All"] + sorted(products_df["category"].unique().tolist())
    cat = st.selectbox("Filter by category", categories)
    view_df = products_df if cat == "All" else products_df[products_df["category"] == cat]

    def stock_status(row):
        if row["current_stock"] <= row["safety_stock"]:
            return "🔴 Critical"
        if row["current_stock"] <= row["reorder_point"]:
            return "🟠 Reorder Now"
        return "🟢 Healthy"

    display_df = view_df.copy()
    display_df["status"] = display_df.apply(stock_status, axis=1)
    display_df = display_df[[
        "sku", "name", "category", "current_stock", "safety_stock", "reorder_point",
        "economic_order_qty", "lead_time_days", "status"
    ]].rename(columns={
        "sku": "SKU", "name": "Product", "category": "Category",
        "current_stock": "Current Stock", "safety_stock": "Safety Stock",
        "reorder_point": "Reorder Point", "economic_order_qty": "EOQ",
        "lead_time_days": "Lead Time (d)", "status": "Status",
    })
    st.dataframe(display_df, width='stretch', hide_index=True)

    st.subheader("Stock Level vs. Reorder Point")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=view_df["sku"], y=view_df["current_stock"], name="Current Stock",
                          marker_color="#2E86AB"))
    fig.add_trace(go.Scatter(x=view_df["sku"], y=view_df["reorder_point"], name="Reorder Point",
                              mode="markers+lines", marker=dict(color="#E63946", size=8),
                              line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=view_df["sku"], y=view_df["safety_stock"], name="Safety Stock",
                              mode="markers", marker=dict(color="#F4A261", size=7, symbol="diamond")))
    fig.update_layout(height=450, xaxis_title="SKU", yaxis_title="Units", barmode="overlay",
                       legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width='stretch')


# --------------------------------------------------------------------------
# Page: Demand Forecasting
# --------------------------------------------------------------------------
elif page == "Demand Forecasting":
    st.title("Demand Forecasting")
    st.caption("Prophet-based forecast with external regressors (promotions, weather, holidays, "
               "competitor pricing) and confidence intervals.")

    sku = st.selectbox("Select product", products_df["sku"] + " — " + products_df["name"])
    sku_code = sku.split(" — ")[0]
    horizon = st.slider("Forecast horizon (days)", 7, 90, 30)

    with st.spinner("Running forecast..."):
        fc = get_forecast(sku_code, horizon)

    points = pd.DataFrame(fc["points"])
    points["date"] = pd.to_datetime(points["date"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Daily Demand (forecast)", f"{fc['avg_daily_demand']:.1f} units")
    col2.metric("Demand Volatility (σ)", f"{fc['demand_std']:.1f}")
    col3.metric("Predicted Stockout Date", fc["predicted_stockout_date"] or "None in horizon")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=points["date"], y=points["yhat_upper"], line=dict(width=0),
                              showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=points["date"], y=points["yhat_lower"], fill="tonexty",
                              fillcolor="rgba(46,134,171,0.2)", line=dict(width=0),
                              name="Confidence Interval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=points["date"], y=points["yhat"], name="Forecast",
                              line=dict(color="#2E86AB", width=2)))
    actuals = points.dropna(subset=["actual"])
    fig.add_trace(go.Scatter(x=actuals["date"], y=actuals["actual"], name="Actual Sales",
                              mode="markers", marker=dict(color="#264653", size=4, opacity=0.6)))
    fig.update_layout(height=500, xaxis_title="Date", yaxis_title="Units / day",
                       title=f"{sku_code} Demand Forecast ({fc['engine']} engine)",
                       legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width='stretch')

    st.info(
        "💡 The shaded band is the 90% confidence interval. Wider bands further out reflect "
        "growing forecast uncertainty."
    )


# --------------------------------------------------------------------------
# Page: Purchase Orders
# --------------------------------------------------------------------------
elif page == "Purchase Orders":
    st.title("Automated Purchase Orders")
    st.caption("Draft POs generated by the agent when stock falls to/below the reorder point. "
               "All POs require human approval before being sent.")

    status_filter = st.selectbox("Filter by status", ["All", "pending_approval", "approved", "rejected"])
    po_df = get_purchase_orders(None if status_filter == "All" else status_filter)

    if po_df.empty:
        st.info("No purchase orders yet. Try running the agent from the 'Run Agent' page.")
    else:
        po_df = po_df.merge(products_df[["id", "sku", "name"]], left_on="product_id", right_on="id",
                             suffixes=("", "_product"))
        po_df = po_df.merge(suppliers_df[["id", "name"]], left_on="supplier_id", right_on="id",
                             suffixes=("", "_supplier"))
        po_df = po_df.rename(columns={"name": "product_name", "name_supplier": "supplier_name"})

        for _, po in po_df.iterrows():
            with st.expander(
                f"PO #{po['id']} · {po['sku']} · {int(po['quantity'])} units from "
                f"{po['supplier_name']} · ${po['total_cost']:,.2f} · [{po['status']}]"
            ):
                st.write(f"**Product:** {po['product_name']} ({po['sku']})")
                st.write(f"**Supplier:** {po['supplier_name']}")
                st.write(f"**Quantity:** {int(po['quantity'])} units @ ${po['unit_cost']:.2f}/unit")
                st.write(f"**Total Cost:** ${po['total_cost']:,.2f}")
                st.write(f"**Expected Delivery:** {po['expected_delivery']}")
                st.write(f"**Agent Reasoning:** {po['reason']}")
                if po["status"] == "pending_approval":
                    c1, c2 = st.columns(2)
                    if c1.button("✅ Approve", key=f"approve_{po['id']}"):
                        approve_po(po["id"])
                        st.rerun()
                    if c2.button("❌ Reject", key=f"reject_{po['id']}"):
                        reject_po(po["id"])
                        st.rerun()


# --------------------------------------------------------------------------
# Page: Risk Alerts
# --------------------------------------------------------------------------
elif page == "Risk Alerts":
    st.title("Supply Chain Risk Alerts")
    st.caption("Weather, port congestion, supplier reliability, and demand-spike risks "
               "identified by the risk agent.")

    risks_df = get_risks()
    if risks_df.empty:
        st.info("No risk alerts yet. Try running the agent from the 'Run Agent' page.")
    else:
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        risks_df["_order"] = risks_df["severity"].map(severity_order)
        risks_df = risks_df.sort_values(["resolved", "_order"])

        severity_colors = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
        for _, risk in risks_df.iterrows():
            icon = severity_colors.get(risk["severity"], "⚪")
            resolved_tag = " ✅ resolved" if risk["resolved"] else ""
            with st.expander(f"{icon} [{risk['severity'].upper()}] {risk['risk_type'].replace('_', ' ').title()} "
                              f"— {int(risk['probability']*100)}% probability{resolved_tag}"):
                st.write(risk["description"])
                st.write(f"**Recommendation:** {risk['recommendation']}")
                st.caption(f"Raised {risk['created_at']}")
                if not risk["resolved"]:
                    if st.button("Mark resolved", key=f"resolve_{risk['id']}"):
                        resolve_risk(risk["id"])
                        st.rerun()


# --------------------------------------------------------------------------
# Page: Savings Report
# --------------------------------------------------------------------------
elif page == "Savings Report":
    st.title("Monthly Savings Report")
    st.caption("Backtested comparison of the AI-optimized policy vs. a naive static-reorder "
               "baseline (representative of typical SMB inventory practice).")

    savings_df = get_savings()
    if savings_df.empty:
        st.info("No savings data yet. Run the seed script to backfill historical months.")
    else:
        total_savings = savings_df["cost_savings_usd"].sum()
        total_stockouts_prevented = savings_df["stockouts_prevented"].sum()
        total_waste_avoided = savings_df["waste_avoided_units"].sum()
        avg_waste_reduction = savings_df["waste_reduction_pct"].mean()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Savings (all months)", f"${total_savings:,.0f}")
        col2.metric("Stockouts Prevented", int(total_stockouts_prevented))
        col3.metric("Waste Avoided (units)", int(total_waste_avoided))
        col4.metric("Avg Waste Reduction", f"{avg_waste_reduction:.0f}%")

        st.divider()

        fig = go.Figure()
        fig.add_trace(go.Bar(x=savings_df["month"], y=savings_df["cost_savings_usd"],
                              name="Cost Savings ($)", marker_color="#2A9D8F"))
        fig.add_trace(go.Bar(x=savings_df["month"], y=savings_df["revenue_protected_usd"],
                              name="Revenue Protected ($)", marker_color="#264653"))
        fig.update_layout(height=420, barmode="group", xaxis_title="Month", yaxis_title="USD",
                           title="Cost Savings & Revenue Protected by Month",
                           legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width='stretch')

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=savings_df["month"], y=savings_df["stockouts_prevented"],
                                   name="Stockouts Prevented", mode="lines+markers",
                                   line=dict(color="#E63946")))
        fig2.add_trace(go.Scatter(x=savings_df["month"], y=savings_df["waste_avoided_units"],
                                   name="Waste Avoided (units)", mode="lines+markers",
                                   line=dict(color="#F4A261"), yaxis="y2"))
        fig2.update_layout(
            height=380, xaxis_title="Month",
            yaxis=dict(title="Stockouts Prevented"),
            yaxis2=dict(title="Waste Avoided (units)", overlaying="y", side="right"),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig2, width='stretch')

        st.dataframe(savings_df, width='stretch', hide_index=True)


# --------------------------------------------------------------------------
# Page: Run Agent
# --------------------------------------------------------------------------
elif page == "Run Agent":
    st.title("Run Agent Cycle")
    st.caption("Triggers one full ReAct-style pass: forecast every product, refresh inventory "
               "plans, draft POs where needed, and check supplier risk.")

    if st.button("▶️ Run Agent Now", type="primary"):
        with st.spinner("Agent is reasoning through inventory, forecasts, and risk..."):
            result = run_agent()
        st.success(
            f"Cycle complete at {result['ran_at']}: {result['products_evaluated']} products evaluated, "
            f"{result['purchase_orders_drafted']} PO(s) drafted, {result['suppliers_checked']} suppliers checked."
        )
        if result.get("llm_narrative"):
            st.subheader("Executive Summary (LLM-generated)")
            st.write(result["llm_narrative"])

        st.subheader("Agent Reasoning Trace")
        for step in result["trace"]:
            icon = {"Thought": "🧠", "Action": "⚙️", "Observation": "👁️"}.get(step["step"], "•")
            st.markdown(f"{icon} **{step['step']}:** {step['detail']}")

        st.cache_data.clear()
