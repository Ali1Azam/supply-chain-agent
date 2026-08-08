# AI-Powered Supply Chain Optimization Agent

An agentic system that monitors inventory, forecasts demand, auto-generates
purchase orders, flags supply-chain risk, and reports on savings — built for
**Project 8** (Poorit Technologies).

Small/medium businesses lose 5–15% of revenue to inventory mismanagement.
This reference implementation demonstrates the full pipeline end-to-end
using realistic simulated data, so it runs with **zero external API keys or
paid services**, while being structured so every simulated piece
(ERP feed, weather API, LLM) can be swapped for a real integration.

## What's included

| Requirement (from brief)            | Implementation |
|--------------------------------------|-----------------|
| Real-time inventory dashboard        | Streamlit dashboard, stock/ROP/safety-stock table + chart |
| Demand forecasting with confidence intervals | Prophet (external regressors: promotions, weather, holidays, competitor price) with a statsmodels Holt-Winters fallback |
| Automated PO generation              | `backend/po_generator.py` — triggers at reorder point, sizes at EOQ, scores suppliers, drafted POs require human approval |
| Risk alerts ("Supplier X has 30% chance of delay...") | `backend/risk_agent.py` — weather, port congestion, supplier reliability, demand-spike detection |
| Monthly savings report               | `backend/savings_report.py` — backtests an AI-optimized policy vs. a naive static-reorder baseline |
| LangChain agent orchestration        | `backend/tools.py` + `backend/agent_orchestrator.py` — ReAct-style Thought/Action/Observation trace over 4 tools; optional real LLM narration if you add an API key |

## Architecture

```
                ┌─────────────────────┐
                │   synthetic_data.py │  (stand-in for ERP/POS/weather feeds)
                └──────────┬──────────┘
                           │ seed
                           ▼
┌────────────────────────────────────────────────┐
│                 PostgreSQL / SQLite              │
│  suppliers · products · sales_records ·          │
│  purchase_orders · risk_alerts · savings_records  │
└───────────────────────┬──────────────────────────┘
                         │
        ┌────────────────┼─────────────────┐
        ▼                ▼                 ▼
 forecasting.py   inventory_optimizer.py  risk_agent.py
 (Prophet / HW)   (ROP / EOQ / safety     (weather / port /
                   stock / stockout date)  reliability / spikes)
        │                │                 │
        └────────┬───────┴────────┬────────┘
                  ▼                ▼
           po_generator.py   agent_orchestrator.py
                              (LangChain ReAct tools,
                               Thought/Action/Observation)
                         │
                         ▼
                  FastAPI (backend/main.py)
                         │
                         ▼
              Streamlit dashboard (dashboard/app.py)
              Plotly charts, PO approval UI, risk feed
```

## Quick start

```bash
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt

cp .env.example .env      # optional — defaults work with no keys at all

# Seed the database with 2 years of synthetic sales history, suppliers,
# and an initial agent run (creates data/supply_chain.db by default):
python -m scripts.seed_db

# Terminal 1: start the API
uvicorn backend.main:app --reload --port 8000

# Terminal 2: start the dashboard
streamlit run dashboard/app.py
```

Or just run `./run.sh`, which does all three steps for you.

Then open the dashboard at **http://localhost:8501** and the interactive
API docs at **http://localhost:8000/docs**.

## Using real data & APIs instead of the simulation

- **ERP / POS data**: replace `synthetic_data.py` with a connector that
  writes rows into the `sales_records` table (same schema) — nothing else
  needs to change.
- **Weather**: set `OPENWEATHERMAP_API_KEY` in `.env`; `risk_agent.py`
  will automatically call the real OpenWeatherMap API instead of the
  simulated feed.
- **PostgreSQL**: set `DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/db`
  in `.env` and `pip install psycopg2-binary` — SQLAlchemy models are
  database-agnostic.
- **LLM narration**: set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` to have
  the agent write a natural-language executive summary of each run
  (`backend/agent_orchestrator.py::_try_llm_narrative`). The core
  decisions (forecasts, ROP/EOQ math, PO drafting, risk scoring) are
  always deterministic and auditable — the LLM is additive narration only,
  by design, so the system doesn't depend on an LLM being available or
  correct for financial decisions.
- **News / commodity price feeds**: `risk_agent.py` is structured so a
  real news API or RAG-over-incident-history layer can be dropped in
  alongside `get_port_congestion_risk` / `get_weather_risk`.

## Key design decisions

- **Forecasting engine is pluggable**: `forecast_demand()` in
  `backend/forecasting.py` tries Prophet first (recommended for the
  seasonal categories in the brief) and transparently falls back to
  Holt-Winters exponential smoothing with bootstrapped confidence
  intervals if Prophet/cmdstan isn't available in your environment. Swap
  in an LSTM (PyTorch/Keras) behind the same interface for non-linear
  demand patterns without touching any downstream code.
- **POs always require human approval** (`status="pending_approval"`):
  the agent drafts, it never auto-sends, matching the brief's "Route for
  human approval" step.
- **Savings report is a backtest**, since a fresh deployment has no
  "before" period — it simulates a naive static-reorder policy (typical
  of SMBs without forecasting) against the AI-optimized policy over the
  same historical demand, to produce a defensible waste/stockout/cost
  comparison.
- **Agent reasoning is fully logged**: every `/agent/run` call returns a
  Thought → Action → Observation trace (visible in the "Run Agent"
  dashboard page) so decisions are auditable, not a black box.

## Project structure

```
backend/
  config.py              # env-driven settings
  database.py             # SQLAlchemy engine/session
  models.py                # ORM schema
  schemas.py                # Pydantic API response models
  synthetic_data.py          # simulated ERP/POS/weather feed
  forecasting.py              # Prophet + Holt-Winters demand forecasting
  inventory_optimizer.py       # ROP / EOQ / safety stock / stockout date
  risk_agent.py                 # weather / port congestion / reliability / demand spikes
  po_generator.py                # supplier scoring + draft PO creation
  tools.py                        # LangChain @tool wrappers
  agent_orchestrator.py            # ReAct-style daily agent cycle
  savings_report.py                 # backtest-based monthly savings calc
  main.py                            # FastAPI app
dashboard/
  app.py                              # Streamlit dashboard (6 pages)
scripts/
  seed_db.py                           # builds schema + seeds all data
requirements.txt
.env.example
run.sh
```

## Reference implementations this design draws on

- **zefang-liu/InvAgent** — multi-agent, chain-of-thought inventory
  reasoning; reflected here in the reasoning strings attached to every
  inventory plan and PO.
- **microsoft/OptiGuide** — plain-language supply chain Q&A over an
  optimization backend; reflected in the LangChain tool interface, which
  can be extended with a chat endpoint on top of the same tools.
- arXiv:2407.11384 (InvAgent), arXiv:2307.03875 (Microsoft Research LLMs
  for Supply Chain Optimization) — informed the ReAct tool design and the
  "reason before acting" trace format.
