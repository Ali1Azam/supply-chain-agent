"""
Central configuration for the Supply Chain Optimization Agent.

All settings are read from environment variables (see .env.example) so the
same codebase can run against SQLite for local demos or PostgreSQL in
production without any code changes.
"""
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# --- Database -----------------------------------------------------------
# Defaults to a local SQLite file so the project runs out of the box.
# For production, set DATABASE_URL="postgresql+psycopg2://user:pass@host:5432/dbname"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(DATA_DIR, 'supply_chain.db')}")

# --- External APIs -------------------------------------------------------
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# --- LLM / Agent config ---------------------------------------------------
# The agent orchestrator will use a real LLM (Anthropic or OpenAI) if a key
# is present, and transparently fall back to a deterministic rule-based
# ReAct-style agent otherwise so the whole system works with zero API keys.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto")  # auto | anthropic | openai | none

# --- Business rules --------------------------------------------------------
DEFAULT_SERVICE_LEVEL_Z = float(os.getenv("SERVICE_LEVEL_Z", "1.65"))  # ~95% service level
DEFAULT_ORDERING_COST = float(os.getenv("ORDERING_COST", "50"))         # $ per PO placed
DEFAULT_HOLDING_COST_PCT = float(os.getenv("HOLDING_COST_PCT", "0.20"))  # 20% of unit cost / yr
FORECAST_HORIZON_DAYS = int(os.getenv("FORECAST_HORIZON_DAYS", "30"))

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_BASE_URL = os.getenv("API_BASE_URL", f"http://localhost:{API_PORT}")
