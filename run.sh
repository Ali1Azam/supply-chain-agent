#!/usr/bin/env bash
# Convenience launcher: seeds the database (if empty) then starts the
# FastAPI backend and Streamlit dashboard together.
set -e

cd "$(dirname "$0")"

if [ ! -f "data/supply_chain.db" ]; then
  echo "No database found -- seeding with synthetic data (this takes ~30-60s)..."
  python -m scripts.seed_db
fi

echo "Starting FastAPI backend on http://localhost:8000 ..."
uvicorn backend.main:app --port 8000 &
BACKEND_PID=$!

sleep 3

echo "Starting Streamlit dashboard on http://localhost:8501 ..."
streamlit run dashboard/app.py --server.port 8501

kill $BACKEND_PID
