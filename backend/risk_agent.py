"""
Risk Identification (Methodology step 5).

Monitors weather APIs, simulated news/port-congestion feeds, and supplier
historical reliability, then classifies risks by severity and generates a
plain-English alert + mitigation recommendation -- e.g. the exact style of
alert requested in the brief: "Supplier X has 30% chance of delay due to
port congestion."

If OPENWEATHERMAP_API_KEY is set, real current weather is pulled for the
supplier's location. Otherwise a deterministic simulated feed is used so
the module works with zero configuration.
"""
import random
from dataclasses import dataclass
from datetime import datetime

import requests

from backend.config import OPENWEATHERMAP_API_KEY

random.seed(7)

SEVERE_WEATHER_EVENTS = ["storm", "hurricane watch", "flooding", "heatwave", "winter storm"]

# Simulated port congestion index feed per location (in production: scrape/RSS/news API)
PORT_CONGESTION_INDEX = {
    "Port of Los Angeles, CA": 0.35,
    "Port of Long Beach, CA": 0.55,
    "Newark, NJ": 0.20,
    "Chicago, IL": 0.05,
    "Austin, TX": 0.03,
}


@dataclass
class RiskAlert:
    risk_type: str
    severity: str
    probability: float
    description: str
    recommendation: str
    supplier_name: str = None
    product_sku: str = None


def _severity_from_probability(p: float) -> str:
    if p >= 0.6:
        return "critical"
    if p >= 0.35:
        return "high"
    if p >= 0.15:
        return "medium"
    return "low"


def get_weather_risk(location: str, lat: float = None, lon: float = None) -> "RiskAlert | None":
    """Checks weather for a supplier's location. Uses real OpenWeatherMap data if a key
    is configured, otherwise a deterministic simulated forecast."""
    condition = None
    if OPENWEATHERMAP_API_KEY and lat is not None and lon is not None:
        try:
            resp = requests.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={"lat": lat, "lon": lon, "appid": OPENWEATHERMAP_API_KEY, "units": "metric"},
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                condition = data.get("weather", [{}])[0].get("main", "").lower()
        except requests.RequestException:
            condition = None

    if condition is None:
        # deterministic simulated signal, seeded by location so it's stable per run
        rnd = random.Random(location)
        condition = rnd.choice(["clear", "clear", "clear", "rain", "storm", "clear"])

    if condition in ("storm", "thunderstorm", "hurricane", "flooding"):
        prob = round(random.uniform(0.35, 0.65), 2)
        return RiskAlert(
            risk_type="weather",
            severity=_severity_from_probability(prob),
            probability=prob,
            description=f"Severe weather ({condition}) forecast near {location}, which may disrupt "
                        f"outbound shipments and last-mile delivery.",
            recommendation="Consider expediting in-transit orders and notifying customers of possible "
                           "delivery delays for weather-sensitive SKUs.",
        )
    return None


def get_port_congestion_risk(supplier) -> "RiskAlert | None":
    if not supplier.port_dependent:
        return None
    congestion = PORT_CONGESTION_INDEX.get(supplier.location, 0.15)
    # add small daily jitter so repeated runs feel "live"
    congestion = min(0.95, max(0.02, congestion + random.uniform(-0.05, 0.08)))
    if congestion >= 0.25:
        prob = round(congestion, 2)
        return RiskAlert(
            risk_type="port_congestion",
            severity=_severity_from_probability(prob),
            probability=prob,
            description=f"Supplier {supplier.name} has a {int(prob*100)}% chance of shipment delay "
                        f"due to port congestion at {supplier.location}.",
            recommendation="Increase safety stock for SKUs sourced from this supplier or split the next "
                           "PO with an alternate, non-port-dependent supplier.",
            supplier_name=supplier.name,
        )
    return None


def get_supplier_reliability_risk(supplier) -> "RiskAlert | None":
    if supplier.on_time_pct >= 0.90:
        return None
    prob = round(1 - supplier.on_time_pct, 2)
    return RiskAlert(
        risk_type="supplier_reliability",
        severity=_severity_from_probability(prob),
        probability=prob,
        description=f"Supplier {supplier.name} has an on-time delivery rate of "
                    f"{supplier.on_time_pct*100:.0f}%, implying a {int(prob*100)}% chance the next "
                    f"order arrives late.",
        recommendation="Add 2-3 days of buffer to the lead time used for reorder-point calculations, "
                       "or qualify a backup supplier for critical SKUs.",
        supplier_name=supplier.name,
    )


def get_demand_spike_risk(product_sku: str, product_name: str, forecast_future_df, history_avg: float) -> "RiskAlert | None":
    """Flags when the forecast shows a sharp upcoming demand spike vs. recent history,
    e.g. from an upcoming holiday/event -- a supply risk in its own right if PO lead
    time can't keep up."""
    if forecast_future_df is None or len(forecast_future_df) == 0 or history_avg <= 0:
        return None
    peak = forecast_future_df["yhat"].max()
    ratio = peak / history_avg
    if ratio >= 1.6:
        prob = round(min(0.95, (ratio - 1) * 0.5), 2)
        return RiskAlert(
            risk_type="demand_spike",
            severity=_severity_from_probability(prob),
            probability=prob,
            description=f"{product_name} ({product_sku}) is forecast to see demand spike to "
                        f"{peak:.0f} units/day (~{ratio:.1f}x recent average), likely from an "
                        f"upcoming seasonal/holiday event.",
            recommendation="Place a PO now sized to the spike rather than the trailing average "
                           "reorder point, to avoid a stockout during the peak window.",
            product_sku=product_sku,
        )
    return None
