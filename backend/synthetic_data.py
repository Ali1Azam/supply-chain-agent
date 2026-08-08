"""
Synthetic data generator.

In production this module would be replaced by real connectors to the
customer's ERP, POS system, and supplier portals (Methodology step 1:
"Data Integration"). For this reference implementation we simulate ~2 years
of realistic daily sales history per SKU, with:

  - Weekly + yearly seasonality
  - Promotion spikes
  - Weather correlation for weather-sensitive categories (e.g. beverages,
    umbrellas, ice cream)
  - Holiday/event demand bumps
  - Competitor pricing pressure
  - Supplier lead times & reliability

This lets every downstream module (forecasting, ROP/EOQ, risk, PO
generation) be demoed end-to-end without any external credentials.
"""
import random
import math
from datetime import date, timedelta

import numpy as np

random.seed(42)
np.random.seed(42)

SUPPLIERS = [
    dict(name="Pacific Rim Distributors", location="Port of Los Angeles, CA",
         port_dependent=True, avg_lead_time_days=14, on_time_pct=0.82, reliability_score=0.80, price_index=0.95),
    dict(name="Midwest Wholesale Co.", location="Chicago, IL",
         port_dependent=False, avg_lead_time_days=5, on_time_pct=0.95, reliability_score=0.93, price_index=1.05),
    dict(name="Global Freight Partners", location="Port of Long Beach, CA",
         port_dependent=True, avg_lead_time_days=18, on_time_pct=0.75, reliability_score=0.72, price_index=0.88),
    dict(name="Regional Fresh Supply", location="Austin, TX",
         port_dependent=False, avg_lead_time_days=3, on_time_pct=0.97, reliability_score=0.96, price_index=1.10),
    dict(name="EverStock Industrial", location="Newark, NJ",
         port_dependent=True, avg_lead_time_days=10, on_time_pct=0.88, reliability_score=0.85, price_index=1.0),
]

# (sku, name, category, unit_cost, unit_price, base_daily_demand, weekly_amp,
#  yearly_amp, weather_sensitive, trend_per_year, noise_std)
PRODUCTS = [
    ("BEV-001", "Sparkling Water 12pk", "Beverages", 4.20, 8.99, 42, 0.25, 0.35, True, 0.06, 6),
    ("BEV-002", "Cold Brew Coffee 4pk", "Beverages", 5.10, 11.49, 30, 0.20, 0.15, True, 0.10, 5),
    ("ICE-001", "Premium Ice Cream Pint", "Frozen", 2.80, 6.49, 25, 0.30, 0.55, True, 0.04, 5),
    ("SNK-001", "Organic Trail Mix 8oz", "Snacks", 2.10, 4.99, 55, 0.15, 0.10, False, 0.08, 7),
    ("SNK-002", "Kettle Chips Family Size", "Snacks", 1.85, 4.29, 60, 0.35, 0.05, False, 0.02, 8),
    ("HHD-001", "Paper Towels 6-roll", "Household", 6.50, 12.99, 35, 0.10, 0.05, False, 0.05, 4),
    ("HHD-002", "Dish Soap 24oz", "Household", 2.20, 4.49, 40, 0.08, 0.05, False, 0.01, 5),
    ("SEA-001", "Rain Umbrella Compact", "Seasonal", 3.90, 12.99, 8, 0.10, 1.40, True, 0.00, 3),
    ("SEA-002", "Portable Fan", "Seasonal", 7.50, 19.99, 6, 0.10, 1.60, True, 0.03, 3),
    ("PPC-001", "Recycled Paper Plates 50ct", "Party & Paper", 3.20, 7.99, 18, 0.40, 0.30, False, 0.02, 4),
    ("BAK-001", "Sourdough Bread Loaf", "Bakery", 1.60, 4.49, 70, 0.45, 0.10, False, 0.03, 9),
    ("DAI-001", "Whole Milk Gallon", "Dairy", 2.40, 4.29, 90, 0.20, 0.10, False, 0.01, 10),
    ("DAI-002", "Greek Yogurt 32oz", "Dairy", 3.10, 6.49, 45, 0.15, 0.10, False, 0.05, 6),
    ("PET-001", "Dog Food 15lb Bag", "Pet Supplies", 12.50, 24.99, 15, 0.05, 0.05, False, 0.09, 3),
    ("ELC-001", "AA Batteries 8pk", "Electronics", 3.80, 8.99, 20, 0.10, 0.20, False, 0.02, 4),
]

HOLIDAY_WINDOWS = [
    # (month, day, spread_days, demand_multiplier)
    (1, 1, 3, 1.3), (2, 14, 2, 1.5), (5, 27, 4, 1.6), (7, 4, 4, 1.7),
    (9, 2, 3, 1.4), (11, 28, 5, 2.2), (12, 24, 6, 2.6),
]


def _is_holiday_window(d: date):
    for month, day, spread, mult in HOLIDAY_WINDOWS:
        try:
            h = date(d.year, month, day)
        except ValueError:
            continue
        if abs((d - h).days) <= spread:
            return True, mult
    return False, 1.0


def _synthetic_temperature(d: date):
    """Rough seasonal temperature curve (Celsius) with daily noise."""
    day_of_year = d.timetuple().tm_yday
    seasonal = 15 + 12 * math.sin(2 * math.pi * (day_of_year - 100) / 365)
    return round(seasonal + np.random.normal(0, 3), 1)


def generate_sales_history(start: date, end: date):
    """Returns dict: sku -> list[dict(date, units_sold, promotion_flag, temperature_c,
    is_holiday_event, competitor_price_index, stock_on_hand)]"""
    all_days = [(start + timedelta(days=i)) for i in range((end - start).days + 1)]
    history = {}

    for (sku, name, category, unit_cost, unit_price, base, weekly_amp, yearly_amp,
         weather_sens, trend_per_year, noise_std) in PRODUCTS:
        records = []
        simulated_stock = base * 20  # start with ~20 days of cover
        promo_cooldown = 0

        for i, d in enumerate(all_days):
            day_of_year = d.timetuple().tm_yday
            years_elapsed = i / 365.0

            weekday_factor = 1 + weekly_amp * (0.6 if d.weekday() < 5 else 1.0) * \
                math.sin(2 * math.pi * (d.weekday() / 7))
            yearly_factor = 1 + yearly_amp * math.sin(2 * math.pi * (day_of_year - 60) / 365)
            trend_factor = 1 + trend_per_year * years_elapsed

            temp = _synthetic_temperature(d)
            weather_factor = 1.0
            if weather_sens:
                # warm-weather products (beverages/ice cream/fans) sell more when hot;
                # umbrellas sell more on simulated "rainy" (cooler + random rain flag) days
                if sku.startswith("SEA-001"):  # umbrella: inverse-ish, more on cold/random rain
                    rain = np.random.random() < 0.18
                    weather_factor = 1.8 if rain else 0.9
                else:
                    weather_factor = 1 + max(-0.4, min(0.6, (temp - 18) / 25))

            is_holiday, holiday_mult = _is_holiday_window(d)

            promo = False
            if promo_cooldown <= 0 and np.random.random() < 0.03:
                promo = True
                promo_cooldown = 10
            promo_cooldown -= 1
            promo_mult = 1.6 if promo else 1.0

            competitor_price_index = round(1.0 + np.random.normal(0, 0.03), 3)
            competitor_pressure = 1 + max(-0.15, min(0.15, (1 - competitor_price_index) * 0.5))

            expected = (base * weekday_factor * yearly_factor * trend_factor *
                        weather_factor * holiday_mult * promo_mult * competitor_pressure)
            units_sold = max(0, int(round(np.random.normal(expected, noise_std))))

            # naive replenishment simulation so stock_on_hand looks realistic in history
            simulated_stock -= units_sold
            if simulated_stock < base * 5:
                simulated_stock += base * 18  # simulate a restock event in the past
            simulated_stock = max(0, simulated_stock)

            records.append(dict(
                date=d,
                units_sold=units_sold,
                promotion_flag=promo,
                temperature_c=temp,
                is_holiday_event=is_holiday,
                competitor_price_index=competitor_price_index,
                stock_on_hand=int(simulated_stock),
            ))
        history[sku] = records

    return history


def product_master():
    return PRODUCTS


def supplier_master():
    return SUPPLIERS
