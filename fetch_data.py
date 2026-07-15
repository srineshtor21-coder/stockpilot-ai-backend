"""
StockPilot AI - Session 1: Data Pipeline
------------------------------------------
Fetches financial data for a given stock ticker from Financial Modeling Prep (FMP)
and returns a clean, structured JSON object ready for the ML models in later sessions.

USAGE:
    export FMP_API_KEY="your-key-here"
    python fetch_data.py AAPL

OUTPUT:
    Prints a JSON object with company profile, key ratios, growth metrics,
    and cash flow data - the raw inputs for the Financial Health / Risk models.
"""

import os
import sys
import json
import requests

BASE_URL = "https://financialmodelingprep.com/api/v3"


def get_api_key():
    key = os.environ.get("FMP_API_KEY")
    if not key:
        raise EnvironmentError(
            "FMP_API_KEY environment variable not set. "
            "Run: export FMP_API_KEY='your-key-here'"
        )
    return key


def fetch_json(endpoint, ticker, api_key, params=None):
    """Generic helper to hit an FMP endpoint and return parsed JSON."""
    url = f"{BASE_URL}/{endpoint}/{ticker}"
    query = params or {}
    query["apikey"] = api_key
    resp = requests.get(url, params=query, timeout=15)
    resp.raise_for_status()
    return resp.json()


def safe_first(data):
    """FMP often returns a list with one dict per period - grab the most recent."""
    if isinstance(data, list) and len(data) > 0:
        return data[0]
    return {}


def get_financial_snapshot(ticker: str) -> dict:
    api_key = get_api_key()

    profile = safe_first(fetch_json("profile", ticker, api_key))
    ratios = safe_first(fetch_json("ratios", ticker, api_key, {"limit": 1}))
    growth = safe_first(fetch_json("financial-growth", ticker, api_key, {"limit": 1}))
    cashflow = safe_first(fetch_json("cash-flow-statement", ticker, api_key, {"limit": 1}))
    key_metrics = safe_first(fetch_json("key-metrics", ticker, api_key, {"limit": 1}))

    snapshot = {
        "ticker": ticker.upper(),
        "companyName": profile.get("companyName"),
        "sector": profile.get("sector"),
        "industry": profile.get("industry"),
        "marketCap": profile.get("mktCap"),
        "price": profile.get("price"),
        "beta": profile.get("beta"),
        "description": profile.get("description"),

        # Growth metrics - used by Financial Health model
        "revenueGrowth": growth.get("revenueGrowth"),
        "epsGrowth": growth.get("epsgrowth"),
        "grossProfitGrowth": growth.get("grossProfitGrowth"),

        # Profitability ratios - used by Financial Health model
        "grossMargin": ratios.get("grossProfitMarginTTM") or ratios.get("grossProfitMargin"),
        "operatingMargin": ratios.get("operatingProfitMarginTTM") or ratios.get("operatingProfitMargin"),
        "netMargin": ratios.get("netProfitMarginTTM") or ratios.get("netProfitMargin"),
        "returnOnEquity": ratios.get("returnOnEquityTTM") or ratios.get("returnOnEquity"),
        "returnOnAssets": ratios.get("returnOnAssetsTTM") or ratios.get("returnOnAssets"),

        # Debt / liquidity - used by Risk model
        "debtToEquity": ratios.get("debtEquityRatioTTM") or ratios.get("debtEquityRatio"),
        "currentRatio": ratios.get("currentRatioTTM") or ratios.get("currentRatio"),
        "quickRatio": ratios.get("quickRatioTTM") or ratios.get("quickRatio"),
        "interestCoverage": ratios.get("interestCoverageTTM") or ratios.get("interestCoverage"),

        # Cash flow - used by both models
        "freeCashFlow": cashflow.get("freeCashFlow"),
        "operatingCashFlow": cashflow.get("operatingCashFlow"),

        # Valuation - used by report generator
        "peRatio": ratios.get("priceEarningsRatioTTM") or ratios.get("priceEarningsRatio"),
        "pbRatio": ratios.get("priceToBookRatioTTM") or ratios.get("priceToBookRatio"),

        # Extra context
        "revenuePerShare": key_metrics.get("revenuePerShare"),
        "cashPerShare": key_metrics.get("cashPerShare"),
    }

    return snapshot


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fetch_data.py TICKER")
        sys.exit(1)

    ticker_arg = sys.argv[1]
    try:
        data = get_financial_snapshot(ticker_arg)
        print(json.dumps(data, indent=2))
    except EnvironmentError as e:
        print(f"Error: {e}")
    except requests.exceptions.HTTPError as e:
        print(f"API error: {e}")
