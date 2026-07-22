import os
import sys
import json
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()  # reads .env in the current folder and loads it into os.environ
except ImportError:
    pass  # if python-dotenv isn't installed, fall back to manually exported vars

BASE_URL = "https://financialmodelingprep.com/stable"


def get_api_key():
    key = os.environ.get("FMP_API_KEY")
    if not key:
        raise EnvironmentError(
            "FMP_API_KEY environment variable not set. "
            "Run: export FMP_API_KEY='your-key-here'"
        )
    return key


def fetch_json(endpoint, ticker, api_key, params=None):
    """Generic helper to hit an FMP /stable/ endpoint and return parsed JSON."""
    url = f"{BASE_URL}/{endpoint}"
    query = params or {}
    query["symbol"] = ticker
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
        "grossMargin": ratios.get("grossProfitMargin"),
        "operatingMargin": ratios.get("operatingProfitMargin"),
        "netMargin": ratios.get("netProfitMargin"),
        "returnOnEquity": key_metrics.get("returnOnEquity"),
        "returnOnAssets": key_metrics.get("returnOnAssets"),

        # Debt / liquidity - used by Risk model
        "debtToEquity": ratios.get("debtToEquityRatio"),
        "currentRatio": ratios.get("currentRatio"),
        "quickRatio": ratios.get("quickRatio"),
        "interestCoverage": ratios.get("interestCoverageRatio"),  # may be null - not confirmed in this API version

        # Cash flow - used by both models
        "freeCashFlow": cashflow.get("freeCashFlow"),
        "operatingCashFlow": cashflow.get("operatingCashFlow"),

        # Valuation - used by report generator
        "peRatio": ratios.get("priceToEarningsRatio"),
        "pbRatio": ratios.get("priceToBookRatioTTM") or ratios.get("priceToBookRatio"),

        # Extra context
        "revenuePerShare": key_metrics.get("revenuePerShare"),
        "cashPerShare": key_metrics.get("cashPerShare"),
    }

    return snapshot


def get_news_headlines(ticker: str, limit: int = 10) -> list:
    """
    Fetch recent news headlines for a ticker - used as input to the
    FinBERT sentiment model (sentiment.py).
    Returns a plain list of headline strings.
    """
    api_key = get_api_key()
    url = f"{BASE_URL}/news/stock"
    try:
        resp = requests.get(
            url,
            params={"symbols": ticker, "limit": limit, "apikey": api_key},
            timeout=15,
        )
        resp.raise_for_status()
        articles = resp.json()
        if isinstance(articles, list):
            return [a.get("title") for a in articles if a.get("title")]
        return []
    except requests.exceptions.HTTPError:
        # If the news endpoint isn't available on this plan, fail gracefully
        # rather than breaking the whole pipeline.
        return []


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fetch_data.py TICKER")
        sys.exit(1)

    ticker_arg = sys.argv[1]
    try:
        data = get_financial_snapshot(ticker_arg)
        print(json.dumps(data, indent=2))

        # Quick integration test: feed this real data into the scoring model
        try:
            from scoring import score_company
            print("\n--- SCORING RESULT ---")
            print(json.dumps(score_company(data), indent=2))
        except ImportError:
            pass  # scoring.py not in this folder yet - skip

    except EnvironmentError as e:
        print(f"Error: {e}")
    except requests.exceptions.HTTPError as e:
        print(f"API error: {e}")
