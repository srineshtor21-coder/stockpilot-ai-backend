"""
StockPilot AI - Session 4: FastAPI Backend
------------------------------------------------
Combines fetch_data.py + scoring.py + sentiment.py into one API that your
Lovable frontend can call.

USAGE (local testing):
    pip install fastapi uvicorn
    uvicorn main:app --reload

Then visit http://127.0.0.1:8000/docs to test it interactively, or call:
    http://127.0.0.1:8000/analyze?ticker=AAPL

DEPLOYMENT:
    This is designed to run on Render/Railway using:
        uvicorn main:app --host 0.0.0.0 --port $PORT
"""

import time
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from fetch_data import get_financial_snapshot, get_news_headlines
from scoring import score_company

app = FastAPI(title="StockPilot AI Backend")

# Allow your Lovable frontend (and local dev) to call this API from the browser.
# Once you know your Lovable site's exact domain, replace "*" with that domain
# for better security.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Simple in-memory cache -----------------------------------------------
# Avoids re-fetching the same ticker's data repeatedly (saves FMP API calls
# and speeds up repeat searches). Cache expires after CACHE_TTL_SECONDS.
_cache = {}
CACHE_TTL_SECONDS = 60 * 60 * 6  # 6 hours


def _get_cached(ticker: str):
    entry = _cache.get(ticker)
    if entry and (time.time() - entry["timestamp"] < CACHE_TTL_SECONDS):
        return entry["data"]
    return None


def _set_cache(ticker: str, data: dict):
    _cache[ticker] = {"data": data, "timestamp": time.time()}


# --- Sentiment model: loaded lazily, kept in memory -----------------------
# We import sentiment.py's function lazily inside the request handler rather
# than at module load time, so the (slow, one-time) FinBERT download/load
# only happens the first time it's actually needed - not every server restart
# during development. Once loaded, it stays cached in memory for all
# subsequent requests (see sentiment.py's own _model cache).
def _get_sentiment(headlines: list) -> dict:
    if not headlines:
        return {"overall": "Neutral", "confidence": 0.0, "breakdown": {}, "results": []}
    from sentiment import analyze_headlines
    return analyze_headlines(headlines)


@app.get("/")
def root():
    return {"status": "StockPilot AI backend is running"}


@app.get("/analyze")
def analyze(ticker: str = Query(..., description="Stock ticker, e.g. AAPL")):
    ticker = ticker.upper().strip()

    cached = _get_cached(ticker)
    if cached:
        return cached

    try:
        financial_data = get_financial_snapshot(ticker)
    except EnvironmentError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch financial data: {e}")

    if not financial_data.get("companyName"):
        raise HTTPException(status_code=404, detail=f"No data found for ticker '{ticker}'")

    scores = score_company(financial_data)

    headlines = get_news_headlines(ticker)
    sentiment = _get_sentiment(headlines)

    result = {
        "ticker": ticker,
        "companyName": financial_data.get("companyName"),
        "sector": financial_data.get("sector"),
        "industry": financial_data.get("industry"),
        "price": financial_data.get("price"),
        "description": financial_data.get("description"),
        "financialHealth": scores["financialHealth"],
        "risk": scores["risk"],
        "sentiment": sentiment,
        "keyMetrics": {
            "revenueGrowth": financial_data.get("revenueGrowth"),
            "grossMargin": financial_data.get("grossMargin"),
            "operatingMargin": financial_data.get("operatingMargin"),
            "netMargin": financial_data.get("netMargin"),
            "peRatio": financial_data.get("peRatio"),
            "pbRatio": financial_data.get("pbRatio"),
            "beta": financial_data.get("beta"),
        },
    }

    _set_cache(ticker, result)
    return result
