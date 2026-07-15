"""
StockPilot AI - Session 2: Financial Health & Risk Scoring
-------------------------------------------------------------
Takes the JSON snapshot produced by fetch_data.py (Session 1) and computes:
  1. Financial Health Score (0-100) - how strong the business fundamentals are
  2. Risk Level (Low / Moderate / High) - how risky the stock is to hold

This is a transparent, rules-based weighted scoring system rather than a
"black box" ML model. That's intentional: it's explainable (you can show
exactly why a company scored the way it did), and it doesn't require a
large labeled training dataset to get started. A trained ML model (e.g.
XGBoost) can be swapped in later using this same feature set once you've
collected enough historical data to create real training labels.

USAGE:
    from fetch_data import get_financial_snapshot
    from scoring import score_company

    snapshot = get_financial_snapshot("AAPL")
    result = score_company(snapshot)
    print(result)
"""

import json


def _clamp(value, low=0, high=100):
    return max(low, min(high, value))


def _safe(value, default=0):
    """Return 0 instead of None so math doesn't break on missing data."""
    return value if isinstance(value, (int, float)) else default


def score_financial_health(data: dict) -> dict:
    """
    Weighted scoring across growth, profitability, and cash generation.
    Each sub-score is 0-100, then combined with weights into a final score.
    """
    revenue_growth = _safe(data.get("revenueGrowth"))
    gross_margin = _safe(data.get("grossMargin"))
    operating_margin = _safe(data.get("operatingMargin"))
    net_margin = _safe(data.get("netMargin"))
    roe = _safe(data.get("returnOnEquity"))
    fcf = _safe(data.get("freeCashFlow"))

    # --- Growth score: reward steady positive growth, penalize decline ---
    growth_score = _clamp(50 + (revenue_growth * 200))

    # --- Profitability score: blend of margins, benchmarked against
    # realistic ranges rather than treating raw % as the score directly.
    # e.g. a 20% operating margin is genuinely strong, not "20/100 strong."
    def _margin_to_score(margin, strong_at):
        # 0% margin -> ~35 score, margin == strong_at -> 100 score
        return _clamp(35 + (margin / strong_at) * 65)

    gross_component = _margin_to_score(gross_margin, strong_at=0.50)
    operating_component = _margin_to_score(operating_margin, strong_at=0.25)
    net_component = _margin_to_score(net_margin, strong_at=0.20)

    profitability_score = _clamp(
        (gross_component * 0.3)
        + (operating_component * 0.4)
        + (net_component * 0.3)
    )

    # --- Efficiency score: how well the company turns equity into profit ---
    efficiency_score = _clamp(50 + (roe * 150))

    # --- Cash generation score: simple positive/negative check, scaled ---
    cash_score = 70 if fcf > 0 else 30

    final_score = round(
        (growth_score * 0.25)
        + (profitability_score * 0.35)
        + (efficiency_score * 0.25)
        + (cash_score * 0.15)
    )
    final_score = int(_clamp(final_score))

    if final_score >= 80:
        label = "Excellent"
    elif final_score >= 65:
        label = "Strong"
    elif final_score >= 50:
        label = "Moderate"
    elif final_score >= 35:
        label = "Weak"
    else:
        label = "Poor"

    reasons = []
    reasons.append(
        ("✓" if revenue_growth > 0.05 else "✗")
        + f" Revenue growth: {round(revenue_growth * 100, 1)}%"
    )
    reasons.append(
        ("✓" if operating_margin > 0.15 else "✗")
        + f" Operating margin: {round(operating_margin * 100, 1)}%"
    )
    reasons.append(
        ("✓" if roe > 0.10 else "✗")
        + f" Return on equity: {round(roe * 100, 1)}%"
    )
    reasons.append(
        ("✓" if fcf > 0 else "✗")
        + f" Free cash flow: {'positive' if fcf > 0 else 'negative'}"
    )

    return {
        "score": final_score,
        "label": label,
        "reasons": reasons,
        "subscores": {
            "growth": round(growth_score),
            "profitability": round(profitability_score),
            "efficiency": round(efficiency_score),
            "cashGeneration": cash_score,
        },
    }


def score_risk(data: dict) -> dict:
    """
    Weighted scoring across leverage, liquidity, and volatility.
    Higher risk_points = riskier. Scale is roughly 0-100.
    """
    debt_to_equity = _safe(data.get("debtToEquity"))
    current_ratio = _safe(data.get("currentRatio"), default=1)
    quick_ratio = _safe(data.get("quickRatio"), default=1)
    beta = _safe(data.get("beta"), default=1)
    interest_coverage = _safe(data.get("interestCoverage"), default=5)

    risk_points = 0
    flags = []

    # Leverage risk
    if debt_to_equity > 2:
        risk_points += 30
        flags.append("High debt relative to equity")
    elif debt_to_equity > 1:
        risk_points += 15
        flags.append("Moderate debt levels")

    # Liquidity risk
    if current_ratio < 1:
        risk_points += 20
        flags.append("Current liabilities exceed current assets")
    elif current_ratio < 1.5:
        risk_points += 10
        flags.append("Thin liquidity buffer")

    if quick_ratio < 0.8:
        risk_points += 10
        flags.append("Low quick ratio (limited liquid assets)")

    # Volatility risk
    if beta > 1.5:
        risk_points += 20
        flags.append("High volatility relative to the market (beta > 1.5)")
    elif beta > 1.1:
        risk_points += 10
        flags.append("Above-average volatility")

    # Interest coverage risk (can the company service its debt from earnings?)
    if interest_coverage < 2:
        risk_points += 20
        flags.append("Low interest coverage - earnings barely cover debt payments")
    elif interest_coverage < 5:
        risk_points += 5

    risk_points = int(_clamp(risk_points))

    if risk_points >= 55:
        level = "High"
    elif risk_points >= 25:
        level = "Moderate"
    else:
        level = "Low"

    if not flags:
        flags.append("No major risk flags detected in available data")

    return {
        "riskPoints": risk_points,
        "level": level,
        "flags": flags,
    }


def score_company(data: dict) -> dict:
    """Combine health + risk into one payload ready for the LLM report layer."""
    health = score_financial_health(data)
    risk = score_risk(data)

    return {
        "ticker": data.get("ticker"),
        "companyName": data.get("companyName"),
        "financialHealth": health,
        "risk": risk,
    }


if __name__ == "__main__":
    # Quick manual test using placeholder data so this runs with zero setup.
    sample_data = {
        "ticker": "TEST",
        "companyName": "Test Corp",
        "revenueGrowth": 0.18,
        "grossMargin": 0.45,
        "operatingMargin": 0.22,
        "netMargin": 0.15,
        "returnOnEquity": 0.28,
        "freeCashFlow": 5_000_000_000,
        "debtToEquity": 0.8,
        "currentRatio": 1.8,
        "quickRatio": 1.2,
        "beta": 1.2,
        "interestCoverage": 12,
    }
    print(json.dumps(score_company(sample_data), indent=2))
