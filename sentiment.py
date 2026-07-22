"""
StockPilot AI - Session 3: News Sentiment (FinBERT)
-------------------------------------------------------
Uses a pretrained finance-specific NLP model (FinBERT) to classify news
headlines as Bullish / Bearish / Neutral, instead of asking a general LLM
each time. This is inference only - no training required.

MODEL USED: ProsusAI/finbert (a widely-used, freely available FinBERT
checkpoint fine-tuned on financial text specifically for sentiment).

USAGE:
    from sentiment import analyze_headlines

    headlines = [
        "Nvidia signs major AI partnership deal with cloud providers",
        "Company faces regulatory scrutiny over data practices",
    ]
    result = analyze_headlines(headlines)
    print(result)

NOTE ON FIRST RUN:
    The first time this runs, it downloads the FinBERT model weights
    (~400MB) from Hugging Face and caches them locally. This requires
    an internet connection the first time only; after that it loads
    from the local cache and works offline.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "ProsusAI/finbert"

_tokenizer = None
_model = None


def _load_model():
    """Lazy-load the model so importing this file doesn't immediately
    trigger a slow download - it only downloads/loads on first actual use."""
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


def analyze_headline(text: str) -> dict:
    """Classify a single headline. Returns label + confidence per class."""
    tokenizer, model = _load_model()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

    # FinBERT (ProsusAI) label order: 0=positive, 1=negative, 2=neutral
    labels = ["Bullish", "Bearish", "Neutral"]
    scores = {label: round(float(prob), 4) for label, prob in zip(labels, probs)}
    top_label = max(scores, key=scores.get)

    return {
        "text": text,
        "sentiment": top_label,
        "confidence": scores[top_label],
        "scores": scores,
    }


def analyze_headlines(headlines: list[str]) -> dict:
    """
    Classify a batch of headlines and return both the individual results
    and an aggregated overall sentiment - what your dashboard will show
    as the "News Sentiment" card for a given ticker.
    """
    if not headlines:
        return {"overall": "Neutral", "confidence": 0.0, "results": []}

    results = [analyze_headline(h) for h in headlines]

    # Aggregate: average the bullish/bearish/neutral scores across all headlines
    totals = {"Bullish": 0.0, "Bearish": 0.0, "Neutral": 0.0}
    for r in results:
        for label, score in r["scores"].items():
            totals[label] += score

    n = len(results)
    averages = {label: round(score / n, 4) for label, score in totals.items()}
    overall = max(averages, key=averages.get)

    return {
        "overall": overall,
        "confidence": averages[overall],
        "breakdown": averages,
        "results": results,
    }


if __name__ == "__main__":
    sample_headlines = [
        "Company reports record quarterly revenue, beating analyst expectations",
        "Stock drops after CEO announces unexpected resignation",
        "Firm maintains steady guidance for next fiscal year",
    ]
    output = analyze_headlines(sample_headlines)
    import json
    print(json.dumps(output, indent=2))
