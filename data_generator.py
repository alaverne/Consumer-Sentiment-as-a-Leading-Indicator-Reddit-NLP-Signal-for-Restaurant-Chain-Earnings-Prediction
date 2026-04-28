import numpy as np
import pandas as pd
from datetime import datetime, timedelta

TICKERS = ["CMG", "MCD", "DPZ", "WEN", "SHAK", "WING", "JACK"]

np.random.seed(42)


def generate_earnings_data():
    rows = []
    for ticker in TICKERS:
        base = datetime(2022, 1, 15)

        if ticker == "CMG":
            mu = 4.0
        elif ticker == "MCD":
            mu = 1.5
        elif ticker == "DPZ":
            mu = 2.0
        elif ticker == "WEN":
            mu = -0.5
        elif ticker == "SHAK":
            mu = 5.0
        elif ticker == "WING":
            mu = 6.0
        else:
            mu = 1.0

        for q in range(10):
            offset = timedelta(days=int(q * 91 + np.random.randint(-7, 7)))
            edate = base + offset
            consensus = np.random.uniform(1.0, 6.0)
            surprise = np.random.normal(mu, 8.0)
            actual = consensus * (1 + surprise / 100)
            rows.append({
                "ticker": ticker,
                "date": edate,
                "actual_eps": round(actual, 2),
                "consensus_eps": round(consensus, 2),
                "eps_surprise_pct": round(surprise, 4),
                "beat": int(surprise > 0),
            })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def generate_reddit_posts(n_posts=12000):
    posts = []
    weights = [0.25, 0.20, 0.15, 0.15, 0.10, 0.10, 0.05]
    for i in range(n_posts):
        ticker = np.random.choice(TICKERS, p=weights)
        post_date = datetime(2022, 1, 1) + timedelta(
            days=int(np.random.uniform(0, 365 * 2.4))
        )

        if ticker == "CMG":
            mu_s = 0.05
        elif ticker == "MCD":
            mu_s = 0.02
        elif ticker == "DPZ":
            mu_s = 0.04
        elif ticker == "WEN":
            mu_s = -0.02
        elif ticker == "SHAK":
            mu_s = 0.07
        elif ticker == "WING":
            mu_s = 0.06
        else:
            mu_s = 0.01

        compound = float(np.clip(np.random.normal(mu_s, 0.35), -1, 1))
        score = int(np.random.exponential(50))
        comments = int(np.random.exponential(20))
        full_text = "went to " + ticker.lower() + " today word0 word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11"

        posts.append({
            "ticker": ticker,
            "date": post_date,
            "post_id": "post_" + str(i),
            "score": score,
            "num_comments": comments,
            "sent_score": compound,
            "full_text": full_text,
        })

    df = pd.DataFrame(posts)
    df["date"] = pd.to_datetime(df["date"])
    df["engagement_percentile"] = 0.0
    for ticker in df["ticker"].unique():
        mask = df["ticker"] == ticker
        df.loc[mask, "engagement_percentile"] = df.loc[mask, "score"].rank(pct=True)
    df["weighted_sent"] = df["sent_score"] * df["engagement_percentile"]
    return df.sort_values("date").reset_index(drop=True)