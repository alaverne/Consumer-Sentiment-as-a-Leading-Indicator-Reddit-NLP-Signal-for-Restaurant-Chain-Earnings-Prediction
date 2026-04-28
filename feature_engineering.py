import numpy as np
import pandas as pd
from datetime import timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf

# ***SENTIMENT SCORING***

# I'm using VADER for the sentiment scoring. I tried FinBERT first since it's more
# finance-specific, but it actually made the model less accurate
# for this dataset — likely because Reddit language is very informal, so
# VADER's training on social media text does a better job at discerning the
# sentiment on this data.

analyzer = SentimentIntensityAnalyzer()

def score_sentiment(text):
    if not isinstance(text, str) or text.strip() == '':
        return 0
    return analyzer.polarity_scores(text)['compound']


def clean_text(text):
    if not isinstance(text, str):
        return ''

    # Decoding HTML entities to further clean up the text

    text = text.replace('&amp;', '&')
    text = text.replace('&gt;', '>')
    text = text.replace('&lt;', '<')
    text = text.replace('&nbsp;', ' ')

    return text.strip()


def filter_short_posts(posts, min_words=10, engagement_percentile=0.75):

    # I added this filter to remove low-signal posts like auto-moderator messages.
    # It keeps a post if it has 10+ words OR significant engagement.
    # I'm using a per-ticker engagement threshold rather than an absolute number —
    # 10 upvotes means something very different on r/McDonalds vs r/ShakeShack,
    # so I keep short posts that are in the top 25% of engagement for that
    # specific community.

    word_counts = []
    for text in posts['full_text']:
        if isinstance(text, str):
            word_counts.append(len(text.split()))
        else:
            word_counts.append(0)
    word_counts = pd.Series(word_counts, index=posts.index)

    thresholds = {}
    for ticker in posts['ticker'].unique():
        mask = posts['ticker'] == ticker
        thresholds[ticker] = posts.loc[mask, 'score'].quantile(engagement_percentile)
    posts['score_threshold'] = posts['ticker'].map(thresholds)

    keep = (word_counts >= min_words) | (posts['score'] >= posts['score_threshold'])
    before = len(posts)
    posts = posts[keep].copy()
    after = len(posts)
    removed = before - after
    if removed > 0:
        print(f'  Text filter: removed {removed:,} low-signal posts — {after:,} remaining')
    return posts


def score_posts(posts):
    posts = posts.copy()

    # Cleaning the text before scoring

    posts['full_text'] = posts['full_text'].apply(clean_text)
    posts = filter_short_posts(posts, min_words=10, engagement_percentile=0.75)

    posts['sent_score'] = posts['full_text'].apply(score_sentiment)

    # I'm weighting the posts by engagement, so that higher engagement posts
    # carry more signal than low engagement ones.
    # I'm using a per-ticker percentile rank rather than raw upvote counts —
    # this keeps the weighting consistent with the filter, so a top 10%
    # SHAK post gets similar weight to a top 10% MCD post regardless of the
    # raw upvote difference between those communities.

    posts['engagement_percentile'] = 0.0
    for ticker in posts['ticker'].unique():
        mask = posts['ticker'] == ticker
        posts.loc[mask, 'engagement_percentile'] = posts.loc[mask, 'score'].rank(pct=True)

    posts['weighted_sent'] = posts['sent_score'] * posts['engagement_percentile']

    return posts


# ***WEEKLY AGGREGATION***

def aggregate_weekly(posts):
    posts = posts.copy()
    posts['date'] = pd.to_datetime(posts['date'])

    weekly_sent = (
        posts
        .groupby(['ticker', pd.Grouper(key='date', freq='W')])
        .agg(
            avg_sent=('sent_score', 'mean'),
            weighted_sent=('weighted_sent', 'sum'),
            post_vol=('sent_score', 'count')
        )
        .reset_index()
    )

    # I'm using a 4-week rolling average to smooth out noise.

    weekly_sent['rolling_sent'] = (
        weekly_sent
        .groupby('ticker')['avg_sent']
        .transform(lambda x: x.rolling(4, min_periods=1).mean())
    )

    # I decided to use z-scores by ticker so that scores are comparable across
    # companies, because the threads for each company had varying baseline tones,
    # so it wasn't apples to apples. A -1.5 z-score means the same thing for both
    # after normalizing.
    # I considered min-max scaling first, but z-score felt more appropriate
    # because I don't want one extreme data point to become the floor or ceiling
    # and distort everything. Z-scoring is more stable for detecting deviations
    # from a company's normal baseline.

    weekly_sent['sent_zscore'] = (
        weekly_sent
        .groupby('ticker')['rolling_sent']
        .transform(lambda x: (x - x.mean()) / x.std())
    )

    return weekly_sent


# ***STOCK PRICES & EARNINGS***

def get_earnings(ticker):
    stock = yf.Ticker(ticker)
    earn = stock.earnings_dates
    if earn is None or len(earn) == 0:
        return pd.DataFrame()
    earn = earn.reset_index()
    earn.columns = [c.lower().replace(' ', '_') for c in earn.columns]
    earn['ticker'] = ticker
    earn['date'] = pd.to_datetime(earn['earnings_date']).dt.tz_localize(None)
    if 'surprise(%)' in earn.columns:
        earn['beat'] = (earn['surprise(%)'] > 0).astype(int)
        earn['eps_surprise_pct'] = earn['surprise(%)']
    return earn[['ticker', 'date', 'beat', 'eps_surprise_pct']].dropna(subset=['beat'])


def get_all_earnings(tickers, start='2022-01-01', end='2026-04-01'):

    # yfinance earnings dates can sometimes be off by 1-2 days, so
    # in production these would need to be validated against a second source
    # like the Nasdaq earnings calendar or Refinitiv before computing
    # any pre-earnings features — a one day error shifts all the window
    # boundaries and could accidentally include post-announcement posts.

    dfs = [get_earnings(t) for t in tickers]
    earn_df = pd.concat([d for d in dfs if len(d) > 0], ignore_index=True)
    earn_df = earn_df[
        (earn_df['date'] >= start) &
        (earn_df['date'] <= end)
    ].copy()
    print(f'Earnings events: {len(earn_df)}')
    print(earn_df['ticker'].value_counts().to_string())
    return earn_df


# ***PRE-EARNINGS SENTIMENT WINDOWS***

# I'm using the average z-scored sentiment in the 30 day window leading up
# to each earnings release. This is long enough to capture trend, but short
# enough to remain relevant to the quarter at hand.
#
# ***UPDATE***
#
# I added a 7-day window to help catch recent sentiment shifts that get
# buried in the 30-day average.
#
# The rigorous approach would be to test all possible lookback periods
# and find the optimal window empirically. I didn't do that here because
# finding a new optimal window would trickle down through the feature
# engineering, model training, and backtest results.

def get_pre_earn_sent(ticker, earn_date, sent_df, window_days=30):
    window_start = earn_date - timedelta(days=window_days)
    mask = (
        (sent_df['ticker'] == ticker) &
        (sent_df['date'] >= window_start) &
        (sent_df['date'] < earn_date)
    )
    window = sent_df[mask]
    if len(window) == 0:
        return np.nan
    return window['sent_zscore'].mean()


def build_features(posts, earn_df, weekly_sent=None):
    posts = posts.copy()
    posts['date'] = pd.to_datetime(posts['date'])
    earn_df = earn_df.copy()
    earn_df['date'] = pd.to_datetime(earn_df['date'])

    if weekly_sent is None:
        weekly_sent = aggregate_weekly(posts)

    # original 30-day window from the first version of this project

    pre30 = []
    for _, row in earn_df.iterrows():
        pre30.append(get_pre_earn_sent(row['ticker'], row['date'], weekly_sent, 30))
    earn_df['pre30_sent'] = pre30

    # Adding the 7-day window

    pre7 = []
    for _, row in earn_df.iterrows():
        pre7.append(get_pre_earn_sent(row['ticker'], row['date'], weekly_sent, 7))
    earn_df['pre7_sent'] = pre7

    # Testing sentiment shift between the 30-day and 7-day windows

    earn_df['sentiment_shift_30_7'] = earn_df['pre7_sent'] - earn_df['pre30_sent']

    return earn_df


def temporal_split(df, cutoff='2024-01-01'):

    # To avoid look-ahead bias I'm doing a walk-forward split —
    # training on older data and testing on newer data

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    cutoff_ts = pd.Timestamp(cutoff)
    train = df[df['date'] < cutoff_ts].copy()
    test = df[df['date'] >= cutoff_ts].copy()
    print(f'Train: {len(train)} events (pre-{cutoff[:7]})')
    print(f'Test:  {len(test)} events ({cutoff[:7]}+) — out of sample')
    return train, test