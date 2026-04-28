import requests
import pandas as pd
from datetime import datetime
import time
import os

ARCTIC_SHIFT_URL = 'https://arctic-shift.photon-reddit.com/api/posts/search'

# I'm scraping one month at a time to keep the time series clean. I tried
# paginating to get more posts per month, but I kept losing data.

def scrape_subreddit_monthly(subreddit_name, ticker, posts_per_month=100,
                              start='2022-01-01', end='2026-04-01'):
    posts = []
    months = pd.date_range(start=start, end=end, freq='MS')

    for month_start in months:
        month_end = month_start + pd.DateOffset(months=1)
        params = {
            'subreddit': subreddit_name,
            'limit': min(posts_per_month, 100),
            'after': month_start.strftime('%Y-%m-%dT%H:%M:%S'),
            'before': month_end.strftime('%Y-%m-%dT%H:%M:%S'),
            'sort': 'asc'
        }
        r = requests.get(ARCTIC_SHIFT_URL, params=params)
        if r.status_code != 200:
            print(f'Error {r.status_code} for {month_start.strftime("%Y-%m")}')
            continue
        data = r.json().get('data', [])
        for post in data:
            text = post.get('selftext', '')
            if text in ['[deleted]', '[removed]']:
                text = ''
            posts.append({
                'ticker': ticker,
                'title': post.get('title', ''),
                'text': text,
                'score': post.get('score', 0),
                'num_comments': post.get('num_comments', 0),
                'timestamp': datetime.utcfromtimestamp(post.get('created_utc', 0)),
                'full_text': post.get('title', '') + ' ' + text
            })
        print(f'r/{subreddit_name} {month_start.strftime("%Y-%m")}: {len(data)} posts')
        time.sleep(0.5)

    df = pd.DataFrame(posts)
    if len(df) == 0:
        df = pd.DataFrame(columns=['ticker','title','text','score',
                                   'num_comments','timestamp','full_text','date'])
    else:
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
    print(f'Total: {len(df)} posts from r/{subreddit_name}\n')
    return df


# ***COMPANY SELECTION***
# I started this analysis looking at CMG, SBUX, NKE, but ended up dropping
# and replacing SBUX and NKE.
#
# The r/starbucks thread is full of employees talking about working conditions
# and unionization, rather than customer experience sentiment. The model
# actually confirmed this, as SBUX had high positive sentiment before misses,
# which is the opposite of what should happen if the signal worked.
#
# In the r/Nike thread, there were a bunch of posts where people were reselling
# things or talking about new releases. It wasn't really representative of
# Nike's broader consumer base. There was also only one earnings miss in the
# whole span of the data, so I couldn't really train the model with that.
#
# After removing those two companies, I added a few other restaurant chains
# where the subreddits were more clearly customer-focused (people complaining
# about portion sizes, wait times, new menu items, etc.), as these are better
# indicators of sales.

# ***UPDATE***

# I'm expanding on my original project and adding WING and JACK.

# I also extended the date range from 2024-01-01 to 2026-04-01 to
# capture more earnings events — the original dataset had 50 events
# across 5 tickers. Expanding the universe to 7 tickers and extending
# through April 2026 brings the total to 119 earnings events, which
# is an improvement from the original, but still falls short of the 30+ positive
# and negative examples in the test set needed for fully robust
# classification metrics.


UNIVERSE = [
    ('chipotle',    'CMG'),
    ('McDonalds',   'MCD'),
    ('Dominos',     'DPZ'),
    ('Wendys',      'WEN'),
    ('ShakeShack',  'SHAK'),
    ('wingstop',    'WING'),
    ('jackinthebox','JACK'),
]


def scrape_all(data_dir='data/raw'):
    os.makedirs(data_dir, exist_ok=True)
    all_dfs = []

    for sub, ticker in UNIVERSE:
        pm = 50 if ticker in ('SHAK', 'WING', 'JACK') else 100
        df = scrape_subreddit_monthly(sub, ticker, posts_per_month=pm)
        csv_path = os.path.join(data_dir, f'{ticker}_posts.csv')
        df.to_csv(csv_path, index=False)
        print(f'Saved {ticker} — {len(df)} posts')
        all_dfs.append(df)

    all_posts = pd.concat(all_dfs, ignore_index=True)
    all_posts.to_csv(os.path.join(data_dir, 'all_posts.csv'), index=False)

    print(f'\nTotal posts: {len(all_posts)}')
    print(all_posts['ticker'].value_counts())
    return all_posts


def load_posts(data_dir='data/raw'):
    path = os.path.join(data_dir, 'all_posts.csv')
    if os.path.exists(path):
        print(f'Loading cached posts from {path}')
        return pd.read_csv(path)
    print('No cached data found. Running scraper...')
    return scrape_all(data_dir=data_dir)