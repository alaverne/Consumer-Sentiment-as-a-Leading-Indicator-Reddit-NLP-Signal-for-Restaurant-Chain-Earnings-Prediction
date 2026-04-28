import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta


def get_earnings_return(ticker, earnings_date, days_before=1, days_after=1):

    # Pulling price data from yfinance for each earnings event
    # Buying at the open on T-1, selling at the close on T+1
    # Returns None if the data isn't available

    try:
        start = pd.Timestamp(earnings_date) - timedelta(days=days_before + 3)
        end = pd.Timestamp(earnings_date) + timedelta(days=days_after + 3)
        hist = yf.Ticker(ticker).history(start=start, end=end)
        if len(hist) < 2:
            return None
        entry_price = hist['Open'].iloc[0]
        exit_price = hist['Close'].iloc[-1]
        return (exit_price - entry_price) / entry_price
    except Exception:
        return None


def simulate_earnings_returns(events, seed=42):

    # Using simulated returns for demo mode — live mode uses real prices

    np.random.seed(seed)
    events = events.copy()
    ret_t0 = []
    ret_t1 = []
    for beat in events['beat']:
        if beat == 1:
            ret_t0.append(np.random.normal(0.03, 0.06))
        else:
            ret_t0.append(np.random.normal(-0.02, 0.06))
        ret_t1.append(np.random.normal(0.001, 0.025))
    events['ret_t0'] = ret_t0
    events['ret_t1'] = ret_t1
    events['ret_2d'] = (1 + events['ret_t0']) * (1 + events['ret_t1']) - 1
    return events


def run_backtest(
    test_features,
    earn_df,
    model_proba,
    long_threshold=0.60,
    short_threshold=0.40,
    max_position_pct=0.05,
    max_loss_pct=0.08,
    starting_nav=1000000.0,
    use_real_prices=False,
    seed=42,
):
    # The T-1/T+1 structure is how I'm measuring signal quality,
    # not describing how to trade it. The right application
    # depends on the strategy — event-driven, continuous monitoring,
    # factor-based, etc. The backtest is there to show that the signal
    # has real predictive power.

    test = test_features.copy()
    test['proba'] = model_proba

    if 'date' in earn_df.columns:
        eid = earn_df[['ticker', 'date', 'beat']].copy()
    else:
        eid = earn_df[['ticker', 'earnings_date', 'beat']].copy()
        eid = eid.rename(columns={'earnings_date': 'date'})
    eid['date'] = pd.to_datetime(eid['date'])

    if 'earnings_date' not in test.columns and 'date' in test.columns:
        test = test.rename(columns={'date': 'earnings_date'})
    test['earnings_date'] = pd.to_datetime(test['earnings_date'])

    if 'beat' not in test.columns:
        test = test.merge(
            eid.rename(columns={'date': 'earnings_date'}),
            on=['ticker', 'earnings_date'], how='left'
        )

    if use_real_prices:
        print('  Pulling real OHLC price data from yfinance...')
        real_returns = []
        for _, row in test.iterrows():
            ret = get_earnings_return(row['ticker'], row['earnings_date'])
            real_returns.append(ret)
        test['ret_2d'] = real_returns

        # Dropping events where price data isn't available.
        # If a lot of events are being dropped here, it likely means the
        # yfinance earnings dates are off and need to be cross-referenced
        # against a second source.

        missing = test['ret_2d'].isna().sum()
        if missing > 0:
            print(f'  {missing} events dropped — price data unavailable from yfinance')
            test = test.dropna(subset=['ret_2d'])
    else:
        test = simulate_earnings_returns(test, seed=seed)

    def size(p):

        # Above 0.60 = long, below 0.40 = short, between 0.4-0.6 = no trade
        # Position scales with how far the probability is from 0.50 —
        # the further away, the higher the model's conviction, so the
        # larger the allocation (up to the 5% cap).

        if p >= long_threshold:
            return max_position_pct * (p - 0.5) / 0.5
        elif p <= short_threshold:
            return -max_position_pct * (0.5 - p) / 0.5
        return 0.0

    positions = []
    for p in test['proba']:
        positions.append(size(p))
    test['position'] = positions

    trades = test[test['position'] != 0].copy()

    if trades.empty:
        return {'trade_log': pd.DataFrame(), 'equity_curve': pd.Series(),
                'metrics': {}}

    trades['gross_ret'] = trades['position'] * trades['ret_2d']

    # applying the stop-loss — if the loss on a trade exceeds 8%,
    # cap it there rather than letting it run further

    net_rets = []
    for i in trades.index:
        gross = trades.loc[i, 'gross_ret']
        floor = -max_loss_pct * abs(trades.loc[i, 'position'])
        net_rets.append(max(gross, floor))
    trades['net_ret'] = net_rets

    # 5bps is the assumed trading cost to get in and out of each position —
    # this is reasonable for liquid names like CMG and MCD, but probably understated
    # for smaller names like WING and JACK where spreads are wider.

    trades['cost'] = trades['position'].abs() * 0.0005
    trades['pnl_pct'] = trades['net_ret'] - trades['cost']

    trades = trades.sort_values('earnings_date').reset_index(drop=True)
    trades['pnl_usd'] = trades['pnl_pct'] * starting_nav
    trades['cum_pnl'] = trades['pnl_usd'].cumsum()
    trades['equity'] = starting_nav + trades['cum_pnl']
    trades['nav_before'] = starting_nav

    equity_curve = trades.set_index('earnings_date')['equity']

    n_trades = len(trades)
    win_rate = (trades['pnl_pct'] > 0).mean()
    avg_win = trades.loc[trades['pnl_pct'] > 0, 'pnl_pct'].mean()
    avg_loss = trades.loc[trades['pnl_pct'] < 0, 'pnl_pct'].mean()

    total_gains = trades.loc[trades['pnl_pct'] > 0, 'pnl_pct'].sum()
    total_losses = abs(trades.loc[trades['pnl_pct'] < 0, 'pnl_pct'].sum())
    if total_losses == 0:
        profit_factor = 0.0
    else:
        profit_factor = total_gains / total_losses

    total_return = (trades['equity'].iloc[-1] / starting_nav) - 1
    mean_ret = trades['pnl_pct'].mean()
    std_ret = trades['pnl_pct'].std() + 0.000000001

    # Sharpe ratio annualised using trading days
    # Each trade is ~2 days, so 252 / 2 = 126 periods per year

    sharpe = (mean_ret / std_ret) * np.sqrt(126)

    # Maximum drawdown calcs

    peak = trades['equity'].cummax()
    dd = (trades['equity'] - peak) / peak
    max_dd = dd.min()

    n_days = max((trades['earnings_date'].max() -
                  trades['earnings_date'].min()).days, 1)
    ann_return = (1 + total_return) ** (252 / n_days) - 1
    if abs(max_dd) == 0:
        calmar = 0.0
    else:
        calmar = ann_return / abs(max_dd)

    return {
        'trade_log': trades,
        'equity_curve': equity_curve,
        'metrics': {
            'n_trades': n_trades,
            'n_long': int((trades['position'] > 0).sum()),
            'n_short': int((trades['position'] < 0).sum()),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'ann_return': ann_return,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'calmar': calmar,
        },
    }


def per_ticker_stats(trade_log):
    tickers = trade_log['ticker'].unique()
    rows = []
    for ticker in tickers:
        group = trade_log[trade_log['ticker'] == ticker]
        rows.append({
            'ticker': ticker,
            'n_trades': len(group),
            'win_rate': (group['pnl_pct'] > 0).mean(),
            'avg_pnl': group['pnl_pct'].mean(),
            'total_pnl': group['pnl_usd'].sum(),
        })
    return pd.DataFrame(rows)


def sweep_probability_thresholds(test_features, earn_df, model_proba):

    # I'm testing a range of probability thresholds around the 0.60/0.40
    # base case to see which cutoff gives the best Sharpe ratio. The range
    # of 0.55 to 0.70 was chosen as a reasonable spread to explore.
    # Tighter thresholds mean fewer trades but higher conviction, and looser
    # ones mean more trades but more noise.

    thresholds_to_test = [
        (0.55, 0.45),
        (0.58, 0.42),
        (0.60, 0.40),
        (0.62, 0.38),
        (0.65, 0.35),
        (0.68, 0.32),
        (0.70, 0.30),
    ]

    results = []
    for long_t, short_t in thresholds_to_test:
        bt = run_backtest(
            test_features, earn_df, model_proba,
            long_threshold=long_t,
            short_threshold=short_t,
        )
        m = bt['metrics']
        if not m:
            continue
        print(f'  threshold {long_t}/{short_t} — '
              f'{m["n_trades"]} trades, '
              f'win rate {m["win_rate"]:.0%}, '
              f'Sharpe {m["sharpe"]:.2f}')
        results.append({
            'long_threshold': long_t,
            'short_threshold': short_t,
            'n_trades': m['n_trades'],
            'win_rate': round(m['win_rate'], 3),
            'sharpe': round(m['sharpe'], 3),
            'ann_return': round(m['ann_return'], 4),
            'max_drawdown': round(m['max_drawdown'], 4),
        })

    df = pd.DataFrame(results).sort_values('sharpe', ascending=False)
    return df.reset_index(drop=True)


def sweep_sentiment_threshold(features):

    # In my original project, the average sentiment before a miss was -0.83 SD,
    # so I'm using -1.5 as a starting point for the miss-flagging screen and
    # testing a range around it. Going lower flags more events, but generates
    # more false alarms. Going higher is more precise but will miss some
    # actual misses where sentiment wasn't that negative.

    thresholds_to_test = [-0.5, -0.75, -1.0, -1.25, -1.5, -1.75, -2.0, -2.25, -2.5]

    sent_col = 'pre30_sent'
    valid = features.dropna(subset=[sent_col, 'beat']).copy()
    actual_misses = (valid['beat'] == 0).sum()
    total_events = len(valid)

    print(f'  {total_events} events, {actual_misses} actual misses '
          f'({actual_misses/total_events:.0%} miss rate)')

    results = []
    for threshold in thresholds_to_test:
        flagged = valid[sent_col] <= threshold
        true_pos = ((flagged) & (valid['beat'] == 0)).sum()
        false_pos = ((flagged) & (valid['beat'] == 1)).sum()
        false_neg = ((~flagged) & (valid['beat'] == 0)).sum()

        n_flagged = flagged.sum()
        if (true_pos + false_pos) > 0:
            precision = true_pos / (true_pos + false_pos)
        else:
            precision = 0
        if actual_misses > 0:
            recall = true_pos / actual_misses
        else:
            recall = 0
        if (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

        print(f'  {threshold} SD — flagged {n_flagged}, '
              f'precision {precision:.0%}, recall {recall:.0%}, f1 {f1:.2f}')

        results.append({
            'threshold_sd': threshold,
            'n_flagged': int(n_flagged),
            'pct_flagged': round(n_flagged / total_events, 3),
            'true_pos': int(true_pos),
            'false_pos': int(false_pos),
            'false_neg': int(false_neg),
            'miss_precision': round(precision, 3),
            'miss_recall': round(recall, 3),
            'miss_f1': round(f1, 3),
            'actual_misses': int(actual_misses),
            'total_events': int(total_events),
        })

    return pd.DataFrame(results)