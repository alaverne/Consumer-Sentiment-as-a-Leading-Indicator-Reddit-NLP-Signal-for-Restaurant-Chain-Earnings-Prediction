import os
import sys
import pandas as pd
import numpy as np

os.makedirs("results/figures", exist_ok=True)


def run(mode="demo"):

    if mode == "live":
        from scraper import scrape_all, load_posts, UNIVERSE
        from feature_engineering import score_posts, aggregate_weekly, get_all_earnings

        raw_posts = load_posts(data_dir="data/raw")
        posts = score_posts(raw_posts)
        weekly = aggregate_weekly(posts)
        tickers = []
        for _, t in UNIVERSE:
            tickers.append(t)
        earn_df = get_all_earnings(tickers, start="2022-01-01", end="2026-04-01")

    else:
        print("  [DEMO MODE] Using synthetic data — run with --mode live for real data")
        from data_generator import generate_reddit_posts, generate_earnings_data
        from feature_engineering import aggregate_weekly

        posts = generate_reddit_posts(12000)
        earn_df = generate_earnings_data()
        weekly = aggregate_weekly(posts)

    print(f"  Posts:           {len(posts):>6,}")
    print(f"  Earnings events: {len(earn_df):>6,}")

    from feature_engineering import build_features, temporal_split

    features = build_features(posts, earn_df, weekly)
    train, test = temporal_split(features, cutoff="2024-01-01")
    features.to_csv("results/features.csv", index=False)

    from models import (cross_validate_models, evaluate_oos,
                        train_magnitude_model, compute_feature_importance,
                        window_comparison)

    print("\n  Cross-Validation (train set):")
    cv = cross_validate_models(train, n_splits=3)
    print(cv[["Model", "CV_AUC_mean", "CV_AUC_std", "CV_Acc_mean"]].to_string(index=False))

    print("\n  Out-of-Sample Evaluation (test set):")
    oos = evaluate_oos(train, test)

    print("\n  Magnitude Regression (EPS surprise %):")
    mag = train_magnitude_model(train, test)
    print(f"     OOS MAE: {mag['metrics']['OOS_MAE']:.3f}%   "
          f"R2: {mag['metrics']['OOS_R2']:.3f}")

    metrics_rows = []
    for name, res in oos.items():
        row = {"Model": name}
        for key, val in res["metrics"].items():
            row[key] = val
        metrics_rows.append(row)
    pd.DataFrame(metrics_rows).to_csv("results/metrics.csv", index=False)

    print("\n  Window Comparison:")
    shift_results = window_comparison(features)
    pooled = shift_results[shift_results["ticker"] == "ALL"]
    if not pooled.empty:
        print(pooled[["window", "corr", "n"]].to_string(index=False))

    from backtest import run_backtest, per_ticker_stats

    best_model = "Gradient Boosting"
    if best_model not in oos:
        for key in oos:
            best_model = key

    proba = oos[best_model]["proba"]
    te_idx = oos[best_model]["te_idx"]
    test_sub = test.loc[te_idx].reset_index(drop=True)

    bt = run_backtest(test_sub, earn_df, proba,
                      use_real_prices=(mode == "live"))
    m = bt["metrics"]

    print(f"\n  Model:        {best_model}")
    print(f"  Trades:       {m['n_trades']}  ({m['n_long']} long / {m['n_short']} short)")
    print(f"  Win Rate:     {m['win_rate']*100:.1f}%")
    print(f"  Ann. Return:  {m['ann_return']*100:.1f}%")
    print(f"  Sharpe:       {m['sharpe']:.2f}")
    print(f"  Max Drawdown: {m['max_drawdown']*100:.1f}%")
    print(f"  Calmar:       {m['calmar']:.2f}")

    if not bt["trade_log"].empty:
        bt["trade_log"].to_csv("results/trade_log.csv", index=False)
        tk = per_ticker_stats(bt["trade_log"])
        print(f"\n  Per-Ticker P&L:\n{tk.to_string(index=False)}")

    from visualizations import (
        plot_sentiment_timeseries, plot_sentiment_distribution,
        plot_confusion_matrix_chart, plot_model_comparison,
        plot_roc_curves, plot_equity_curve, plot_feature_importance,
        plot_window_comparison,
    )

    imp = compute_feature_importance(oos)
    FIG = "results/figures/"

    if "Logistic Regression (Original)" in oos:
        orig_model = oos["Logistic Regression (Original)"]
    else:
        orig_model = list(oos.values())[0]

    plot_sentiment_timeseries(weekly, features, FIG + "chart1_sentiment_timeseries.png")
    print("  saved: chart1_sentiment_timeseries")
    plot_sentiment_distribution(features, FIG + "chart2_sentiment_distribution.png")
    print("  saved: chart2_sentiment_distribution")
    plot_confusion_matrix_chart(orig_model["y_true"], orig_model["pred"], "Logistic Regression (Original)", FIG + "chart3_confusion_matrix.png")
    print("  saved: chart3_confusion_matrix")
    plot_model_comparison(oos, cv, FIG + "chart4_model_comparison.png")
    print("  saved: chart4_model_comparison")
    plot_roc_curves(oos, FIG + "chart5_roc_curves.png")
    print("  saved: chart5_roc_curves")
    plot_equity_curve(bt, FIG + "chart6_equity_curve.png")
    print("  saved: chart6_equity_curve")
    plot_feature_importance(imp, 8, FIG + "chart7_feature_importance.png")
    print("  saved: chart7_feature_importance")
    plot_window_comparison(shift_results, FIG + "chart8_window_comparison.png")
    print("  saved: chart8_window_comparison")

    print("\nDone.")
    print("  results/features.csv  - feature matrix")
    print("  results/metrics.csv   - model metrics")
    print("  results/trade_log.csv - trade-level P&L")
    print("  results/figures/      - 8 charts")


if __name__ == "__main__":
    mode = "demo"
    if len(sys.argv) > 1 and sys.argv[1] == "--mode":
        mode = sys.argv[2]
    run(mode=mode)