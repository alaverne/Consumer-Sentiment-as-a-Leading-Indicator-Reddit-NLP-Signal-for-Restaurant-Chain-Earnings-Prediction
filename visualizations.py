import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix

TICKER_COLORS = {
    "CMG": "red",
    "MCD": "gold",
    "DPZ": "blue",
    "WEN": "darkred",
    "SHAK": "purple",
    "WING": "orange",
    "JACK": "green",
}

TICKER_NAMES = {
    "CMG": "Chipotle",
    "MCD": "McDonald's",
    "DPZ": "Domino's",
    "WEN": "Wendy's",
    "SHAK": "Shake Shack",
    "WING": "Wingstop",
    "JACK": "Jack in the Box",
}


def plot_sentiment_timeseries(weekly_sent, model_df, save_path=None):
    plt.style.use("seaborn-v0_8-darkgrid")
    tickers = list(TICKER_COLORS.keys())
    fig, axes = plt.subplots(len(tickers), 1, figsize=(14, 18))
    for ax, ticker in zip(axes, tickers):
        sent = weekly_sent[weekly_sent["ticker"] == ticker].copy()
        sent["date"] = pd.to_datetime(sent["date"])
        ax.plot(sent["date"], sent["sent_zscore"],
                color=TICKER_COLORS[ticker], linewidth=1.5)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
        ax.fill_between(sent["date"], sent["sent_zscore"], 0,
                        where=sent["sent_zscore"] > 0, alpha=0.15, color="green")
        ax.fill_between(sent["date"], sent["sent_zscore"], 0,
                        where=sent["sent_zscore"] < 0, alpha=0.15, color="red")
        earn = model_df[model_df["ticker"] == ticker]
        for _, row in earn.iterrows():
            c = "green" if row["beat"] == 1 else "red"
            ls = "-" if row["beat"] == 1 else "--"
            ax.axvline(pd.to_datetime(row.get("date", row.get("earnings_date"))),
                       color=c, alpha=0.6, linewidth=1.5, linestyle=ls)
        ax.set_title(f'{TICKER_NAMES.get(ticker, ticker)} ({ticker})',
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("Sentiment Z-Score", fontsize=9)
        ax.set_xlim(pd.Timestamp("2022-01-01"), pd.Timestamp("2026-04-01"))
    fig.suptitle(
        "Reddit Consumer Sentiment Z-Score vs Earnings Outcomes\n"
        "Green solid lines = Beat  |  Red dashed lines = Miss",
        fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_sentiment_distribution(model_df, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    sent_col = "pre30_sent" if "pre30_sent" in model_df.columns else "pre_earn_sent"
    beats = model_df[model_df["beat"] == 1][sent_col].dropna()
    misses = model_df[model_df["beat"] == 0][sent_col].dropna()
    ax.hist(beats, bins=12, alpha=0.6, color="green",
            label=f"Earnings Beat (n={len(beats)})")
    ax.hist(misses, bins=6, alpha=0.6, color="red",
            label=f"Earnings Miss (n={len(misses)})")
    ax.axvline(beats.mean(), color="darkgreen", linestyle="--",
               linewidth=2, label=f"Beat Mean: {beats.mean():.2f}")
    ax.axvline(misses.mean(), color="darkred", linestyle="--",
               linewidth=2, label=f"Miss Mean: {misses.mean():.2f}")
    ax.set_title("30-Day Pre-Earnings Sentiment: Beats vs Misses",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Average Z-Score Sentiment (30 days pre-earnings)")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_confusion_matrix_chart(y_true, y_pred,
                                model_name="Logistic Regression (Original)",
                                save_path=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Predicted Miss", "Predicted Beat"],
                yticklabels=["Actual Miss", "Actual Beat"], ax=ax)
    ax.set_title(f"Confusion Matrix — {model_name}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_model_comparison(oos_results, cv_results, save_path=None):
    models = list(oos_results.keys())
    oos_auc = [oos_results[m]["metrics"]["OOS_AUC"] for m in models]
    cv_auc = cv_results.set_index("Model")["CV_AUC_mean"].reindex(models).fillna(0).values
    cv_std = cv_results.set_index("Model")["CV_AUC_std"].reindex(models).fillna(0).values
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - 0.2, cv_auc, 0.35, label="CV AUC (train)", alpha=0.75,
           yerr=cv_std, capsize=4)
    ax.bar(x + 0.2, oos_auc, 0.35, label="OOS AUC (test)", alpha=0.9)
    ax.axhline(0.5, color="red", ls="--", lw=1.2, label="Random baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9, rotation=12, ha="right")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Model Comparison: Cross-Val vs Out-of-Sample AUC", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0.3, 1.0)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_roc_curves(oos_results, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, res in oos_results.items():
        fpr, tpr, _ = roc_curve(res["y_true"], res["proba"])
        auc = res["metrics"]["OOS_AUC"]
        ax.plot(fpr, tpr, lw=2.0, label=f"{name} ({auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Random (0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Out-of-Sample", fontsize=12)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_equity_curve(bt_results, save_path=None):
    trades = bt_results.get("trade_log", pd.DataFrame())
    if trades.empty:
        return None
    equity = trades["equity"]
    dates = pd.to_datetime(trades["earnings_date"])
    peak = equity.cummax()
    dd = (equity - peak) / peak * 100
    fig = plt.figure(figsize=(11, 7))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax1.plot(dates, equity / 1e6, color="steelblue", lw=1.8,
             marker="o", markersize=4)
    ax1.fill_between(dates, equity / 1e6, equity.iloc[0] / 1e6,
                     where=(equity >= equity.iloc[0]), color="green", alpha=0.12)
    ax1.fill_between(dates, equity / 1e6, equity.iloc[0] / 1e6,
                     where=(equity < equity.iloc[0]), color="red", alpha=0.15)
    m = bt_results["metrics"]
    stats = (f"Sharpe: {m['sharpe']:.2f}  |  Ann. Return: {m['ann_return']*100:.1f}%  |"
             f"  Max DD: {m['max_drawdown']*100:.1f}%  |  Win Rate: {m['win_rate']*100:.0f}%"
             f"  |  Trades: {m['n_trades']}")
    ax1.text(0.01, 0.97, stats, transform=ax1.transAxes, fontsize=8, va="top")
    ax1.set_ylabel("NAV ($M)")
    ax1.set_title("Sentiment Signal P&L — Out-of-Sample Backtest", fontsize=12)
    ax2.fill_between(dates, dd, 0, color="red", alpha=0.5)
    ax2.plot(dates, dd, color="red", lw=0.9)
    ax2.set_ylabel("Drawdown %")
    plt.setp(ax1.get_xticklabels(), visible=False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_feature_importance(importance_df, top_n=8, save_path=None):
    models = importance_df["model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 6))
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        sub = (importance_df[importance_df["model"] == model]
               .sort_values("importance", ascending=False).head(top_n))
        ax.barh(sub["feature"], sub["importance"], alpha=0.85)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title(model, fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Feature Importance by Model", fontsize=12, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_window_comparison(shift_df, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    pivot = shift_df[shift_df["ticker"] != "ALL"].pivot(
        index="ticker", columns="window", values="corr")
    col_order = ["30-day", "7-day"]
    pivot = pivot.reindex(columns=[c for c in col_order if c in pivot.columns])
    sns.heatmap(pivot, ax=axes[0], cmap="RdYlGn", center=0,
                annot=True, fmt=".2f", linewidths=0.4, annot_kws={"size": 9})
    axes[0].set_title("Sentiment→Beat Correlation by Window", fontsize=11)
    pooled = shift_df[shift_df["ticker"] == "ALL"].copy()
    if not pooled.empty:
        axes[1].plot(range(len(pooled)), pooled["corr"].values,
                     color="steelblue", lw=2.2, marker="o", markersize=7)
        axes[1].fill_between(range(len(pooled)), pooled["corr"].values,
                              color="steelblue", alpha=0.12)
        axes[1].axhline(0, color="gray", ls="--", lw=0.8)
        axes[1].set_xticks(range(len(pooled)))
        axes[1].set_xticklabels(pooled["window"].tolist(), fontsize=9)
        axes[1].set_ylabel("Correlation with Beat")
        axes[1].set_title("Pooled Window Comparison (All Tickers)", fontsize=11)
    fig.suptitle("Window Comparison: Which Window Predicts Best?", fontsize=12, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path