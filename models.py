import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, log_loss, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ***LOGISTIC REGRESSION***

# I'm using class_weight='balanced' because most earnings events are beats —
# without it the model just predicts beat every time and learns nothing.
# In a trading context, catching misses is even more important than
# just maximizing raw accuracy, so I'm optimizing for recall here.

# ***UPDATE***

# I kept the original logistic regression as a baseline and added
# Random Forest and Gradient Boosting to test whether non-linear
# models capture interactions between features that logistic regression misses.

# I also added a Ridge regression as an exploratory analysis of the
# relationship between sentiment and EPS surprise magnitude — to see
# whether sentiment strength predicts not just direction but how large
# the beat or miss tends to be. Given the limitations (noted below in
# train_magnitude_model), it isn't used for position sizing in the
# current backtest.


# Feature columns used across all models

FEATURE_COLS = [
    'pre30_sent',
    'pre7_sent',
    'sentiment_shift_30_7',
]

# The original feature from the first version of this project

ORIGINAL_FEATURE = ['pre30_sent']

# ***MODEL SELECTION AND EVALUATION***

# I'm using AUC-ROC as the primary metric to compare models rather than
# accuracy. AUC-ROC measures how well the model actually separates beats
# from misses regardless of where you set the threshold. 0.5 means no better
# than random, 1.0 is perfect.

# I'm including Random Forest even though 3 features and ~60 training events
# is not ideal — I just want to check whether a non-linear relationship exists
# that logistic regression would miss. If all models perform similarly, that's
# actually also useful to know.

# I'm keeping both logistic regression versions so there's a clear
# comparison showing whether adding the 7-day window improved the original model.


TARGET = 'beat'
TARGET_REG = 'eps_surprise_pct'


def prep(df, feature_cols=None):

    # Dropping rows where the primary sentiment feature is missing —
    # if there were no posts in the window, we have no signal and
    # shouldn't be making a prediction for that event

    if feature_cols is None:
        feature_cols = FEATURE_COLS
    cols = [c for c in feature_cols if c in df.columns]
    primary = [c for c in cols if c == 'pre30_sent']
    secondary = [c for c in cols if c != 'pre30_sent']
    sub = df[cols + [TARGET]].copy()
    sub[secondary] = sub[secondary].fillna(0.0)
    sub = sub.dropna(subset=primary + [TARGET])
    X = sub[cols].values
    y = sub[TARGET].values.astype(int)
    y_reg = df.loc[sub.index, TARGET_REG].values.astype(float) \
            if TARGET_REG in df.columns else np.zeros(len(sub))
    return X, y, y_reg, sub.index


def build_pipelines():
    return {

        # Original Logistic Regression model

        'Logistic Regression (Original)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                C=1.0,
                random_state=42,
            )),
        ]),

        # New logistic regression model with added features
        # and stronger regularization to prevent overfitting to noise

        'Logistic Regression (Enhanced)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                class_weight='balanced',
                C=0.5,
                max_iter=1000,
                random_state=42,
            )),
        ]),

        # For random forest, I'm adding a depth cap to prevent memorising
        # the small training set
        # Giving each leaf at least 5 events, given the ~60 training events

        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=300,
                max_depth=5,
                min_samples_leaf=5,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
            )),
        ]),

        # For gradient boosting, I'm using a small learning rate
        # to reduce the risk of overfitting. Training each tree
        # on a random 80% sample of the data.

        'Gradient Boosting': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )),
        ]),
    }


# using 3-fold instead of the default 5-fold — with ~56 training events,
# 5-fold produces folds too small to compute AUC reliably

def cross_validate_models(train, n_splits=3):

    # Standard k-fold assigns folds randomly which could let the model
    # train on future events and validate on past ones, so I'm using
    # TimeSeriesSplit instead.

    tscv = TimeSeriesSplit(n_splits=n_splits)
    pipelines = build_pipelines()
    rows = []
    for name, pipe in pipelines.items():
        feat_cols = ORIGINAL_FEATURE if 'Original' in name else FEATURE_COLS
        X, y, _, _ = prep(train, feature_cols=feat_cols)
        if len(X) < n_splits + 1:
            continue
        auc_scores = cross_val_score(pipe, X, y, cv=tscv, scoring='roc_auc')
        acc_scores = cross_val_score(pipe, X, y, cv=tscv, scoring='accuracy')
        rows.append({
            'Model': name,
            'CV_AUC_mean': auc_scores.mean(),
            'CV_AUC_std': auc_scores.std(),
            'CV_Acc_mean': acc_scores.mean(),
            'CV_Acc_std': acc_scores.std(),
            'N_train': len(X),
        })
    return pd.DataFrame(rows)


def evaluate_oos(train, test):
    pipelines = build_pipelines()
    results = {}
    for name, pipe in pipelines.items():
        feat_cols = ORIGINAL_FEATURE if 'Original' in name else FEATURE_COLS
        X_tr, y_tr, _, _ = prep(train, feature_cols=feat_cols)
        X_te, y_te, _, te_idx = prep(test, feature_cols=feat_cols)
        if len(X_tr) == 0 or len(X_te) == 0:
            continue
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        pred = pipe.predict(X_te)
        metrics = {
            'OOS_AUC': roc_auc_score(y_te, proba),
            'OOS_Accuracy': accuracy_score(y_te, pred),
            'OOS_Precision': precision_score(y_te, pred),
            'OOS_Recall': recall_score(y_te, pred),
            'OOS_F1': f1_score(y_te, pred),
            'OOS_LogLoss': log_loss(y_te, proba),
        }
        results[name] = {
            'metrics': metrics,
            'proba': proba,
            'pred': pred,
            'y_true': y_te,
            'pipe': pipe,
            'features': feat_cols,
            'te_idx': te_idx,
        }
        print(f'\n--- {name} ---')
        print(f'OOS AUC: {metrics["OOS_AUC"]:.3f}  '
              f'Accuracy: {metrics["OOS_Accuracy"]:.1%}  '
              f'F1: {metrics["OOS_F1"]:.3f}')
        print(classification_report(y_te, pred,
              target_names=['Miss', 'Beat']))
    return results


def train_magnitude_model(train, test):

    # Ridge regression predicts EPS surprise magnitude as a continuous number —
    # similar to linear regression, but with a penalty to prevent overfitting on the
    # small dataset.

    # A problem with this is that EPS surprise magnitude varies a lot by ticker.
    # A model across all tickers is not ideal here. I kept this as an exploratory
    # analysis, but it's not being used for position sizing.

    from sklearn.metrics import mean_absolute_error, r2_score
    X_tr, _, y_tr_reg, _ = prep(train)
    X_te, _, y_te_reg, te_idx = prep(test)
    if len(X_tr) == 0 or len(X_te) == 0:
        return {'metrics': {'OOS_MAE': np.nan, 'OOS_R2': np.nan}}

    # I'm using alpha=1.0 to shrink the coefficients and reduce overfitting

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('reg', Ridge(
            alpha=1.0
        ))
    ])
    pipe.fit(X_tr, y_tr_reg)
    pred = pipe.predict(X_te)
    return {
        'metrics': {
            'OOS_MAE': mean_absolute_error(y_te_reg, pred),
            'OOS_R2': r2_score(y_te_reg, pred),
        },
        'pred': pred, 'y_true': y_te_reg, 'pipe': pipe, 'te_idx': te_idx,
    }


def compute_feature_importance(oos_results):
    rows = []
    for model_name, res in oos_results.items():
        feat_cols = res['features']
        pipe = res['pipe']
        clf = pipe.steps[-1][1]

        if model_name in ('Logistic Regression (Original)', 'Logistic Regression (Enhanced)'):
            imp = np.abs(clf.coef_[0])
        elif model_name in ('Random Forest', 'Gradient Boosting'):
            imp = clf.feature_importances_
        else:
            imp = np.zeros(len(feat_cols))

        for i, feat in enumerate(feat_cols):
            rows.append({
                'model': model_name,
                'feature': feat,
                'importance': float(imp[i]) if i < len(imp) else 0.0,
            })
    return pd.DataFrame(rows)


# Seeing how much the sentiment signal changes as earnings approaches

def window_comparison(features):

    rows = []
    windows = {
        '30-day': 'pre30_sent',
        '7-day': 'pre7_sent',
    }
    tickers = list(features['ticker'].unique()) + ['ALL']
    for ticker in tickers:
        if ticker == 'ALL':
            sub = features.copy()
        else:
            sub = features[features['ticker'] == ticker].copy()
        for window_name, col in windows.items():
            if col not in sub.columns:
                continue
            valid = sub.dropna(subset=[col, 'beat'])
            if len(valid) < 5:
                continue
            corr = valid[col].corr(valid['beat'])
            rows.append({
                'ticker': ticker,
                'window': window_name,
                'corr': corr,
                'n': len(valid),
            })
    return pd.DataFrame(rows)