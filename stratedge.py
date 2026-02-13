"""
DePrado Dollar Pipeline
========================
Alpha Factory industrielle — López de Prado, AFML (2018).

Pipeline ETL : Dollar Bars, Triple Barrière, Différentiation Fractionnaire,
Meta-Labeling, Sequential Bootstrap, Purged CV, CSCV PBO Gate, SADF,
Entropy, Bet Sizing, Backtest net de coûts.

Stratégies : Momentum, Trend Following (SMA 63), Mean Reversion, Breakout, Bollinger Bands.
Sélection walk-forward 50/50 OOS par Sharpe × (0.5 + WinRate), corrigée DSR.
Retrain-every-20, CSCV Gate (PBO check avant trading).

Usage:
    python pipe.py                          # Prompt ticker, auto-sélection stratégie
    python pipe.py --fetch SPY              # autre ticker
    python pipe.py --strategy momentum      # forcer Momentum
    python pipe.py --strategy trend         # forcer Trend Following (SMA 63)
    python pipe.py --strategy mr            # forcer Mean Reversion
    python pipe.py --strategy bo            # forcer Breakout
    python pipe.py --strategy bb            # forcer Bollinger Bands
    python pipe.py --strategy-select        # compare les 5 stratégies + lance le meilleur
    python pipe.py data.csv --research      # mode recherche (CPCV, MDA, rapports)
    python pipe.py --optimize --n-iter 100  # random search paramètres
    python pipe.py --daily                  # mode cron incrémental

Gaëtan Music — 2025
"""

import os
import sys
import json
import shutil
import logging
import argparse
import warnings
import re
from typing import Optional, Tuple, List, Dict, Any
from collections import Counter
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, norm, rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("DePrado")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TF_PRESETS = {
    "h4": {
        "horizon": 6, "momentum_lookback": 12,
        "entropy_window": 20, "sadf_min_window": 8,
        "spread_pct": 0.0004, "swap_daily_pct": 0.0001,
        "label": "H4 Intraday Swing",
    },
    "d1": {
        "horizon": 10, "momentum_lookback": 20,
        "entropy_window": 30, "sadf_min_window": 10,
        "spread_pct": 0.0004, "swap_daily_pct": 0.00015,
        "label": "D1 Daily Swing",
    },
    "w1": {
        "horizon": 8, "momentum_lookback": 8,
        "entropy_window": 15, "sadf_min_window": 6,
        "spread_pct": 0.0004, "swap_daily_pct": 0.0002,
        "label": "W1 Position/Regime",
    },
}

FEATURE_COLS = [
    "Close_FracDiff", "sadf",
    "entropy_shannon", "entropy_plugin", "entropy_lz",
    "mom_ret", "mom_side", "vol",
    "signal_strength",
    "bb_pct_b", "bb_bandwidth",
]

STRATEGY_PRESETS = {
    "momentum": {
        "upper_width": 1.0, "lower_width": 1.0,
        "horizon_mult": 1.0,
        "label": "Momentum",
    },
    "trend": {
        "upper_width": 1.0, "lower_width": 1.0,
        "horizon_mult": 1.0,
        "label": "Trend Following",
    },
    "mr": {
        "upper_width": 0.5, "lower_width": 0.5,
        "horizon_mult": 0.6,
        "label": "Mean Reversion",
    },
    "bo": {
        "upper_width": 1.5, "lower_width": 0.8,
        "horizon_mult": 1.2,
        "label": "Breakout",
    },
    "bb": {
        "upper_width": 0.6, "lower_width": 0.6,
        "horizon_mult": 0.7,
        "label": "Bollinger Bands",
    },
}

MIN_TRADES_STRATEGY = 20
RETRAIN_EVERY = 20

TICKER_MAP = {
    "GLD": "GLD",     "SPY": "SPY",      "AAPL": "AAPL",
    "XAUUSD": "GC=F", "XAGUSD": "SI=F",  "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X", "USDJPY": "JPY=X",
    "SPX": "^GSPC",   "NDX": "^IXIC",    "DXY": "DX-Y.NYB",
    "BTC": "BTC-USD",  "ETH": "ETH-USD",
    "OIL": "CL=F",    "NATGAS": "NG=F",
}

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]
VOL_WARMUP = 20


# ═══════════════════════════════════════════════════════════════════════════
# Ch.2 — Dollar Bars
# ═══════════════════════════════════════════════════════════════════════════

def dollar_bars(opens, highs, lows, closes, volumes, threshold):
    """Agrège les barres OHLCV en dollar bars (AFML Ch.2)."""
    n = len(closes)
    idx = np.empty(n, dtype=np.int64)
    o = np.empty(n, dtype=np.float64)
    h = np.empty(n, dtype=np.float64)
    l = np.empty(n, dtype=np.float64)
    c = np.empty(n, dtype=np.float64)
    v = np.empty(n, dtype=np.float64)

    cum_dv = 0.0
    cur_h, cur_l, cur_v = -1e9, 1e9, 0.0
    cur_o = opens[0]
    count = 0

    for i in range(n):
        dv = (highs[i] + lows[i] + closes[i]) / 3.0 * volumes[i]
        cum_dv += dv
        cur_h = max(cur_h, highs[i])
        cur_l = min(cur_l, lows[i])
        cur_v += volumes[i]

        if cum_dv >= threshold:
            idx[count] = i
            o[count], h[count], l[count] = cur_o, cur_h, cur_l
            c[count], v[count] = closes[i], cur_v
            count += 1
            cum_dv, cur_h, cur_l, cur_v = 0.0, -1e9, 1e9, 0.0
            if i < n - 1:
                cur_o = opens[i + 1]

    return idx[:count], o[:count], h[:count], l[:count], c[:count], v[:count], cum_dv


# ═══════════════════════════════════════════════════════════════════════════
# Ch.2.5 — Filtre CUSUM
# ═══════════════════════════════════════════════════════════════════════════

def cusum_filter(prices, threshold):
    """Détecte les ruptures structurelles par cumul symétrique (AFML Ch.2)."""
    log_ret = np.log(prices / prices.shift(1)).dropna()
    events = []
    s_pos, s_neg = 0.0, 0.0
    for t, r in log_ret.items():
        s_pos = max(0.0, s_pos + r)
        s_neg = min(0.0, s_neg + r)
        if s_pos >= threshold or s_neg <= -threshold:
            events.append(t)
            s_pos, s_neg = 0.0, 0.0
    return pd.DatetimeIndex(events)


# ═══════════════════════════════════════════════════════════════════════════
# Signal primaire — Momentum
# ═══════════════════════════════════════════════════════════════════════════

def momentum_signal(close_vals, idx, lookback=20):
    """Calcule le signal momentum à `lookback` périodes."""
    if idx < lookback:
        return None
    ret = close_vals[idx] / close_vals[idx - lookback] - 1.0
    side = 1 if ret >= 0 else -1
    return {"side": side, "mom_side": side, "mom_ret": ret, "signal_strength": abs(ret)}


def trend_signal(close_vals, idx, lookback=20, sma_period=63):
    """Signal Trend Following : prix vs SMA(63)."""
    if idx < sma_period:
        return None
    sma = np.mean(close_vals[idx - sma_period + 1 : idx + 1])
    price = close_vals[idx]
    distance = (price - sma) / sma
    if abs(distance) < 1e-6:
        return None
    side = 1 if price > sma else -1
    ret = close_vals[idx] / close_vals[idx - lookback] - 1.0 if idx >= lookback else 0.0
    return {"side": side, "mom_side": side, "mom_ret": ret, "signal_strength": abs(distance)}


def mean_reversion_signal(close_vals, idx, lookback=20):
    """Signal mean-reversion : z-score du prix vs SMA, side inversé."""
    if idx < lookback:
        return None
    window = close_vals[idx - lookback : idx + 1]
    ma = np.mean(window)
    std = np.std(window)
    if std < 1e-10:
        return None
    zscore = (close_vals[idx] - ma) / std
    if abs(zscore) < 0.5:
        return None
    side = -1 if zscore > 0 else 1
    ret = close_vals[idx] / close_vals[idx - lookback] - 1.0
    return {"side": side, "mom_side": side, "mom_ret": ret, "signal_strength": abs(zscore)}


def breakout_signal(close_vals, idx, lookback=20):
    """Signal breakout : cassure du canal Donchian."""
    if idx < lookback:
        return None
    window = close_vals[idx - lookback : idx]
    upper = np.max(window)
    lower = np.min(window)
    span = upper - lower
    if span < 1e-10:
        return None
    price = close_vals[idx]
    if price > upper:
        side = 1
        strength = (price - upper) / span
    elif price < lower:
        side = -1
        strength = (lower - price) / span
    else:
        return None
    ret = close_vals[idx] / close_vals[idx - lookback] - 1.0
    return {"side": side, "mom_side": side, "mom_ret": ret, "signal_strength": strength}


def bollinger_signal(close_vals, idx, lookback=20, n_std=2.0):
    """Signal Bollinger Bands : touche de bande → mean reversion."""
    if idx < lookback:
        return None
    window = close_vals[idx - lookback : idx + 1]
    ma = np.mean(window)
    std = np.std(window)
    if std < 1e-10:
        return None
    upper = ma + n_std * std
    lower = ma - n_std * std
    price = close_vals[idx]
    if price >= upper:
        side = -1
        strength = (price - upper) / (n_std * std)
    elif price <= lower:
        side = 1
        strength = (lower - price) / (n_std * std)
    else:
        return None
    ret = close_vals[idx] / close_vals[idx - lookback] - 1.0
    return {"side": side, "mom_side": side, "mom_ret": ret, "signal_strength": abs(strength)}


SIGNAL_FN_MAP = {
    "momentum": momentum_signal,
    "trend": trend_signal,
    "mr": mean_reversion_signal,
    "bo": breakout_signal,
    "bb": bollinger_signal,
}


# ═══════════════════════════════════════════════════════════════════════════
# Ch.3 — Triple Barrière + Meta-Labeling
# ═══════════════════════════════════════════════════════════════════════════

def triple_barrier(close, events, horizon, upper_w, lower_w,
                   vol_lookback=20, mom_lookback=20, signal_fn=None):
    """Labeling par barrières symétriques calibrées en volatilité (AFML Ch.3)."""
    if signal_fn is None:
        signal_fn = momentum_signal
    ret = close.pct_change()
    vol = ret.rolling(window=vol_lookback).std()
    vals = close.values
    ts = close.index
    ts_map = {t: i for i, t in enumerate(ts)}
    records = []

    for t0 in events:
        if t0 not in ts_map:
            continue
        i0 = ts_map[t0]
        p0 = vals[i0]
        v = vol.iloc[i0] if i0 >= vol_lookback else np.nan
        if np.isnan(v) or v <= 1e-8:
            continue

        sig = signal_fn(vals, i0, mom_lookback)
        if sig is None:
            continue

        up = p0 * (1.0 + v * upper_w)
        lo = p0 * (1.0 - v * lower_w)
        end = min(i0 + horizon, len(vals) - 1)

        touch_ret = vals[end] / p0 - 1.0
        label = 1 if touch_ret >= 0 else -1  # AFML Ch.3.2 — timeout = sign(ret)
        touch_i = end
        bars_held = end - i0

        for j in range(i0 + 1, end + 1):
            if vals[j] >= up:
                label, touch_i = 1, j
                touch_ret = vals[j] / p0 - 1.0
                bars_held = j - i0
                break
            if vals[j] <= lo:
                label, touch_i = -1, j
                touch_ret = vals[j] / p0 - 1.0
                bars_held = j - i0
                break

        rec = {
            "t0": t0, "t1": ts[touch_i], "ret": touch_ret, "Label": label,
            "price": p0, "vol": v, "stop_loss": lo, "take_profit": up,
            "bars_held": bars_held,
        }
        rec.update(sig)
        records.append(rec)

    df = pd.DataFrame(records)
    if df.empty:
        return df
    df = df.set_index("t0")
    df.index.name = "Date"
    return df


def meta_label(events_df):
    """Le meta-label indique si le signal primaire est correct (AFML Ch.3.6)."""
    df = events_df.copy()
    df["meta_label"] = (df["side"] * df["ret"] > 0).astype(int)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Ch.4 — Sample Weights
# ═══════════════════════════════════════════════════════════════════════════

def indicator_matrix(events_df, close_idx):
    ts_map = {t: i for i, t in enumerate(close_idx)}
    mat = np.zeros((len(close_idx), len(events_df)), dtype=np.float64)
    for j, (t0, row) in enumerate(events_df.iterrows()):
        i0, i1 = ts_map.get(t0), ts_map.get(row["t1"])
        if i0 is not None and i1 is not None:
            mat[i0 : i1 + 1, j] = 1.0
    return pd.DataFrame(mat, index=close_idx, columns=events_df.index)


def avg_uniqueness(ind_mat):
    conc = ind_mat.sum(axis=1)
    conc[conc == 0] = 1.0
    return (ind_mat.div(conc, axis=0).sum(axis=0) / ind_mat.sum(axis=0)).fillna(1.0)


def sample_weights(events_df, close):
    """Pondération par unicité × amplitude du rendement (AFML Ch.4)."""
    ind = indicator_matrix(events_df, close.index)
    w = avg_uniqueness(ind) * events_df["ret"].abs()
    return w / w.sum()


def seq_bootstrap(ind_mat, n_samples=None, rng=None):
    """Sequential Bootstrap — échantillonnage IID (AFML Ch.4.5)."""
    if rng is None:
        rng = np.random.RandomState(42)
    mat = ind_mat.values if hasattr(ind_mat, "values") else ind_mat
    n_bars, n_total = mat.shape
    if n_samples is None:
        n_samples = n_total
    cum_conc = np.zeros(n_bars, dtype=np.float64)
    selected = []
    for _ in range(n_samples):
        avg_u = np.zeros(n_total)
        for j in range(n_total):
            col_j = mat[:, j]
            active = col_j > 0
            if not active.any():
                continue
            conc = cum_conc[active] + col_j[active]
            avg_u[j] = (col_j[active] / conc).mean()
        total = avg_u.sum()
        p = avg_u / total if total > 0 else np.ones(n_total) / n_total
        chosen = rng.choice(n_total, p=p)
        selected.append(chosen)
        cum_conc += mat[:, chosen]
    return np.array(selected)


# ═══════════════════════════════════════════════════════════════════════════
# Ch.5 — Fractional Differentiation (FFD)
# ═══════════════════════════════════════════════════════════════════════════

def frac_diff_ffd(series, d, threshold=1e-4):
    """Stationnarise la série tout en préservant la mémoire longue (AFML Ch.5)."""
    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
        k += 1
    weights = np.array(weights[::-1])
    win = len(weights)
    if len(series) < win:
        win = max(2, len(series) // 2)
        weights = weights[-win:]
    res = np.convolve(series.values, weights, mode="valid")
    return pd.Series(res, index=series.index[win - 1 :], name="Close_FracDiff")


# ═══════════════════════════════════════════════════════════════════════════
# Ch.7 — Purged K-Fold CV
# ═══════════════════════════════════════════════════════════════════════════

class PurgedKFoldCV:
    def __init__(self, n_splits=5, embargo_pct=0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, events_df, close_idx):
        n = len(events_df)
        indices = np.arange(n)
        fold_sz = n // self.n_splits
        embargo = max(1, int(n * self.embargo_pct))
        t1 = events_df["t1"]

        for i in range(self.n_splits):
            ts = i * fold_sz
            te = (i + 1) * fold_sz if i < self.n_splits - 1 else n
            test_idx = indices[ts:te]
            t0_min = events_df.index[ts]
            t1_max = t1.iloc[ts:te].max()

            mask = np.ones(n, dtype=bool)
            mask[ts:te] = False
            for j in range(n):
                if mask[j] and t1.iloc[j] >= t0_min and events_df.index[j] <= t1_max:
                    mask[j] = False
            mask[te : min(te + embargo, n)] = False
            yield indices[mask], test_idx


# ═══════════════════════════════════════════════════════════════════════════
# Ch.11 — Combinatorial Purged CV
# ═══════════════════════════════════════════════════════════════════════════

class CombPurgedCV:
    """C(N, k) chemins de backtest avec purge + embargo (AFML Ch.11)."""

    def __init__(self, n_splits=6, n_test=2, embargo_pct=0.01):
        self.n_splits = n_splits
        self.n_test = n_test
        self.embargo_pct = embargo_pct

    @property
    def n_paths(self):
        return comb(self.n_splits, self.n_test)

    def split(self, events_df, close_idx=None):
        n = len(events_df)
        indices = np.arange(n)
        fold_sz = n // self.n_splits
        embargo = max(1, int(n * self.embargo_pct))
        t1 = events_df["t1"]

        folds = []
        for i in range(self.n_splits):
            s = i * fold_sz
            e = (i + 1) * fold_sz if i < self.n_splits - 1 else n
            folds.append(indices[s:e])

        for combo in combinations(range(self.n_splits), self.n_test):
            test_idx = np.sort(np.concatenate([folds[i] for i in combo]))
            train_folds = [i for i in range(self.n_splits) if i not in combo]
            train_idx = np.sort(np.concatenate([folds[i] for i in train_folds]))

            keep = np.ones(len(train_idx), dtype=bool)
            for fi in combo:
                f = folds[fi]
                t0_min = events_df.index[f[0]]
                t1_max = t1.iloc[f[0] : f[-1] + 1].max()
                end = f[-1]
                for k, ti in enumerate(train_idx):
                    if not keep[k]:
                        continue
                    if t1.iloc[ti] >= t0_min and events_df.index[ti] <= t1_max:
                        keep[k] = False
                    elif end < ti <= end + embargo:
                        keep[k] = False

            if keep.sum() > 0:
                yield train_idx[keep], test_idx


def cscv_pbo(X, y, w, cv_splits, n_splits=6):
    """Combinatorial Symmetric CV — Probability of Backtest Overfitting (AFML Ch.11).

    Retourne PBO (proportion OOS ≤ 0), logit(w_bar) et rank correlation IS↔OOS.
    """
    is_sharpes, oos_sharpes = [], []
    for tr, te in cv_splits:
        if len(tr) < 10 or len(te) < 5:
            continue
        wn = w[tr] / w[tr].sum() * len(w[tr])
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=5, max_features=1,
            min_samples_leaf=max(1, len(tr) // 20),
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        try:
            clf.fit(X[tr], y[tr], sample_weight=wn)
            p_is = clf.predict_proba(X[tr])
            p1_is = p_is[:, 1] if p_is.shape[1] == 2 else p_is[:, 0]
            bs_is = np.abs(bet_size(p1_is))
            is_sr = bs_is.mean() / bs_is.std() if bs_is.std() > 1e-10 else 0.0

            p_oos = clf.predict_proba(X[te])
            p1_oos = p_oos[:, 1] if p_oos.shape[1] == 2 else p_oos[:, 0]
            bs_oos = np.abs(bet_size(p1_oos))
            oos_sr = bs_oos.mean() / bs_oos.std() if bs_oos.std() > 1e-10 else 0.0

            is_sharpes.append(is_sr)
            oos_sharpes.append(oos_sr)
        except Exception:
            continue

    n = len(is_sharpes)
    if n < 3:
        return {"pbo": 0.0, "logit_w_bar": 0.0, "rank_corr": 0.0, "n_paths": n,
                "oos_sharpe_mean": 0.0, "oos_sharpe_std": 0.0}

    is_arr = np.array(is_sharpes)
    oos_arr = np.array(oos_sharpes)

    is_ranks = rankdata(is_arr)
    oos_ranks = rankdata(oos_arr)

    pbo = float((oos_arr <= 0).sum() / n)

    best_is_idx = np.argmax(is_arr)
    w_bar = oos_ranks[best_is_idx] / (n + 1)
    w_bar = np.clip(w_bar, 1e-6, 1.0 - 1e-6)
    logit_w = float(np.log(w_bar / (1.0 - w_bar)))

    rank_corr = float(np.corrcoef(is_ranks, oos_ranks)[0, 1]) if n > 2 else 0.0

    return {
        "pbo": round(pbo, 4),
        "logit_w_bar": round(logit_w, 4),
        "rank_corr": round(rank_corr, 4),
        "n_paths": n,
        "oos_sharpe_mean": round(float(oos_arr.mean()), 4),
        "oos_sharpe_std": round(float(oos_arr.std()), 4),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Ch.8 — MDA Feature Importance
# ═══════════════════════════════════════════════════════════════════════════

def mda_importance(X, y, w, feat_names, cv_splits, n_repeats=3):
    """Mean Decrease Accuracy par permutation (AFML Ch.8)."""
    imp = {f: [] for f in feat_names}
    rng = np.random.RandomState(42)

    for tr, te in cv_splits:
        if len(tr) < 10 or len(te) < 5:
            continue
        w_n = w[tr] / w[tr].sum() * len(w[tr])
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=5, max_features=1,
            min_samples_leaf=max(1, len(tr) // 20),
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        try:
            clf.fit(X[tr], y[tr], sample_weight=w_n)
        except Exception:
            continue
        base = accuracy_score(y[te], clf.predict(X[te]))

        for j, f in enumerate(feat_names):
            scores = []
            for _ in range(n_repeats):
                Xp = X[te].copy()
                rng.shuffle(Xp[:, j])
                scores.append(accuracy_score(y[te], clf.predict(Xp)))
            imp[f].append(base - np.mean(scores))

    return {
        f: {"mean": round(float(np.mean(v)), 6), "std": round(float(np.std(v)), 6)}
        for f, v in imp.items() if v
    }


# ═══════════════════════════════════════════════════════════════════════════
# Ch.10 — Bet Sizing
# ═══════════════════════════════════════════════════════════════════════════

def bet_size(prob):
    """Taille de position par transformation CDF (AFML Ch.10)."""
    p = np.clip(prob, 1e-6, 1.0 - 1e-6)
    z = (p - 0.5) / (p * (1 - p)) ** 0.5
    return np.clip(2 * norm.cdf(z) - 1, -1, 1)


# ═══════════════════════════════════════════════════════════════════════════
# Ch.14 — Deflated Sharpe Ratio
# ═══════════════════════════════════════════════════════════════════════════

def deflated_sharpe(observed_sr, n_trials, n_obs, skew=0.0, kurt=3.0):
    """Probabilité que le Sharpe observé dépasse le max attendu sous H0 (AFML Ch.14)."""
    if n_trials <= 1 or n_obs <= 1:
        return 1.0
    gamma = 0.5772156649
    e_max = ((1 - gamma) * norm.ppf(1.0 - 1.0 / n_trials)
             + gamma * norm.ppf(1.0 - 1.0 / (n_trials * np.e)))
    e_max *= np.sqrt(1.0 / max(n_obs - 1, 1))
    se = np.sqrt(
        (1 - skew * observed_sr + (kurt - 1) / 4 * observed_sr ** 2)
        / max(n_obs - 1, 1)
    )
    if se < 1e-10:
        return 1.0 if observed_sr > e_max else 0.0
    return round(float(norm.cdf((observed_sr - e_max) / se)), 4)


# ═══════════════════════════════════════════════════════════════════════════
# Ch.17 — SADF (Structural Breaks)
# ═══════════════════════════════════════════════════════════════════════════

def _adf_stat(log_p):
    dy = np.diff(log_p)
    y_lag = log_p[:-1]
    n = len(dy)
    if n < 3:
        return 0.0
    X = np.column_stack([np.ones(n), y_lag])
    try:
        beta = np.linalg.lstsq(X, dy, rcond=None)[0]
        res = dy - X @ beta
        s2 = np.sum(res ** 2) / (n - 2)
        return beta[1] / np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    except Exception:
        return 0.0


def sadf(log_prices, min_window=10):
    """Supreme ADF — détection de bulles (AFML Ch.17)."""
    vals = log_prices.values
    n = len(vals)
    result = pd.Series(np.nan, index=log_prices.index)
    for t in range(min_window, n):
        best = -np.inf
        for t0 in range(0, t - min_window + 1):  # AFML Ch.17 — full history
            s = _adf_stat(vals[t0 : t + 1])
            if s > best:
                best = s
        result.iloc[t] = best
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Ch.18 — Entropy Features
# ═══════════════════════════════════════════════════════════════════════════

def _quantize(series, n_bins=10):
    bins = np.linspace(series.min() - 1e-10, series.max() + 1e-10, n_bins + 1)
    return np.digitize(series.values, bins) - 1


def shannon_entropy(series, n_bins=10):
    _, counts = np.unique(_quantize(series, n_bins), return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def plugin_entropy(series, n_bins=10, window=2):
    q = _quantize(series, n_bins)
    n = len(q)
    if n < window:
        return 0.0
    ngrams = [tuple(q[i : i + window]) for i in range(n - window + 1)]
    counts = Counter(ngrams)
    total = sum(counts.values())
    p = np.array(list(counts.values())) / total
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def lempel_ziv(series, n_bins=2):
    s = "".join(map(str, _quantize(series, n_bins)))
    n = len(s)
    if n == 0:
        return 0.0
    i, c, l = 0, 1, 1
    while i + l <= n:
        if s[i : i + l] in s[: i + l - 1] and i + l < n:
            l += 1
        else:
            c += 1
            i += l
            l = 1
    return c / (n / np.log2(max(n, 2))) if n > 1 else 0.0


def entropy_features(close, window=30):
    """Trois mesures d'entropie sur les log-rendements (AFML Ch.18)."""
    ret = np.log(close / close.shift(1)).dropna()
    sh, pi, lz, idx = [], [], [], []
    for i in range(window, len(ret)):
        chunk = ret.iloc[i - window : i]
        sh.append(shannon_entropy(chunk))
        pi.append(plugin_entropy(chunk))
        lz.append(lempel_ziv(chunk))
        idx.append(ret.index[i])
    return pd.DataFrame(
        {"entropy_shannon": sh, "entropy_plugin": pi, "entropy_lz": lz},
        index=pd.DatetimeIndex(idx),
    )


def bollinger_features(close, window=20, n_std=2.0):
    """Bollinger Bands %B et Bandwidth."""
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    pct_b = (close - lower) / (upper - lower)
    bandwidth = (upper - lower) / ma
    return pd.DataFrame(
        {"bb_pct_b": pct_b, "bb_bandwidth": bandwidth},
        index=close.index,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Backtest Engine
# ═══════════════════════════════════════════════════════════════════════════

class Backtest:
    """Simulation P&L avec spread, slippage et swap overnight."""

    def __init__(self, spread=0.0004, swap=0.00015, slippage=0.0001, bars_per_day=1.0):
        self.spread = spread
        self.swap = swap
        self.slippage = slippage
        self.bpd = bars_per_day

    def run(self, signals_df, capital=100_000.0):
        equity = [capital]
        trades = []

        for _, sig in signals_df.iterrows():
            if sig["side"] == 0:
                equity.append(equity[-1])
                continue

            days = max(1, sig.get("bars_held", 5) / self.bpd)
            gross = sig["side"] * sig["realized_ret"]
            cost = self.spread + self.slippage * 2 + self.swap * days
            net = (gross - cost) * sig["bet_size"]
            pnl = capital * net
            capital += pnl
            equity.append(capital)

            trades.append({
                "date": sig["date"], "side": sig["side"],
                "bet_size": sig["bet_size"],
                "gross_ret": round(gross, 6), "costs": round(cost, 6),
                "net_ret": round(net, 6), "pnl": round(pnl, 2),
                "equity": round(capital, 2), "days_held": round(days, 1),
            })

        equity = np.array(equity)
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        return {"equity": equity, "trades": trades_df, "stats": self._stats(equity, trades_df, equity[0])}

    def _stats(self, eq, tdf, cap0):
        if tdf.empty:
            return {"total_return_pct": 0, "sharpe": 0, "max_drawdown_pct": 0, "win_rate": 0}

        total = eq[-1] / cap0 - 1
        rets = tdf["net_ret"].values
        active = rets[rets != 0]

        if len(active) > 1 and active.std() > 0:
            tpy = max(1, len(active) / max(1, len(tdf)))
            sharpe = active.mean() / active.std() * np.sqrt(252 * tpy)
        else:
            sharpe = 0.0

        peak = np.maximum.accumulate(eq)
        dd = ((eq - peak) / peak).min()
        wr = (active > 0).sum() / len(active) if len(active) > 0 else 0
        gp = active[active > 0].sum()
        gl = abs(active[active < 0].sum())
        pf = gp / gl if gl > 0 else float("inf")
        aw = active[active > 0].mean() if (active > 0).any() else 0
        al = active[active < 0].mean() if (active < 0).any() else 0
        calmar = total / abs(dd) if dd != 0 else 0
        sk = float(pd.Series(active).skew()) if len(active) > 2 else 0.0
        ku = float(pd.Series(active).kurtosis()) + 3.0 if len(active) > 3 else 3.0

        return {
            "total_return_pct": round(total * 100, 2),
            "final_equity": round(eq[-1], 2),
            "sharpe": round(sharpe, 4),
            "max_drawdown_pct": round(dd * 100, 2),
            "calmar": round(calmar, 4),
            "win_rate": round(wr, 4),
            "profit_factor": round(pf, 4),
            "total_trades": len(active),
            "avg_win_pct": round(aw * 100, 4),
            "avg_loss_pct": round(al * 100, 4),
            "avg_days_held": round(tdf["days_held"].mean(), 1) if "days_held" in tdf else 0,
            "total_costs_pct": round(tdf["costs"].sum() * 100, 4) if "costs" in tdf else 0,
            "ret_skew": round(sk, 4),
            "ret_kurt": round(ku, 4),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Dashboard HTML
# ═══════════════════════════════════════════════════════════════════════════

def build_dashboard(signals_df, bt_result, meta, path):
    stats = bt_result["stats"]
    trades = bt_result["trades"]
    eq = bt_result["equity"]

    if not trades.empty:
        dates_js = json.dumps(trades["date"].tolist())
        eq_js = json.dumps([round(e, 2) for e in eq[1 : len(trades) + 1]])
        pnl_js = json.dumps(trades["pnl"].tolist())
    else:
        dates_js = eq_js = pnl_js = "[]"

    peak = np.maximum.accumulate(eq)
    dd_raw = ((eq - peak) / peak * 100).tolist()
    dd_js = json.dumps([round(d, 2) for d in dd_raw[1 : len(trades) + 1]] if not trades.empty else [])

    rows = ""
    if not trades.empty:
        for _, t in trades.tail(50).iterrows():
            sc = "long" if t["side"] == 1 else "short"
            sl = "LONG" if t["side"] == 1 else "SHORT"
            pc = "pos" if t["pnl"] >= 0 else "neg"
            rows += (
                f'<tr><td>{t["date"]}</td><td class="{sc}">{sl}</td>'
                f'<td>{t["bet_size"]:.2%}</td><td>{t["gross_ret"]:.4%}</td>'
                f'<td>{t["costs"]:.4%}</td><td class="{pc}">{t["net_ret"]:.4%}</td>'
                f'<td class="{pc}">${t["pnl"]:,.0f}</td><td>{t["days_held"]:.0f}d</td></tr>\n'
            )

    latest = signals_df.iloc[-1] if len(signals_df) > 0 else None
    lhtml = ""
    if latest is not None:
        sc = "long" if latest.get("side", 0) == 1 else ("short" if latest.get("side", 0) == -1 else "flat")
        lhtml = f"""
<div class="signal-card {sc}">
<h3>DERNIER SIGNAL — {latest.get('date','N/A')}</h3>
<div class="signal-grid">
  <div><span class="label">Direction</span><span class="value {sc}">{latest.get('side_label','FLAT')}</span></div>
  <div><span class="label">Confiance</span><span class="value">{latest.get('confidence',0):.1%}</span></div>
  <div><span class="label">Taille</span><span class="value">{latest.get('bet_size',0):.1%}</span></div>
  <div><span class="label">Prix</span><span class="value">{latest.get('price',0):,.2f}</span></div>
  <div><span class="label">Stop Loss</span><span class="value">{latest.get('stop_loss',0):,.2f}</span></div>
  <div><span class="label">Take Profit</span><span class="value">{latest.get('take_profit',0):,.2f}</span></div>
  <div><span class="label">Régime</span><span class="value">{latest.get('regime','N/A')}</span></div>
  <div><span class="label">Stratégie</span><span class="value">{meta.get('strategy','N/A').upper()}</span></div>
  <div><span class="label">Momentum</span><span class="value">{int(latest.get('mom_side',0))}</span></div>
</div></div>"""

    rp = stats.get("total_return_pct", 0)
    html = f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>DePrado Dollar Pipeline — Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root{{--bg:#0f1117;--card:#1a1d28;--border:#2a2d3a;--text:#e1e4ea;--green:#00d68f;--red:#ff4d6a;--blue:#4d8cff;--dim:#6b7185}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:var(--bg);color:var(--text);font-family:'SF Mono','Fira Code',monospace;font-size:13px;padding:20px}}
h1{{font-size:18px;margin-bottom:20px;color:var(--blue)}}
h2{{font-size:14px;margin-bottom:12px;color:var(--dim);text-transform:uppercase;letter-spacing:1px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:24px}}
.stat-card{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:14px}}
.stat-card .label{{display:block;color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.5px}}
.stat-card .value{{display:block;font-size:20px;font-weight:bold;margin-top:4px}}
.pos{{color:var(--green)}}.neg{{color:var(--red)}}
.long{{color:var(--green)}}.short{{color:var(--red)}}.flat{{color:var(--dim)}}
.chart-container{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px;margin-bottom:24px}}
.chart-row{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px}}
canvas{{width:100%!important}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{text-align:left;padding:8px;border-bottom:2px solid var(--border);color:var(--dim);text-transform:uppercase;font-size:10px}}
td{{padding:6px 8px;border-bottom:1px solid var(--border)}}
tr:hover{{background:rgba(77,140,255,.05)}}
.signal-card{{background:var(--card);border:2px solid var(--border);border-radius:8px;padding:20px;margin-bottom:24px}}
.signal-card.long{{border-color:var(--green)}}.signal-card.short{{border-color:var(--red)}}
.signal-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:12px}}
.signal-grid .label{{display:block;color:var(--dim);font-size:10px;text-transform:uppercase}}
.signal-grid .value{{display:block;font-size:16px;font-weight:bold;margin-top:2px}}
.meta-row{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:24px}}
.meta-card{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:12px;font-size:11px}}
.meta-card span{{color:var(--dim)}}
@media(max-width:768px){{.chart-row{{grid-template-columns:1fr}}.signal-grid{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body>
<h1>DEPRADO DOLLAR PIPELINE — DASHBOARD</h1>
{lhtml}
<h2>Performance</h2>
<div class="grid">
<div class="stat-card"><span class="label">Return</span><span class="value {'pos' if rp>=0 else 'neg'}">{rp:+.2f}%</span></div>
<div class="stat-card"><span class="label">Sharpe</span><span class="value">{stats.get('sharpe',0):.2f}</span></div>
<div class="stat-card"><span class="label">Max DD</span><span class="value neg">{stats.get('max_drawdown_pct',0):.2f}%</span></div>
<div class="stat-card"><span class="label">Win Rate</span><span class="value">{stats.get('win_rate',0):.1%}</span></div>
<div class="stat-card"><span class="label">Profit Factor</span><span class="value">{stats.get('profit_factor',0):.2f}</span></div>
<div class="stat-card"><span class="label">Calmar</span><span class="value">{stats.get('calmar',0):.2f}</span></div>
<div class="stat-card"><span class="label">Trades</span><span class="value">{stats.get('total_trades',0)}</span></div>
<div class="stat-card"><span class="label">Avg Hold</span><span class="value">{stats.get('avg_days_held',0):.0f}d</span></div>
</div>
<h2>Equity &amp; Drawdown</h2>
<div class="chart-row">
<div class="chart-container"><canvas id="eqC"></canvas></div>
<div class="chart-container"><canvas id="ddC"></canvas></div>
</div>
<h2>P&amp;L par trade</h2>
<div class="chart-container"><canvas id="plC"></canvas></div>
<h2>50 derniers trades</h2>
<div class="chart-container" style="overflow-x:auto">
<table><thead><tr><th>Date</th><th>Side</th><th>Size</th><th>Gross</th><th>Costs</th><th>Net</th><th>P&amp;L</th><th>Hold</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<h2>Configuration</h2>
<div class="meta-row">
<div class="meta-card"><span>Source :</span> {meta.get('source_rows','?')} barres<br>
<span>{"Time Bars" if meta.get('raw_bars') else "Dollar Bars"} :</span> {meta.get('actual_bars','?')}<br>
<span>Stratégie :</span> {meta.get('strategy','auto').upper()}<br>
<span>CUSUM :</span> {meta.get('cusum_events','?')} events</div>
<div class="meta-card"><span>Horizon :</span> {meta.get('horizon','?')} bars<br>
<span>Frac d :</span> {meta.get('frac_d','?')}<br>
<span>Samples :</span> {meta.get('final_samples','?')}</div>
<div class="meta-card"><span>Momentum :</span> {meta.get('momentum_lookback','?')}j<br>
<span>Upper :</span> {meta.get('upper_width','?')}<br>
<span>Lower :</span> {meta.get('lower_width','?')}</div>
</div>
<script>
const D={dates_js},E={eq_js},DD={dd_js},P={pnl_js};
const O={{responsive:true,plugins:{{legend:{{display:false}}}},scales:{{x:{{ticks:{{color:'#6b7185',maxTicksLimit:10}},grid:{{color:'#2a2d3a'}}}},y:{{ticks:{{color:'#6b7185'}},grid:{{color:'#2a2d3a'}}}}}}}};
new Chart(document.getElementById('eqC'),{{type:'line',data:{{labels:D,datasets:[{{data:E,borderColor:'#4d8cff',borderWidth:1.5,pointRadius:0,fill:true,backgroundColor:'rgba(77,140,255,.08)'}}]}},options:{{...O,plugins:{{...O.plugins,title:{{display:true,text:'Equity ($)',color:'#6b7185'}}}}}}}});
new Chart(document.getElementById('ddC'),{{type:'line',data:{{labels:D,datasets:[{{data:DD,borderColor:'#ff4d6a',borderWidth:1.5,pointRadius:0,fill:true,backgroundColor:'rgba(255,77,106,.1)'}}]}},options:{{...O,plugins:{{...O.plugins,title:{{display:true,text:'Drawdown (%)',color:'#6b7185'}}}}}}}});
new Chart(document.getElementById('plC'),{{type:'bar',data:{{labels:D,datasets:[{{data:P,backgroundColor:P.map(v=>v>=0?'rgba(0,214,143,.6)':'rgba(255,77,106,.6)'),borderWidth:0}}]}},options:{{...O,plugins:{{...O.plugins,title:{{display:true,text:'P&L ($)',color:'#6b7185'}}}}}}}});
</script></body></html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    log.info(f"Dashboard : {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline principal
# ═══════════════════════════════════════════════════════════════════════════

class Pipeline:
    def __init__(self, tf="d1", multiplier=None, frac_d=0.4, horizon=None,
                 upper_width=1.0, lower_width=1.0, target_bars=None, cusum_pct=None,
                 n_splits=5, embargo_pct=0.01, min_train=50, momentum_lookback=None,
                 spread_pct=None, swap_daily_pct=None, raw_bars=False, strategy=None, **kw):

        preset = TF_PRESETS.get(tf, TF_PRESETS["d1"])

        self.tf = tf
        self.multiplier = multiplier
        self.frac_d = frac_d
        self.base_horizon = horizon or preset["horizon"]
        self.base_upper_w = upper_width
        self.base_lower_w = lower_width
        self.target_bars = target_bars
        self.cusum_pct = cusum_pct
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.min_train = min_train
        self.mom_lb = momentum_lookback or preset["momentum_lookback"]
        self.ent_win = preset["entropy_window"]
        self.sadf_min = preset["sadf_min_window"]
        self.spread = spread_pct or preset["spread_pct"]
        self.swap = swap_daily_pct or preset["swap_daily_pct"]
        self.raw_bars = raw_bars
        self.bpd = {"h4": 6, "d1": 1, "w1": 0.2}.get(tf, 1)
        self.strategy = strategy

        self._apply_strategy(strategy)

        assert multiplier is None or multiplier > 0, "multiplier doit être > 0"
        assert 0 < frac_d < 1, "frac_d doit être dans (0, 1)"
        assert self.horizon >= 1, "horizon doit être >= 1"

        strat_label = STRATEGY_PRESETS[self.strategy]["label"] if self.strategy else "Auto"
        log.info(f"Config : {preset['label']} | H={self.horizon} Mom={self.mom_lb} | Stratégie={strat_label}")

    def _apply_strategy(self, strategy):
        """Applique les paramètres de barrière selon la stratégie."""
        if strategy and strategy in STRATEGY_PRESETS:
            sp = STRATEGY_PRESETS[strategy]
            self.horizon = max(1, int(self.base_horizon * sp["horizon_mult"]))
            self.upper_w = self.base_upper_w * sp["upper_width"]
            self.lower_w = self.base_lower_w * sp["lower_width"]
            self.signal_fn = SIGNAL_FN_MAP[strategy]
        else:
            self.horizon = self.base_horizon
            self.upper_w = self.base_upper_w
            self.lower_w = self.base_lower_w
            self.signal_fn = momentum_signal

    # --- Strategy Selection (walk-forward + DSR) ------------------------------

    def _eval_strategy_oos(self, filepath, strategy, train_ratio=0.5):
        """Évalue une stratégie sur la portion OOS uniquement (walk-forward selection).

        Le dataset est splitté 50/50 : on entraîne le ML sur [0..split] et on évalue
        les signaux uniquement sur [split..end]. Cela élimine le look-ahead bias
        dans le choix de stratégie.
        """
        prev_strat = self.strategy
        prev_h, prev_uw, prev_lw, prev_fn = self.horizon, self.upper_w, self.lower_w, self.signal_fn
        self.strategy = strategy
        self._apply_strategy(strategy)

        prev_level = log.level
        log.setLevel(logging.WARNING)
        try:
            core = self._build(filepath)
            if not core:
                return None
            feat = core["features"]
            fc = [c for c in FEATURE_COLS if c in feat.columns]
            if not fc or len(feat) < self.min_train + 1:
                return None

            signals = self._walk_forward(feat, fc, silent=True, ind_mat=core.get("ind_mat"))
            if not signals:
                return None
            sdf = pd.DataFrame(signals)

            split_idx = int(len(sdf) * train_ratio)
            if split_idx >= len(sdf) - 1:
                return None
            sdf_oos = sdf.iloc[split_idx:].reset_index(drop=True)

            bt = Backtest(self.spread, self.swap, 0.0001, self.bpd)
            res = bt.run(sdf_oos)
            st = res["stats"]
            st["active_count"] = len(sdf_oos[sdf_oos["side"] != 0])
            st["signal_count"] = len(sdf_oos)
            st["total_signals_all"] = len(sdf)
            st["oos_start_idx"] = split_idx
            return st
        except Exception:
            return None
        finally:
            log.setLevel(prev_level)
            self.strategy = prev_strat
            self.horizon, self.upper_w, self.lower_w, self.signal_fn = prev_h, prev_uw, prev_lw, prev_fn

    @staticmethod
    def _strategy_score(stats):
        """Score combiné Sharpe × (0.5 + WinRate). Min trades = MIN_TRADES_STRATEGY."""
        if not stats:
            return -999.0
        sharpe = stats.get("sharpe", 0)
        wr = stats.get("win_rate", 0)
        trades = stats.get("total_trades", 0)
        if trades < MIN_TRADES_STRATEGY:
            return -999.0
        return sharpe * (0.5 + wr)

    @staticmethod
    def _strategy_dsr(stats, n_trials=5):
        """Deflated Sharpe Ratio appliqué à la sélection de stratégie."""
        if not stats:
            return 0.0
        return deflated_sharpe(
            stats.get("sharpe", 0),
            n_trials,
            max(stats.get("total_trades", 1), 1),
            stats.get("ret_skew", 0),
            stats.get("ret_kurt", 3),
        )

    def _select_strategy(self, filepath):
        """Évalue les 5 stratégies en walk-forward OOS et retourne la meilleure."""
        log.info("=" * 60)
        log.info("STRATEGY SELECTION — Walk-Forward OOS (50/50 split)")
        log.info(f"Min trades requis : {MIN_TRADES_STRATEGY}")
        log.info("=" * 60)

        results = {}
        for strat in STRATEGY_PRESETS:
            log.info(f"  Évaluation : {STRATEGY_PRESETS[strat]['label']}...")
            st = self._eval_strategy_oos(filepath, strat)
            score = self._strategy_score(st)
            dsr = self._strategy_dsr(st)
            results[strat] = {"stats": st, "score": score, "dsr": dsr}
            if st:
                trades = st.get("total_trades", 0)
                flag = ""
                if trades < MIN_TRADES_STRATEGY:
                    flag = f" [REJETÉ: {trades}t < {MIN_TRADES_STRATEGY}]"
                log.info(f"    SR={st['sharpe']:.2f} WR={st['win_rate']:.1%} "
                         f"Ret={st['total_return_pct']:+.1f}% DD={st['max_drawdown_pct']:.1f}% "
                         f"T={trades} DSR={dsr:.4f} → Score={score:.4f}{flag}")
            else:
                log.info(f"    ÉCHEC (pas assez de signaux)")

        viable = {k: v for k, v in results.items() if v["score"] > -999}
        if not viable:
            log.warning(f"  AUCUNE stratégie viable (toutes < {MIN_TRADES_STRATEGY} trades OOS). Fallback → Momentum.")
            best = "momentum"
        else:
            best = max(viable, key=lambda k: viable[k]["score"])

        best_dsr = results[best]["dsr"]
        sig_tag = "SIGNIFICATIF" if best_dsr > 0.95 else "NON SIGNIFICATIF"
        log.info(f"  SÉLECTION : {STRATEGY_PRESETS[best]['label']} "
                 f"(score={results[best]['score']:.4f}, DSR={best_dsr:.4f} — {sig_tag})")
        if best_dsr <= 0.95:
            log.warning(f"  ⚠ DSR={best_dsr:.4f} < 0.95 — le Sharpe observé n'est pas "
                        f"statistiquement significatif après correction pour {len(STRATEGY_PRESETS)} essais.")
        log.info("=" * 60)
        return best, results

    def run_strategy_select(self, filepath):
        """Mode dédié : compare les 5 stratégies, affiche le rapport, lance le signal."""
        best, results = self._select_strategy(filepath)

        out = os.path.dirname(os.path.abspath(filepath))
        report = {
            "selected": best,
            "selection_method": "walk_forward_oos_50_50",
            "min_trades": MIN_TRADES_STRATEGY,
            "scores": {k: {"score": round(v["score"], 4),
                           "dsr": round(v["dsr"], 4),
                           "stats": v["stats"]} for k, v in results.items()},
        }
        with open(os.path.join(out, "strategy_comparison.json"), "w") as f:
            json.dump(report, f, indent=4, default=str)
        log.info(f"Rapport : {os.path.join(out, 'strategy_comparison.json')}")

        self.strategy = best
        self._apply_strategy(best)
        self.run_signal(filepath)

    # --- Core ETL ----------------------------------------------------------

    def _build(self, path):
        raw = self._load(path)
        log.info(f"Données : {len(raw)} barres")

        if self.raw_bars:
            bars = raw.copy()
            thresh, avg_dv = 0, 0
            log.info(f"Mode time bars : {len(bars)} barres")
        else:
            total_dv = ((raw["High"] + raw["Low"] + raw["Close"]) / 3.0 * raw["Volume"]).sum()
            avg_dv = total_dv / len(raw)
            if self.target_bars:
                thresh = total_dv / self.target_bars
            elif self.multiplier:
                thresh = avg_dv * self.multiplier
            else:
                thresh = total_dv / len(raw)
                log.info(f"Threshold auto (1:1 → {len(raw)} bars)")
            bars = self._make_dollar_bars(raw, thresh)
            log.info(f"Dollar Bars : {len(bars)}")

        close = bars["Close"]

        if self.cusum_pct:
            ch = self.cusum_pct
        else:
            ch = np.log(close / close.shift(1)).dropna().std()
            log.info(f"CUSUM auto : {ch:.6f}")

        evts = cusum_filter(close, ch)
        log.info(f"CUSUM : {len(evts)} events ({len(evts)/len(bars)*100:.1f}%)")

        if len(evts) < self.horizon + VOL_WARMUP:
            log.error(f"Pas assez d'events ({len(evts)})")
            return None

        log.info(f"Triple Barrière (H={self.horizon} Mom={self.mom_lb} Strat={self.strategy or 'trend'})...")
        edf = triple_barrier(close, evts, self.horizon, self.upper_w, self.lower_w,
                             VOL_WARMUP, self.mom_lb, self.signal_fn)
        if edf.empty:
            log.error("Aucun event labellisé")
            return None
        log.info(f"Events labellisés : {len(edf)}")

        ok = (edf["mom_side"] * edf["ret"] > 0).sum()
        log.info(f"  Momentum : {ok}/{len(edf)} ({ok/len(edf):.1%})")

        edf = meta_label(edf)
        log.info("Sample weights...")
        edf["sample_weight"] = sample_weights(edf, close)

        log.info(f"FFD (d={self.frac_d})...")
        ffd = frac_diff_ffd(close, self.frac_d, 1e-4)

        log.info(f"Entropy (w={self.ent_win})...")
        ent = entropy_features(close, self.ent_win)

        log.info(f"SADF (min={self.sadf_min})...")
        sadf_s = sadf(np.log(close), self.sadf_min)

        feat = edf[["ret", "side", "Label", "meta_label", "sample_weight",
                     "mom_side", "mom_ret", "signal_strength", "price", "vol",
                     "stop_loss", "take_profit", "t1", "bars_held"]].copy()
        feat["Close_FracDiff"] = ffd
        feat["sadf"] = sadf_s
        for col in ent.columns:
            feat[col] = ent[col]

        bb = bollinger_features(close)
        for col in bb.columns:
            feat[col] = bb[col]

        feat = feat.dropna()
        log.info(f"Matrice : {len(feat)} samples × {len(feat.columns)} colonnes")

        ind_mat = indicator_matrix(edf, close.index)

        return {
            "raw": raw, "bars": bars, "close": close,
            "events_df": edf, "features": feat,
            "threshold": thresh, "avg_dv": avg_dv, "cusum_h": ch,
            "sadf_series": sadf_s, "events": evts,
            "ind_mat": ind_mat,
        }

    # --- Signal (walk-forward) ---------------------------------------------

    def _walk_forward(self, features, feat_cols, silent=False, ind_mat=None):
        X = features[feat_cols].values
        y = features["meta_label"].values
        w = features["sample_weight"].values
        sides = features["side"].values
        n = len(features)
        signals = []
        clf = None
        last_train_i = -1

        for i in range(self.min_train, n):
            need_retrain = (clf is None or (i - last_train_i) >= RETRAIN_EVERY)
            if need_retrain:
                X_tr, y_tr, w_tr = X[:i], y[:i], w[:i]

                if ind_mat is not None and ind_mat.shape[1] >= i:
                    try:
                        sub_mat = ind_mat.iloc[:, :i]
                        boot_idx = seq_bootstrap(sub_mat, n_samples=i)
                        X_tr = X_tr[boot_idx]
                        y_tr = y_tr[boot_idx]
                        w_tr = w_tr[boot_idx]
                    except Exception:
                        pass

                wn = w_tr / w_tr.sum() * len(w_tr)
                clf = RandomForestClassifier(
                    n_estimators=100, max_depth=5, max_features=1,
                    min_samples_leaf=max(1, len(w_tr) // 20),
                    class_weight="balanced", random_state=42, n_jobs=-1,
                )
                try:
                    clf.fit(X_tr, y_tr, sample_weight=wn)
                    last_train_i = i
                except Exception:
                    clf = None
                    continue

            try:
                prob = clf.predict_proba(X[i : i + 1])
                p1 = prob[0, 1] if prob.shape[1] == 2 else prob[0, 0]
                pred = clf.predict(X[i : i + 1])[0]
            except Exception:
                continue

            sz = float(bet_size(np.array([p1]))[0])
            row = features.iloc[i]
            ps = int(sides[i])
            sv = float(row.get("sadf", 0))

            fs = 0 if pred == 0 else ps
            fsz = 0.0 if pred == 0 else abs(sz)

            dt = features.index[i]
            signals.append({
                "date": str(dt.date()) if hasattr(dt, "date") else str(dt),
                "side": fs, "side_label": {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(fs, "FLAT"),
                "confidence": round(float(p1), 4), "bet_size": round(fsz, 4),
                "meta_pred": int(pred), "meta_actual": int(y[i]),
                "mom_side": int(row["mom_side"]),
                "price": round(float(row["price"]), 2),
                "stop_loss": round(float(row["stop_loss"]), 2),
                "take_profit": round(float(row["take_profit"]), 2),
                "vol": round(float(row["vol"]), 6),
                "regime": "BUBBLE" if sv > 0 else "NORMAL",
                "sadf": round(sv, 4),
                "realized_ret": round(float(row["ret"]), 6),
                "label": int(row["Label"]),
                "bars_held": int(row["bars_held"]),
                "train_size": i,
            })
        return signals

    # --- Modes publics -----------------------------------------------------

    def run_signal(self, filepath):
        out = os.path.dirname(os.path.abspath(filepath))

        if self.strategy is None:
            best, results = self._select_strategy(filepath)
            self.strategy = best
            self._apply_strategy(best)

            report = {
                "selected": best,
                "selection_method": "walk_forward_oos_50_50",
                "min_trades": MIN_TRADES_STRATEGY,
                "scores": {k: {"score": round(v["score"], 4),
                               "dsr": round(v["dsr"], 4),
                               "stats": v["stats"]} for k, v in results.items()},
            }
            with open(os.path.join(out, "strategy_comparison.json"), "w") as f:
                json.dump(report, f, indent=4, default=str)

        core = self._build(filepath)
        if not core:
            return
        feat = core["features"]
        close = core["close"]
        edf = core["events_df"]

        fc = [c for c in FEATURE_COLS if c in feat.columns]
        if not fc:
            log.error("Aucune feature disponible")
            return
        n = len(feat)
        if n < self.min_train + 1:
            log.error(f"Pas assez de samples ({n})")
            return

        # MDA pruning
        log.info(f"MDA pruning ({self.min_train} premiers samples)...")
        edf_mda = edf.loc[feat.index[: self.min_train]]
        Xm = feat[fc].values[: self.min_train]
        ym = feat["meta_label"].values[: self.min_train]
        wm = feat["sample_weight"].values[: self.min_train]
        cv = PurgedKFoldCV(min(5, max(2, self.min_train // 10)), self.embargo_pct)
        splits = list(cv.split(edf_mda, close.index))
        if splits:
            mda = mda_importance(Xm, ym, wm, fc, splits)
            kept = [f for f in fc if f in mda and mda[f]["mean"] > 0]
            if kept and len(kept) >= 3:
                fc = kept
                log.info(f"  Retenu : {len(fc)} features")
            else:
                log.info("  MDA non concluant — toutes les features conservées")

        # CSCV Gate — Probability of Backtest Overfitting (AFML Ch.11)
        pbo_metrics = {"pbo": 0.0}
        bet_scale = 1.0
        X_all = feat[fc].values
        y_all = feat["meta_label"].values
        w_all = feat["sample_weight"].values
        edf_cv = edf.loc[feat.index]
        cpcv = CombPurgedCV(6, 2, self.embargo_pct)
        cpcv_splits = list(cpcv.split(edf_cv, close.index))
        if len(cpcv_splits) >= 3:
            pbo_metrics = cscv_pbo(X_all, y_all, w_all, cpcv_splits)
            pbo = pbo_metrics["pbo"]
            if pbo > 0.75:
                bet_scale = 0.0
                log.warning(f"  CSCV GATE: PBO={pbo:.1%} > 75% → bet_scale=0 (BLOCKED)")
            elif pbo > 0.50:
                bet_scale = 0.5
                log.warning(f"  CSCV GATE: PBO={pbo:.1%} > 50% → bet_scale=0.5")
            else:
                log.info(f"  CSCV GATE: PBO={pbo:.1%} ≤ 50% → bet_scale=1.0 (PASS)")
            log.info(f"  Rank corr IS↔OOS: {pbo_metrics['rank_corr']:.4f}")
            log.info(f"  OOS Sharpe: {pbo_metrics['oos_sharpe_mean']:.4f} ± {pbo_metrics['oos_sharpe_std']:.4f}")

        log.info(f"Walk-forward ({self.min_train} min, {n} total, {len(fc)} features)...")
        signals = self._walk_forward(feat, fc, ind_mat=core.get("ind_mat"))
        if not signals:
            log.error("Aucun signal généré")
            return

        # Appliquer bet_scale CSCV
        if bet_scale < 1.0:
            for sig in signals:
                sig["bet_size"] = round(sig["bet_size"] * bet_scale, 4)

        sdf = pd.DataFrame(signals)
        log.info(f"Signaux : {len(sdf)}")

        active = sdf[sdf["side"] != 0]
        hit = 0.0
        if len(active) > 0:
            hit = (active["meta_pred"] == active["meta_actual"]).sum() / len(active)
            log.info(f"Actifs : {len(active)}/{len(sdf)} ({len(active)/len(sdf):.1%})")
            log.info(f"Hit rate : {hit:.4f}")

        bt = Backtest(self.spread, self.swap, 0.0001, self.bpd)
        res = bt.run(sdf)
        st = res["stats"]

        log.info("--- BACKTEST ---")
        log.info(f"  Return  : {st['total_return_pct']:+.2f}%")
        log.info(f"  Sharpe  : {st['sharpe']:.4f}")
        log.info(f"  Max DD  : {st['max_drawdown_pct']:.2f}%")
        log.info(f"  Win Rate: {st['win_rate']:.1%}")
        log.info(f"  PF      : {st['profit_factor']:.2f}")
        log.info(f"  Calmar  : {st['calmar']:.4f}")
        log.info(f"  Trades  : {st['total_trades']}")
        log.info(f"  Hold    : {st['avg_days_held']:.0f}j")
        log.info(f"  Coûts   : {st['total_costs_pct']:.4f}%")

        last = signals[-1]
        log.info(f"--- SIGNAL : {last['side_label']} @ {last['price']} "
                 f"(conf={last['confidence']:.1%}, bet={last['bet_size']:.1%}) ---")

        # Exports
        with open(os.path.join(out, "signal_latest.json"), "w") as f:
            json.dump(last, f, indent=4)
        sdf.to_csv(os.path.join(out, "signals_history.csv"), index=False)
        if not res["trades"].empty:
            res["trades"].to_csv(os.path.join(out, "backtest_trades.csv"), index=False)

        meta = self._meta(core, feat)
        meta.update(st)
        meta["signal_count"] = len(sdf)
        meta["active_count"] = int(len(active))
        meta["hit_rate"] = round(float(hit), 4)
        meta["kept_features"] = fc
        meta["pbo"] = pbo_metrics.get("pbo", 0.0)
        meta["pbo_logit_w_bar"] = pbo_metrics.get("logit_w_bar", 0.0)
        meta["pbo_rank_corr"] = pbo_metrics.get("rank_corr", 0.0)
        meta["bet_scale"] = bet_scale
        with open(os.path.join(out, "signal_meta.json"), "w") as f:
            json.dump(meta, f, indent=4, default=str)

        build_dashboard(sdf, res, meta, os.path.join(out, "dashboard.html"))
        log.info(f"Signal  : {os.path.join(out, 'signal_latest.json')}")
        log.info(f"History : {os.path.join(out, 'signals_history.csv')}")
        log.info(f"Dashboard: {os.path.join(out, 'dashboard.html')}")

    def run_eval(self, filepath):
        """Walk-forward silencieux pour l'optimiseur. Retourne les stats ou None."""
        prev = log.level
        log.setLevel(logging.WARNING)
        try:
            core = self._build(filepath)
            if not core:
                return None
            feat = core["features"]
            fc = [c for c in FEATURE_COLS if c in feat.columns]
            if not fc or len(feat) < self.min_train + 1:
                return None
            signals = self._walk_forward(feat, fc, silent=True, ind_mat=core.get("ind_mat"))
            if not signals:
                return None
            sdf = pd.DataFrame(signals)
            bt = Backtest(self.spread, self.swap, 0.0001, self.bpd)
            res = bt.run(sdf)
            st = res["stats"]
            st["active_count"] = len(sdf[sdf["side"] != 0])
            st["signal_count"] = len(sdf)
            return st
        except Exception:
            return None
        finally:
            log.setLevel(prev)

    def run_research(self, filepath):
        out = os.path.dirname(os.path.abspath(filepath))
        core = self._build(filepath)
        if not core:
            return
        feat = core["features"]
        edf = core["events_df"]
        close = core["close"]

        if len(feat) < self.n_splits * 2:
            log.error(f"Pas assez de samples ({len(feat)})")
            return

        kept = self._cv(feat, edf, close)
        self._report(core["raw"], core["bars"], edf, feat)

        fn = os.path.join(out, f"PROCESSED_{os.path.basename(filepath)}")
        feat.to_csv(fn)
        meta = self._meta(core, feat)
        meta["kept_features"] = kept
        with open(fn.replace(".csv", "_meta.json"), "w") as f:
            json.dump(meta, f, indent=4, default=str)
        log.info(f"Export : {fn} ({len(feat)} samples)")

    def run_daily(self, filepath):
        out = os.path.dirname(os.path.abspath(filepath))
        hist_fn = os.path.join(out, "signals_history.csv")
        latest_fn = os.path.join(out, "signal_latest.json")

        # Charger stratégie depuis le meta précédent
        meta_fn = os.path.join(out, "signal_meta.json")
        if self.strategy is None and os.path.exists(meta_fn):
            try:
                sm = json.load(open(meta_fn))
                saved_strat = sm.get("strategy")
                if saved_strat and saved_strat in STRATEGY_PRESETS:
                    self.strategy = saved_strat
                    self._apply_strategy(saved_strat)
                    log.info(f"Stratégie chargée depuis meta : {saved_strat}")
            except Exception:
                pass

        if self.strategy is None:
            best, _ = self._select_strategy(filepath)
            self.strategy = best
            self._apply_strategy(best)

        last_date = None
        if os.path.exists(hist_fn):
            hist = pd.read_csv(hist_fn)
            if len(hist) > 0:
                last_date = hist["date"].iloc[-1]
                log.info(f"Dernier signal : {last_date}")

        core = self._build(filepath)
        if not core:
            return
        feat = core["features"]

        if last_date is not None:
            new = feat[feat.index > pd.Timestamp(last_date)]
            if new.empty:
                log.info("Aucun nouvel event CUSUM. Rien à faire.")
                return
            log.info(f"Nouveaux events depuis {last_date} : {len(new)}")
        else:
            new = feat
            log.info(f"Pas d'historique. Traitement de {len(new)} events.")

        fc = [c for c in FEATURE_COLS if c in feat.columns]
        if not fc:
            log.error("Aucune feature")
            return

        meta_fn = os.path.join(out, "signal_meta.json")
        if os.path.exists(meta_fn):
            try:
                sm = json.load(open(meta_fn))
                if "kept_features" in sm:
                    sc = [c for c in sm["kept_features"] if c in feat.columns]
                    if sc and len(sc) >= 3:
                        fc = sc
            except Exception:
                pass

        X = feat[fc].values
        y = feat["meta_label"].values
        w = feat["sample_weight"].values
        sides = feat["side"].values
        n = len(feat)

        start = feat.index.get_loc(new.index[0])
        if isinstance(start, slice):
            start = start.start
        start = max(start, self.min_train)

        new_sigs = []
        for i in range(start, n):
            if feat.index[i] not in new.index:
                continue
            wn = w[:i] / w[:i].sum() * i
            clf = RandomForestClassifier(
                n_estimators=100, max_depth=5, max_features=1,
                min_samples_leaf=max(1, i // 20),
                class_weight="balanced", random_state=42, n_jobs=-1,
            )
            try:
                clf.fit(X[:i], y[:i], sample_weight=wn)
                prob = clf.predict_proba(X[i : i + 1])
                p1 = prob[0, 1] if prob.shape[1] == 2 else prob[0, 0]
                pred = clf.predict(X[i : i + 1])[0]
            except Exception:
                continue

            sz = float(bet_size(np.array([p1]))[0])
            row = feat.iloc[i]
            ps = int(sides[i])
            sv = float(row.get("sadf", 0))
            fs = 0 if pred == 0 else ps
            fsz = 0.0 if pred == 0 else abs(sz)
            dt = feat.index[i]

            sig = {
                "date": str(dt.date()) if hasattr(dt, "date") else str(dt),
                "side": fs, "side_label": {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(fs, "FLAT"),
                "confidence": round(float(p1), 4), "bet_size": round(fsz, 4),
                "meta_pred": int(pred), "meta_actual": int(y[i]),
                "mom_side": int(row["mom_side"]),
                "price": round(float(row["price"]), 2),
                "stop_loss": round(float(row["stop_loss"]), 2),
                "take_profit": round(float(row["take_profit"]), 2),
                "vol": round(float(row["vol"]), 6),
                "regime": "BUBBLE" if sv > 0 else "NORMAL",
                "sadf": round(sv, 4),
                "realized_ret": round(float(row["ret"]), 6),
                "label": int(row["Label"]),
                "bars_held": int(row["bars_held"]),
                "train_size": i,
            }
            new_sigs.append(sig)
            log.info(f"  NEW: {sig['date']} {sig['side_label']} conf={sig['confidence']:.1%}")

        if not new_sigs:
            log.info("Aucun nouveau signal.")
            return

        ndf = pd.DataFrame(new_sigs)
        combined = pd.concat([pd.read_csv(hist_fn), ndf], ignore_index=True) if os.path.exists(hist_fn) else ndf
        combined.to_csv(hist_fn, index=False)
        log.info(f"Historique mis à jour : {len(combined)} signaux")

        with open(latest_fn, "w") as f:
            json.dump(new_sigs[-1], f, indent=4)
        last = new_sigs[-1]
        log.info(f"=> {last['side_label']} @ {last['price']} (conf={last['confidence']:.1%})")

    # --- Interne -----------------------------------------------------------

    def _cv(self, feat, edf, close):
        fc = [c for c in FEATURE_COLS if c in feat.columns]
        if not fc:
            return fc
        X, y, w = feat[fc].values, feat["meta_label"].values, feat["sample_weight"].values
        edf_cv = edf.loc[feat.index]

        cpcv = CombPurgedCV(6, 2, self.embargo_pct)
        log.info(f"CPCV 6/2 ({cpcv.n_paths} chemins)...")
        splits = list(cpcv.split(edf_cv, close.index))
        scores, bets = [], []
        for tr, te in splits:
            if not len(tr) or not len(te):
                continue
            wn = w[tr] / w[tr].sum() * len(w[tr])
            clf = RandomForestClassifier(n_estimators=100, max_depth=5, max_features=1,
                                         min_samples_leaf=max(1, len(tr) // 20),
                                         class_weight="balanced", random_state=42, n_jobs=-1)
            try:
                clf.fit(X[tr], y[tr], sample_weight=wn)
                scores.append(accuracy_score(y[te], clf.predict(X[te])))
                p = clf.predict_proba(X[te])
                p1 = p[:, 1] if p.shape[1] == 2 else p[:, 0]
                bets.append(np.abs(bet_size(p1)).mean())
            except Exception:
                continue
        if scores:
            log.info(f"  Accuracy : {np.mean(scores):.4f} ± {np.std(scores):.4f}")
            log.info(f"  |Bet|    : {np.mean(bets):.4f}")

        log.info("MDA (Ch.8)...")
        mda = mda_importance(X, y, w, fc, splits[: min(5, len(splits))])
        kept, dropped = [], []
        for f in fc:
            if f in mda:
                tag = "KEEP" if mda[f]["mean"] > 0 else "DROP"
                log.info(f"  {f:25s} MDA={mda[f]['mean']:+.6f} [{tag}]")
                (kept if mda[f]["mean"] > 0 else dropped).append(f)
            else:
                kept.append(f)
        if dropped:
            log.info(f"  Élagué : {dropped}")
        return kept

    def _meta(self, core, feat):
        return {
            "tf": self.tf, "strategy": self.strategy or "trend",
            "multiplier": self.multiplier, "target_bars": self.target_bars,
            "raw_bars": self.raw_bars,
            "threshold": core["threshold"],
            "effective_multiplier": core["threshold"] / core["avg_dv"] if core["avg_dv"] > 0 else None,
            "cusum_threshold": core["cusum_h"], "cusum_events": len(core["events"]),
            "actual_bars": len(core["bars"]), "labeled_events": len(core["events_df"]),
            "final_samples": len(feat), "source_rows": len(core["raw"]),
            "frac_d": self.frac_d, "horizon": self.horizon,
            "upper_width": self.upper_w, "lower_width": self.lower_w,
            "n_splits": self.n_splits, "embargo_pct": self.embargo_pct,
            "min_train": self.min_train, "momentum_lookback": self.mom_lb,
            "entropy_window": self.ent_win, "sadf_min_window": self.sadf_min,
            "spread_pct": self.spread, "swap_daily_pct": self.swap,
            "gsadf": float(core["sadf_series"].max()),
        }

    def run(self, filepath=None, mode="signal"):
        path = filepath or self._gui()
        if not path or not os.path.exists(path):
            log.error(f"Fichier introuvable : {path}")
            return
        log.info(f"Mode : {mode.upper()} | TF : {self.tf.upper()} | {os.path.basename(path)}")
        {"signal": self.run_signal, "daily": self.run_daily,
         "dashboard": self._regen_dash, "research": self.run_research,
         "strategy_select": self.run_strategy_select}.get(mode, self.run_research)(path)

    def _regen_dash(self, filepath):
        out = os.path.dirname(os.path.abspath(filepath))
        hf = os.path.join(out, "signals_history.csv")
        mf = os.path.join(out, "signal_meta.json")
        if not os.path.exists(hf):
            log.error("Pas de signals_history.csv")
            return
        sdf = pd.read_csv(hf)
        meta = json.load(open(mf)) if os.path.exists(mf) else {}
        bt = Backtest(self.spread, self.swap, 0.0001, self.bpd)
        build_dashboard(sdf, bt.run(sdf), meta, os.path.join(out, "dashboard.html"))

    def _gui(self):
        if sys.platform != "win32" and not os.environ.get("DISPLAY"):
            return ""
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.askopenfilename(title="Sélectionner un fichier OHLCV")
            root.destroy()
            return path
        except Exception:
            return ""

    def _load(self, path):
        dec = "."
        try:
            with open(path, "r") as f:
                sample = "".join(f.readline() for _ in range(5))
            if re.search(r"\d,\d{1,3}(?:[;\t\n]|$)", sample):
                dec = ","
        except Exception:
            pass
        df = pd.read_csv(path, sep=None, engine="python", decimal=dec)
        df.columns = [c.strip().lower() for c in df.columns]
        aliases = {
            "date": ["date", "datetime", "time", "timestamp", "dt", "period"],
            "open": ["open", "o"], "high": ["high", "h"],
            "low": ["low", "l"], "close": ["close", "c", "adj close", "adj_close"],
            "volume": ["volume", "vol", "v"],
        }
        remap = {}
        for std, names in aliases.items():
            for a in names:
                if a in df.columns:
                    remap[a] = std.capitalize()
                    break
        df = df.rename(columns=remap)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes : {missing}")
        for c in REQUIRED_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df[REQUIRED_COLS].dropna().sort_index()

    def _make_dollar_bars(self, df, thresh):
        if thresh <= 0:
            raise ValueError(f"Threshold invalide : {thresh}")
        idx, o, h, l, c, v, _ = dollar_bars(
            df["Open"].values.astype(np.float64), df["High"].values.astype(np.float64),
            df["Low"].values.astype(np.float64), df["Close"].values.astype(np.float64),
            df["Volume"].values.astype(np.float64), thresh,
        )
        res = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v},
                           index=df.index[idx])
        res.index.name = df.index.name or "Date"
        return res

    def _report(self, raw, bars, edf, feat):
        try:
            rt = np.log(raw["Close"] / raw["Close"].shift(1)).dropna()
            rd = np.log(bars["Close"] / bars["Close"].shift(1)).dropna()
            log.info(f"Kurtosis — Time: {kurtosis(rt):.4f} | Dollar: {kurtosis(rd):.4f}")
            ld = edf["Label"].value_counts()
            log.info(f"Labels — TP:{ld.get(1,0)} SL:{ld.get(-1,0)} Time:{ld.get(0,0)}")

            fig, ax = plt.subplots(2, 2, figsize=(14, 10))
            ax[0, 0].hist(rt, bins=100, alpha=0.5, label="Time", density=True, color="gray")
            ax[0, 0].hist(rd, bins=100, alpha=0.7, label="Dollar", density=True, color="blue")
            ax[0, 0].legend()
            ax[0, 0].set_title("Return Distribution")
            ax[0, 1].bar(["TP", "SL", "Time"], [ld.get(1, 0), ld.get(-1, 0), ld.get(0, 0)],
                         color=["green", "red", "gray"], alpha=0.7)
            ax[0, 1].set_title("Labels")
            if "sadf" in feat.columns:
                s = feat["sadf"].dropna()
                if len(s) > 0:
                    ax[1, 0].plot(s.index, s.values, color="orange", linewidth=0.8)
                    ax[1, 0].axhline(0, color="red", linestyle="--", alpha=0.5)
                    ax[1, 0].set_title("SADF")
            if "sample_weight" in feat.columns:
                ax[1, 1].hist(feat["sample_weight"], bins=50, color="purple", alpha=0.7)
                ax[1, 1].set_title("Sample Weights")
            for a in ax.flat:
                a.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("distribution_report.png", dpi=150)
            plt.close()
            log.info("Rapport : distribution_report.png")
        except Exception as e:
            log.warning(f"Rapport échoué : {e}")


# ═══════════════════════════════════════════════════════════════════════════
# Data — Yahoo Finance
# ═══════════════════════════════════════════════════════════════════════════

def volume_quality(df):
    """Évalue la fiabilité du volume pour les dollar bars."""
    vol = df["Volume"]
    dv = (df["High"] + df["Low"] + df["Close"]) / 3.0 * vol
    zero = (vol == 0).mean()
    cv = vol.std() / vol.mean() if vol.mean() > 0 else float("inf")
    if zero > 0.05 or cv > 3.0:
        grade = "POOR"
    elif zero > 0.01 or cv > 1.5:
        grade = "FAIR"
    else:
        grade = "GOOD"
    return {"grade": grade, "cv": round(cv, 2), "zero_pct": round(zero * 100, 1)}


def fetch(ticker, start="2020-01-01", out_dir="."):
    """Télécharge les données D1 depuis Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError:
        log.error("yfinance requis : pip install yfinance")
        sys.exit(1)

    yf_sym = TICKER_MAP.get(ticker.upper(), ticker)
    log.info(f"Téléchargement {ticker} ({yf_sym}) depuis {start}...")
    data = yf.download(yf_sym, start=start, interval="1d", auto_adjust=True, progress=False)
    if data.empty:
        log.error(f"Aucune donnée pour {yf_sym}")
        sys.exit(1)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data[["Open", "High", "Low", "Close", "Volume"]].dropna()
    data.index.name = "Date"
    fn = os.path.join(out_dir, f"{ticker.upper()}_D1.csv")
    data.to_csv(fn)
    log.info(f"Sauvé : {fn} ({len(data)} barres, {data.index[0].date()} → {data.index[-1].date()})")
    return fn


# ═══════════════════════════════════════════════════════════════════════════
# Optimiseur — Random Search
# ═══════════════════════════════════════════════════════════════════════════

def optimize(filepath, n_iter=50, tf="d1", raw_bars=False, min_trades=30):
    rng = np.random.RandomState(42)
    ranges = {
        "horizon": (3, 20), "momentum_lookback": (3, 30),
        "upper_width": (0.3, 2.5), "lower_width": (0.3, 2.5),
    }

    log.info(f"{'=' * 60}")
    log.info(f"RANDOM SEARCH — {n_iter} itérations | Min trades : {min_trades}")
    log.info(f"{'=' * 60}")

    results = []
    for i in range(n_iter):
        p = {
            "horizon": int(rng.randint(*ranges["horizon"])),
            "momentum_lookback": int(rng.randint(*ranges["momentum_lookback"])),
            "upper_width": round(float(rng.uniform(*ranges["upper_width"])), 2),
            "lower_width": round(float(rng.uniform(*ranges["lower_width"])), 2),
        }
        try:
            pipe = Pipeline(tf=tf, horizon=p["horizon"], momentum_lookback=p["momentum_lookback"],
                            upper_width=p["upper_width"], lower_width=p["lower_width"], raw_bars=raw_bars)
            st = pipe.run_eval(filepath)
        except Exception:
            st = None

        if not st or st.get("total_trades", 0) < min_trades:
            tag = "SKIP" if not st else f"SKIP ({st.get('total_trades', 0)}t)"
            log.info(f"  [{i+1:3d}/{n_iter}] H={p['horizon']:2d} Mom={p['momentum_lookback']:2d} "
                     f"Up={p['upper_width']:.2f} Lo={p['lower_width']:.2f} → {tag}")
            continue

        results.append({"params": p, "stats": st})
        log.info(f"  [{i+1:3d}/{n_iter}] H={p['horizon']:2d} Mom={p['momentum_lookback']:2d} "
                 f"Up={p['upper_width']:.2f} Lo={p['lower_width']:.2f} → "
                 f"SR={st['sharpe']:.2f} Ret={st['total_return_pct']:+.1f}% "
                 f"DD={st['max_drawdown_pct']:.1f}% WR={st['win_rate']:.1%}")

    if not results:
        log.error("Aucune configuration valide.")
        return {}

    results.sort(key=lambda x: x["stats"]["sharpe"], reverse=True)

    log.info(f"\n{'=' * 60}")
    log.info("TOP 5 :")
    for rank, r in enumerate(results[:5], 1):
        p, s = r["params"], r["stats"]
        log.info(f"  #{rank} SR={s['sharpe']:.2f} Ret={s['total_return_pct']:+.1f}% "
                 f"DD={s['max_drawdown_pct']:.1f}% WR={s['win_rate']:.1%} "
                 f"T={s['total_trades']} | H={p['horizon']} Mom={p['momentum_lookback']} "
                 f"Up={p['upper_width']:.2f} Lo={p['lower_width']:.2f}")

    best = results[0]
    bs = best["stats"]
    dsr = deflated_sharpe(bs["sharpe"], len(results), bs["total_trades"],
                          bs.get("ret_skew", 0), bs.get("ret_kurt", 3))
    bs["dsr"] = dsr
    sig = "SIGNIFICATIF" if dsr > 0.95 else "NON SIGNIFICATIF"
    log.info(f"\nDSR : {dsr:.4f} ({sig})")
    log.info(f"Meilleur : SR={bs['sharpe']:.2f} — lancement signal mode...")
    return best


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="DePrado Dollar Pipeline — AFML Trading System")
    p.add_argument("file", nargs="?", default=None, help="Fichier OHLCV (.csv)")

    # Modes
    p.add_argument("--signal", action="store_true", help="Signal + backtest + dashboard")
    p.add_argument("--daily", action="store_true", help="Mode cron incrémental")
    p.add_argument("--dashboard", action="store_true", help="Régénérer le dashboard")
    p.add_argument("--optimize", action="store_true", help="Random search")
    p.add_argument("--research", action="store_true", help="Recherche (CPCV, MDA)")
    p.add_argument("--strategy-select", action="store_true", help="Compare les 5 stratégies et lance le meilleur")
    p.add_argument("--n-iter", type=int, default=50, help="Itérations random search")

    # Data
    p.add_argument("--fetch", type=str, default=None, help="Ticker Yahoo Finance")
    p.add_argument("--start", type=str, default="2020-01-01", help="Date de début")

    # Paramètres
    p.add_argument("--tf", type=str, default="d1", nargs="+", help="Timeframe(s) : h4 d1 w1")
    p.add_argument("--strategy", type=str, default=None,
                   choices=["momentum", "trend", "mr", "bo", "bb"],
                   help="Forcer une stratégie (sinon auto-sélection)")
    p.add_argument("--multiplier", type=float, default=None)
    p.add_argument("--target-bars", type=int, default=None)
    p.add_argument("--cusum", type=float, default=None)
    p.add_argument("--frac-d", type=float, default=0.4)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--upper", type=float, default=1.0)
    p.add_argument("--lower", type=float, default=1.0)
    p.add_argument("--momentum", type=int, default=None)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--embargo", type=float, default=0.01)
    p.add_argument("--min-train", type=int, default=50)
    p.add_argument("--spread", type=float, default=None)
    p.add_argument("--swap", type=float, default=None)
    p.add_argument("--raw", action="store_true", help="Forcer les time bars")

    args = p.parse_args()

    if args.optimize:
        mode = "optimize"
    elif args.strategy_select:
        mode = "strategy_select"
    elif args.signal:
        mode = "signal"
    elif args.daily:
        mode = "daily"
    elif args.dashboard:
        mode = "dashboard"
    elif args.research:
        mode = "research"
    else:
        mode = "signal"

    filepath = args.file

    if args.fetch:
        filepath = fetch(args.fetch, args.start)
    elif not filepath:
        filepath = fetch("GLD", args.start)

    if not args.raw:
        try:
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            vq = volume_quality(df)
            log.info(f"Volume : {vq['grade']} (CV={vq['cv']}, zéros={vq['zero_pct']}%)")
            if vq["grade"] == "POOR":
                log.warning("Volume insuffisant pour les dollar bars — fallback time bars")
                args.raw = True
        except Exception:
            pass

    tfs = args.tf if isinstance(args.tf, list) else [args.tf]
    tfs = [t for t in tfs if t in ("h4", "d1", "w1")]
    if not tfs:
        log.error("TF invalide")
        sys.exit(1)

    for tf in tfs:
        log.info(f"{'=' * 60}")
        log.info(f"TF : {tf.upper()}")
        log.info(f"{'=' * 60}")

        base = os.path.dirname(os.path.abspath(filepath))
        tf_dir = os.path.join(base, tf.upper())
        os.makedirs(tf_dir, exist_ok=True)
        tf_path = os.path.join(tf_dir, os.path.basename(filepath))
        if not os.path.exists(tf_path) or os.path.abspath(filepath) != os.path.abspath(tf_path):
            shutil.copy2(filepath, tf_path)

        if mode == "optimize":
            best = optimize(tf_path, args.n_iter, tf, args.raw)
            if best:
                bp = best["params"]
                log.info(f"Exécution finale : H={bp['horizon']} Mom={bp['momentum_lookback']} "
                         f"Up={bp['upper_width']:.2f} Lo={bp['lower_width']:.2f}")
                pipe = Pipeline(
                    tf=tf, horizon=bp["horizon"], momentum_lookback=bp["momentum_lookback"],
                    upper_width=bp["upper_width"], lower_width=bp["lower_width"],
                    raw_bars=args.raw, frac_d=args.frac_d, n_splits=args.n_splits,
                    embargo_pct=args.embargo, min_train=args.min_train,
                    spread_pct=args.spread, swap_daily_pct=args.swap,
                )
                pipe.run(tf_path, "signal")
                with open(os.path.join(tf_dir, "optimize_results.json"), "w") as f:
                    json.dump({"best_params": best["params"], "best_stats": best["stats"],
                               "n_iter": args.n_iter, "tf": tf}, f, indent=4, default=str)
        else:
            try:
                pipe = Pipeline(
                    tf=tf, multiplier=args.multiplier, frac_d=args.frac_d,
                    horizon=args.horizon, upper_width=args.upper, lower_width=args.lower,
                    target_bars=args.target_bars, cusum_pct=args.cusum,
                    n_splits=args.n_splits, embargo_pct=args.embargo,
                    min_train=args.min_train, momentum_lookback=args.momentum,
                    spread_pct=args.spread, swap_daily_pct=args.swap, raw_bars=args.raw,
                    strategy=args.strategy,
                )
                pipe.run(tf_path, mode)
            except (ValueError, AssertionError) as e:
                log.error(f"Erreur ({tf}) : {e}")

    log.info(f"{'=' * 60}")
    log.info(f"Terminé — {len(tfs)} timeframe(s) : {', '.join(t.upper() for t in tfs)}")


if __name__ == "__main__":
    main()
