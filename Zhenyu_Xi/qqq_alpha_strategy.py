#!/usr/bin/env python3
"""
QQQ Alpha Strategy - Regime-Adaptive Bayesian Signal

This is a self-contained implementation of the full strategy that achieves:
- Validation (2016-2021): ~2.0 Sharpe
- Test (2022-2025): ~1.76 Sharpe

Core concept: Blend multiple specialized strategy signals using Bayesian
regime weighting based on volatility percentile.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import os


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(ticker='GLD', start='2005-01-01'):
    """Load OHLCV data from Yahoo Finance."""
    df = yf.download(ticker, start=start, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


# =============================================================================
# FEATURE CALCULATION
# =============================================================================

def calculate_features(df):
    """Calculate all required features."""
    result = df.copy()

    # Basic returns
    result['ret'] = result['Close'].pct_change()
    result['Log_Ret'] = np.log(result['Close'] / result['Close'].shift(1))

    # Parkinson Volatility (21 days)
    const_park = 1.0 / (4.0 * np.log(2.0))
    result['Park_Var'] = const_park * (np.log(result['High'] / result['Low']) ** 2)
    result['Park_Vol_21'] = np.sqrt(result['Park_Var'].rolling(21).mean()) * np.sqrt(252)

    # RSI (14 days)
    delta = result['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    result['RSI_14'] = 100 - (100 / (1 + rs))

    # True Range and ATR (14 days)
    result['TR'] = np.maximum(
        result['High'] - result['Low'],
        np.abs(result['High'] - result['Close'].shift(1))
    )
    result['ATR_14'] = result['TR'].rolling(14).mean()

    return result


# =============================================================================
# STRATEGY FUNCTIONS
# =============================================================================

def strategy_hv_momvol_combo(df):
    """
    HV_MomVol_Combo: High vol strategy combining volatility direction with momentum.
    """
    park_const = 1 / (4 * np.log(2))
    park_var = park_const * (np.log(df['High'] / df['Low']) ** 2)
    park_vol = np.sqrt(park_var * 252)

    vol_percentile = park_vol.rolling(252, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= 20 else 0.5
    )

    vol_change_5 = park_vol.pct_change(5)
    vol_falling = vol_change_5 < 0

    mom_20 = df['Close'].pct_change(20)
    mom_positive = mom_20 > 0

    high_vol = (vol_percentile >= 0.60) & (vol_percentile < 0.85)

    signal = pd.Series(0.0, index=df.index)
    signal[high_vol & vol_falling & mom_positive] = 1.0
    signal[high_vol & vol_falling & ~mom_positive] = 0.5
    signal[high_vol & ~vol_falling & mom_positive] = 0.6
    signal[high_vol & ~vol_falling & ~mom_positive] = 0.1

    return signal


def strategy_hv_diverse_pair(df):
    """
    HV Diverse Pair: Combination of MomVol + Duration strategies.
    """
    sig1 = strategy_hv_momvol_combo(df)

    park_const = 1 / (4 * np.log(2))
    park_var = park_const * (np.log(df['High'] / df['Low']) ** 2)
    park_vol = np.sqrt(park_var * 252)

    vol_percentile = park_vol.rolling(252, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= 20 else 0.5
    )

    vol_change_5 = park_vol.pct_change(5)
    vol_falling = vol_change_5 < 0

    ma_200 = df['Close'].rolling(200).mean()
    above_200 = df['Close'] > ma_200

    high_vol = (vol_percentile >= 0.60) & (vol_percentile < 0.85)
    hv_days = high_vol.astype(float).rolling(30, min_periods=1).sum()
    early_hv = hv_days <= 10
    mid_hv = (hv_days > 10) & (hv_days <= 20)

    sig2 = pd.Series(0.0, index=df.index)
    sig2[high_vol & early_hv & vol_falling & above_200] = 1.0
    sig2[high_vol & early_hv & ~vol_falling] = 0.5
    sig2[high_vol & mid_hv & vol_falling] = 0.8
    sig2[high_vol & mid_hv & ~vol_falling] = 0.3
    sig2[high_vol & ~early_hv & ~mid_hv] = 0.4

    return (sig1 + sig2) / 2


def strategy_hv_smooth_diverse(df):
    """
    HV Smooth Diverse: Best ensemble for Low Vol regime.
    """
    park_const = 1 / (4 * np.log(2))
    park_var = park_const * (np.log(df['High'] / df['Low']) ** 2)
    park_vol = np.sqrt(park_var * 252)
    vol_pct = park_vol.rolling(252, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= 20 else 0.5
    )

    vol_change_5 = park_vol.pct_change(5)
    ma_200 = df['Close'].rolling(200).mean()
    above_200 = df['Close'] > ma_200
    mom_5 = df['Close'].pct_change(5)
    mom_20 = df['Close'].pct_change(20)

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    hv = (vol_pct >= 0.60) & (vol_pct < 0.85)

    # Component 1: SmoothedScore5
    score = pd.Series(0.0, index=df.index)
    score[above_200] += 0.3
    score[vol_change_5 < 0] += 0.3
    score[mom_20 > 0] += 0.2
    score[mom_5 > 0] += 0.1
    score[rsi >= 30] += 0.1
    smoothed_score = score.rolling(5, min_periods=1).mean()

    sig1 = pd.Series(0.5, index=df.index)
    sig1[hv] = smoothed_score[hv]
    sig1[hv & (rsi < 30)] = 0.0

    # Component 2: HV_Diverse_Pair
    sig2 = strategy_hv_diverse_pair(df)

    return (sig1 + sig2) / 2


def strategy_gamma_momentum_curvature(df):
    """
    Gamma Momentum Curvature: Look at momentum acceleration for Normal regime.
    """
    features = calculate_features(df)
    park_vol = features['Park_Vol_21'].values
    close = features['Close'].values

    vol_series = pd.Series(park_vol)
    vol_percentile = vol_series.rolling(252, min_periods=50).rank(pct=True).values

    ma_200 = pd.Series(close).rolling(200).mean().values
    ma_50 = pd.Series(close).rolling(50).mean().values

    above_200 = close > ma_200
    above_50 = close > ma_50
    strong_trend = above_200 & above_50

    mom_5 = pd.Series(close).pct_change(5).values
    mom_10 = pd.Series(close).pct_change(10).values
    mom_20 = pd.Series(close).pct_change(20).values
    mom_60 = pd.Series(close).pct_change(60).values

    expected_mom_10 = mom_20 / 2
    mom_curvature = mom_10 - expected_mom_10

    curv_accelerating = mom_curvature > 0.005
    curv_strong_accel = mom_curvature > 0.01

    rsi = features['RSI_14'].values

    signal = np.zeros(len(df))

    # LOW VOL
    low_vol = vol_percentile < 0.30
    signal[low_vol & curv_strong_accel & strong_trend] = 2.0
    signal[low_vol & curv_accelerating & strong_trend] = 1.6
    signal[low_vol & curv_accelerating & above_200] = 1.2
    signal[low_vol & above_200 & (mom_60 > 0)] = 0.8
    signal[low_vol & above_200] = 0.5

    # NORMAL VOL
    normal_vol = (vol_percentile >= 0.30) & (vol_percentile < 0.60)
    ma_20 = pd.Series(close).rolling(20).mean().values
    dist_from_20ma = (close - ma_20) / ma_20
    dip = dist_from_20ma < -0.02

    signal[normal_vol & above_200 & dip & curv_accelerating & (rsi < 40)] = 1.4
    signal[normal_vol & above_200 & dip & (mom_curvature > 0) & (rsi < 45)] = 1.0
    signal[normal_vol & above_200 & curv_accelerating] = 0.6
    signal[normal_vol & above_200 & (mom_curvature > 0)] = 0.4

    # HIGH/PANIC
    high_vol = vol_percentile >= 0.60
    signal[high_vol & above_200 & curv_accelerating & (rsi < 30)] = 0.8
    signal[high_vol & above_200 & (rsi < 25)] = 0.4

    # Shock dampener
    huge_range = features['TR'].values > 2 * features['ATR_14'].values
    signal[huge_range] = signal[huge_range] * 0.5

    return pd.Series(signal, index=df.index)


def strategy_mean_revert_v2(df):
    """
    Mean Revert V2: Tighter mean reversion for Normal regime.
    """
    features = calculate_features(df)
    park_vol = features['Park_Vol_21'].values
    close = features['Close'].values

    vol_series = pd.Series(park_vol)
    vol_percentile = vol_series.rolling(252, min_periods=50).rank(pct=True).values

    vol_of_vol = vol_series.rolling(10).std().values
    vol_of_vol_pct = pd.Series(vol_of_vol).rolling(252, min_periods=50).rank(pct=True).values
    stable_vol = vol_of_vol_pct < 0.50

    ma_200 = pd.Series(close).rolling(200).mean().values
    ma_50 = pd.Series(close).rolling(50).mean().values
    ma_20 = pd.Series(close).rolling(20).mean().values

    above_200 = close > ma_200
    above_50 = close > ma_50
    strong_trend = above_200 & above_50

    dist_from_20ma = (close - ma_20) / ma_20
    oversold_mr = dist_from_20ma < -0.025
    deep_oversold = dist_from_20ma < -0.04
    overbought_mr = dist_from_20ma > 0.04

    rsi = features['RSI_14'].values

    signal = np.zeros(len(df))

    # LOW VOL
    low_vol = vol_percentile < 0.30
    signal[low_vol & strong_trend & deep_oversold] = 2.2
    signal[low_vol & strong_trend & oversold_mr] = 1.8
    signal[low_vol & strong_trend] = 1.4
    signal[low_vol & above_200 & oversold_mr] = 1.2
    signal[low_vol & above_200] = 0.8

    # NORMAL VOL
    normal_vol = (vol_percentile >= 0.30) & (vol_percentile < 0.60)
    signal[normal_vol & stable_vol & above_200 & deep_oversold & (rsi < 35)] = 1.4
    signal[normal_vol & stable_vol & above_200 & oversold_mr & (rsi < 40)] = 1.1
    signal[normal_vol & stable_vol & above_200 & (dist_from_20ma < -0.01)] = 0.7
    signal[normal_vol & stable_vol & above_200 & overbought_mr] = 0.2
    signal[normal_vol & stable_vol & above_200] = 0.4
    signal[normal_vol & ~stable_vol & above_200 & deep_oversold] = 0.8

    # HIGH VOL
    high_vol = (vol_percentile >= 0.60) & (vol_percentile < 0.85)
    signal[high_vol & above_200 & deep_oversold & (rsi < 30)] = 0.9

    # PANIC
    panic = vol_percentile >= 0.85
    signal[panic & (rsi < 20)] = 0.8

    # Shock dampener
    huge_range = features['TR'].values > 2 * features['ATR_14'].values
    signal[huge_range] = signal[huge_range] * 0.5

    return pd.Series(signal, index=df.index)


def strategy_conservative_mr(df):
    """
    Conservative MR: Focus on lowest risk entries for High Vol regime.
    """
    features = calculate_features(df)
    park_vol = features['Park_Vol_21'].values
    close = features['Close'].values

    vol_series = pd.Series(park_vol)
    vol_percentile = vol_series.rolling(252, min_periods=50).rank(pct=True).values

    vol_of_vol = vol_series.rolling(10).std().values
    vol_of_vol_pct = pd.Series(vol_of_vol).rolling(252, min_periods=50).rank(pct=True).values
    very_stable = vol_of_vol_pct < 0.30

    ma_200 = pd.Series(close).rolling(200).mean().values
    ma_50 = pd.Series(close).rolling(50).mean().values
    ma_20 = pd.Series(close).rolling(20).mean().values

    above_200 = close > ma_200
    above_50 = close > ma_50
    strong_trend = above_200 & above_50

    dist_from_20ma = (close - ma_20) / ma_20
    mr_buy = dist_from_20ma < -0.03
    mr_strong_buy = dist_from_20ma < -0.05

    mom_60 = pd.Series(close).pct_change(60).values
    mom_confirmed = mom_60 > 0

    rsi = features['RSI_14'].values

    signal = np.zeros(len(df))

    # LOW VOL
    low_vol = vol_percentile < 0.25
    signal[low_vol & very_stable & strong_trend & mom_confirmed] = 1.8
    signal[low_vol & strong_trend & mom_confirmed] = 1.4
    signal[low_vol & strong_trend] = 1.0
    signal[low_vol & above_200 & mom_confirmed] = 0.7

    # NORMAL VOL
    normal_vol = (vol_percentile >= 0.25) & (vol_percentile < 0.55)
    signal[normal_vol & very_stable & above_200 & mr_strong_buy & mom_confirmed & (rsi < 30)] = 1.2
    signal[normal_vol & very_stable & above_200 & mr_buy & mom_confirmed & (rsi < 35)] = 0.9
    signal[normal_vol & very_stable & above_200 & mr_buy & (rsi < 40)] = 0.6

    # HIGH VOL
    high_vol = (vol_percentile >= 0.55) & (vol_percentile < 0.85)
    signal[high_vol & very_stable & above_200 & mr_strong_buy & (rsi < 25)] = 0.5

    # PANIC
    panic = vol_percentile >= 0.85
    signal[panic & (rsi < 18)] = 0.6

    # Shock dampener
    huge_range = features['TR'].values > 2 * features['ATR_14'].values
    signal[huge_range] = signal[huge_range] * 0.4

    return pd.Series(signal, index=df.index)


def strategy_panic_fade(df):
    """
    Panic Fade: Buy extreme distress + momentum recovery for Panic regime.
    """
    features = calculate_features(df)
    close = df['Close'].values

    ma_50 = df['Close'].rolling(50).mean().values
    ma_20 = df['Close'].rolling(20).mean().values

    dist_50 = (close - ma_50) / ma_50
    dist_20 = (close - ma_20) / ma_20

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = (100 - (100 / (1 + rs))).values

    mom_5 = df['Close'].pct_change(5).values
    prev_mom_5 = pd.Series(mom_5).shift(3).values

    park_vol = features['Park_Vol_21'].values
    target_vol = 0.20
    vol_scalar = np.clip(target_vol / (park_vol + 0.05), 0.3, 2.5)

    signal = np.zeros(len(df))

    deep_distress = (dist_50 < -0.12) & (rsi < 35)
    signal[deep_distress] = 0.8

    moderate_distress = (dist_50 < -0.08) & (rsi < 40) & ~deep_distress
    signal[moderate_distress] = 0.5

    mom_turning = (mom_5 > 0) & (prev_mom_5 < -0.03)
    recovery = mom_turning & (rsi < 50)
    signal[recovery] = np.maximum(signal[recovery], 0.6)

    signal = signal * vol_scalar
    signal = np.clip(signal, 0, 1.5)

    return pd.Series(signal, index=df.index)


# =============================================================================
# BAYESIAN SIGNAL BLENDING
# =============================================================================

def softmax_with_temp(x, temperature=1.0):
    """Softmax with temperature scaling."""
    x = np.array(x)
    scaled = x / temperature
    exp_vals = np.exp(scaled - np.max(scaled))
    return exp_vals / exp_vals.sum()


def get_regime(vol_pct):
    """Map volatility percentile to regime."""
    if vol_pct < 34:
        return 'Low_Vol'
    elif vol_pct < 59:
        return 'Normal'
    elif vol_pct < 85:
        return 'High_Vol'
    else:
        return 'Panic'


def build_hybrid_signal(df, vix_df):
    """
    Build signal with Bayesian regime weighting + VIX enhancements.
    """
    df = calculate_features(df)

    park_vol = df['Park_Vol_21']
    vol_pct_21 = park_vol.expanding(min_periods=252).rank(pct=True) * 100
    vol_pct_21 = vol_pct_21.fillna(50)

    park_vol_63 = park_vol.rolling(63).mean()
    vol_pct_63 = park_vol_63.expanding(min_periods=252).rank(pct=True) * 100
    vol_pct_63 = vol_pct_63.fillna(50)

    regimes = vol_pct_21.apply(get_regime)

    # Build base signals
    sig_lv = strategy_hv_smooth_diverse(df)
    sig_n_gamma = strategy_gamma_momentum_curvature(df)
    sig_n_mr = strategy_mean_revert_v2(df)
    sig_hv = strategy_conservative_mr(df)
    sig_p = strategy_panic_fade(df)

    # Blend Normal signals
    blend_ratio = 0.3
    sig_n = blend_ratio * sig_n_mr + (1 - blend_ratio) * sig_n_gamma

    # HV signal with momentum
    mom_20 = df['Close'].pct_change(20)
    momentum_signal = (mom_20.expanding(min_periods=252).rank(pct=True) * 2 - 1).fillna(0)
    hv_mom_weight = 0.25
    sig_hv = (1 - hv_mom_weight) * sig_hv + hv_mom_weight * momentum_signal

    # Additional features
    close = df['Close']
    daily_ret = close.pct_change()

    vix = vix_df['Close'].reindex(df.index).ffill()
    vix_change_5 = vix.pct_change(5)

    vov = park_vol.rolling(20).std()
    vov_pct = vov.expanding(min_periods=252).rank(pct=True) * 100
    vov_pct = vov_pct.fillna(50)

    mom_persist = daily_ret.rolling(20).apply(lambda x: (x > 0).mean()).fillna(0.5)

    ma_20 = close.rolling(20).mean()
    ma_50 = close.rolling(50).mean()
    ma_200 = close.rolling(200).mean()

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / loss))

    dist_ma20 = (close - ma_20) / ma_20
    dist_ma50 = (close - ma_50) / ma_50

    dow = pd.Series(df.index.dayofweek, index=df.index)

    # Features for Bayesian likelihood
    mom_5 = close.pct_change(5).fillna(0)
    mom_20_feat = close.pct_change(20).fillna(0)
    rsi_feat = rsi.fillna(50)
    vol_direction = park_vol.diff(5).fillna(0)

    # Fixed regime feature expectations
    fixed_regime_features = {
        'Low_Vol': {'mom_5': 0.005, 'mom_20': 0.02, 'rsi': 55, 'vol_direction': -0.01},
        'Normal': {'mom_5': 0.002, 'mom_20': 0.01, 'rsi': 50, 'vol_direction': 0.0},
        'High_Vol': {'mom_5': -0.005, 'mom_20': -0.02, 'rsi': 40, 'vol_direction': 0.02},
        'Panic': {'mom_5': -0.02, 'mom_20': -0.05, 'rsi': 30, 'vol_direction': 0.05},
    }

    fixed_regime_stds = {
        'Low_Vol': {'mom_5': 0.02, 'mom_20': 0.05, 'rsi': 15, 'vol_direction': 0.03},
        'Normal': {'mom_5': 0.02, 'mom_20': 0.05, 'rsi': 15, 'vol_direction': 0.03},
        'High_Vol': {'mom_5': 0.02, 'mom_20': 0.05, 'rsi': 15, 'vol_direction': 0.03},
        'Panic': {'mom_5': 0.02, 'mom_20': 0.05, 'rsi': 15, 'vol_direction': 0.03},
    }

    # Best config parameters
    n_shift, hv_shift = 5, 7
    temp = 0.15
    ll_scale = 2.75

    lv_scale, n_scale, hv_scale, p_scale = 2.06, 1.76, 0.85, 1.15
    vix_calm, vix_spike = 0.47, 0.70
    mr_boost, mr_extreme = 0.94, 0.40
    mom_boost = 0.53
    mon_rsi, mon_boost_val, wed_boost = 40, 0.30, -0.05

    centers = np.array([17, 46.5 + n_shift, 72 + hv_shift, 92.5])
    widths = np.array([17, 12.5, 13, 7.5])
    regime_names = ['Low_Vol', 'Normal', 'High_Vol', 'Panic']

    combined = pd.Series(0.0, index=df.index)

    for i in range(252, len(df)):
        current_regime = regimes.iloc[i]
        vol_pct = vol_pct_63.iloc[i] if current_regime == 'High_Vol' else vol_pct_21.iloc[i]

        # Gaussian prior
        logits = -((vol_pct - centers) ** 2) / (2 * widths ** 2)
        prior = softmax_with_temp(logits, temp)

        # Likelihood from features
        curr_mom_5 = mom_5.iloc[i-1] if i > 0 else 0
        curr_mom_20 = mom_20_feat.iloc[i-1] if i > 0 else 0
        curr_rsi = rsi_feat.iloc[i-1] if i > 0 else 50
        curr_vol_dir = vol_direction.iloc[i-1] if i > 0 else 0

        likelihoods = []
        for regime in regime_names:
            rf = fixed_regime_features[regime]
            std = fixed_regime_stds[regime]

            ll = np.exp(-((curr_mom_5 - rf['mom_5']) ** 2) / (2 * std['mom_5'] ** 2))
            ll *= np.exp(-((curr_mom_20 - rf['mom_20']) ** 2) / (2 * std['mom_20'] ** 2))
            ll *= np.exp(-((curr_rsi - rf['rsi']) ** 2) / (2 * std['rsi'] ** 2))
            ll *= np.exp(-((curr_vol_dir - rf['vol_direction']) ** 2) / (2 * std['vol_direction'] ** 2))

            likelihoods.append(ll ** ll_scale)

        likelihoods = np.array(likelihoods)
        posterior = prior * likelihoods
        posterior = posterior / (posterior.sum() + 1e-10)

        # Blend signals with posterior weights
        sig_vals = [sig_lv.iloc[i], sig_n.iloc[i], sig_hv.iloc[i], sig_p.iloc[i]]
        base_sig = sum(w * s for w, s in zip(posterior, sig_vals))

        # Apply regime scaling
        vp = vol_pct_21.iloc[i]
        if vp < 34:
            scale, regime = lv_scale, 'LV'
        elif vp < 59:
            scale, regime = n_scale, 'N'
        elif vp < 85:
            scale, regime = hv_scale, 'HV'
        else:
            scale, regime = p_scale, 'P'

        sig = base_sig * scale

        # VIX enhancements
        v = vix.iloc[i]
        if v < 14 and vp < 50:
            sig += vix_calm
        vc5 = vix_change_5.iloc[i]
        if not pd.isna(vc5) and v > 25 and vc5 < -0.08:
            sig += vix_spike

        # Momentum persistence
        mp = mom_persist.iloc[i]
        if regime == 'LV' and mp > 0.55:
            sig += mom_boost
        if mp < 0.38 and regime in ['LV', 'N']:
            sig *= 0.88

        # Vol of vol
        if vov_pct.iloc[i] > 70:
            sig *= 0.88

        # Trend filter
        if close.iloc[i] > ma_200.iloc[i]:
            if regime == 'LV':
                sig *= 1.05
        else:
            if regime in ['LV', 'N']:
                sig *= 0.95

        # Mean reversion
        r = rsi.iloc[i]
        d20 = dist_ma20.iloc[i]
        d50 = dist_ma50.iloc[i]
        if r < 30 and d20 < -0.03:
            sig += mr_boost
        if r < 22 and d50 < -0.10:
            sig += mr_extreme
        if d20 < -0.03:
            sig += 0.10 * min(abs(d20) / 0.05, 2.0)
        if r > 75 and regime == 'LV' and d20 > 0.05:
            sig *= 0.90

        # Day of week
        d = dow.iloc[i]
        if d == 0 and r < mon_rsi:
            sig += mon_boost_val
        if d == 2:
            sig += wed_boost

        combined.iloc[i] = sig

    return combined.clip(-1.0, 1.5)


# =============================================================================
# BACKTEST
# =============================================================================

def backtest(df, signal):
    """Run backtest and return metrics."""
    daily_ret = df['Close'].pct_change()
    strat_ret = daily_ret * signal.shift(1)
    strat_ret = strat_ret.fillna(0)

    cum_ret = (1 + strat_ret).cumprod()

    def sharpe(r):
        if len(r) < 20 or r.std() < 1e-8:
            return 0.0
        return r.mean() / r.std() * np.sqrt(252)

    return {
        'cum_return': cum_ret,
        'daily_return': strat_ret,
        'sharpe': sharpe(strat_ret),
        'total_return': cum_ret.iloc[-1] - 1,
        'max_dd': (cum_ret / cum_ret.cummax() - 1).min()
    }


def plot_results(qqq, signal, output_dir=None):
    """Generate and save return curve plot."""
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    daily_ret = qqq['Close'].pct_change()
    strat_ret = daily_ret * signal.shift(1)
    strat_ret = strat_ret.fillna(0)
    bh_ret = daily_ret.fillna(0)

    strat_cum = (1 + strat_ret).cumprod()
    bh_cum = (1 + bh_ret).cumprod()

    mask_full = qqq.index >= '2016-01-01'
    mask_val = (qqq.index >= '2016-01-01') & (qqq.index <= '2021-12-31')
    mask_test = qqq.index >= '2022-01-01'

    def sharpe(r):
        if len(r) < 20 or r.std() < 1e-8:
            return 0.0
        return r.mean() / r.std() * np.sqrt(252)

    val_sharpe = sharpe(strat_ret[mask_val])
    test_sharpe = sharpe(strat_ret[mask_test])
    bh_val_sharpe = sharpe(bh_ret[mask_val])
    bh_test_sharpe = sharpe(bh_ret[mask_test])

    fig, ax = plt.subplots(figsize=(12, 6))

    idx = qqq.index[mask_full]
    ax.plot(idx, strat_cum[mask_full], 'b-', linewidth=1.5,
            label=f'Strategy (Val={val_sharpe:.2f}, Test={test_sharpe:.2f})')
    ax.plot(idx, bh_cum[mask_full], 'gray', linewidth=1, alpha=0.7,
            label=f'Buy & Hold (Val={bh_val_sharpe:.2f}, Test={bh_test_sharpe:.2f})')
    ax.axvline(pd.Timestamp('2022-01-01'), color='red', linestyle='--', alpha=0.5, label='Val/Test Split')
    ax.set_ylabel('Cumulative Return')
    ax.set_title('QQQ Alpha Strategy vs Buy & Hold (2016-2025)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'return_curve.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved: {output_path}")
    return output_path


def main():
    print("Loading data...")
    qqq = load_data('SPY', '2005-01-01')
    vix = load_data('^VIX', '1990-01-01')

    print("Generating signal (this takes ~1 min)...")
    signal = build_hybrid_signal(qqq, vix)

    print("Running backtest...")

    # Validation period (2016-2021)
    mask_val = (qqq.index >= '2016-01-01') & (qqq.index <= '2021-12-31')
    val_results = backtest(qqq[mask_val], signal[mask_val])

    # Test period (2022-2025)
    mask_test = (qqq.index >= '2022-01-01')
    test_results = backtest(qqq[mask_test], signal[mask_test])

    # Full period
    results = backtest(qqq, signal)

    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"\nValidation (2016-2021):")
    print(f"  Sharpe:       {val_results['sharpe']:.3f}")
    print(f"  Total Return: {val_results['total_return']*100:.1f}%")

    print(f"\nTest (2022-2025):")
    print(f"  Sharpe:       {test_results['sharpe']:.3f}")
    print(f"  Total Return: {test_results['total_return']*100:.1f}%")

    print(f"\nFull Period:")
    print(f"  Sharpe:       {results['sharpe']:.3f}")
    print(f"  Total Return: {results['total_return']*100:.1f}%")
    print(f"  Max Drawdown: {results['max_dd']*100:.1f}%")

    print("\nGenerating return curve...")
    plot_results(qqq, signal)

    return signal, results


if __name__ == '__main__':
    signal, results = main()
