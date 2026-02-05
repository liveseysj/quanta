#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quanta Fellowship — M2 baseline
Cleaned + report artifacts (plots + LaTeX tables) WITHOUT changing the trading model.

Model logic is identical:
- same signals
- same grouping + normalization
- same bucket scaling + throttle + caps
- same 2-regime weights + optional per-bucket weights with shrinkage
- same hedge overlay + lambda grid search constraint on VALID
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==============================================================================
# Global constants
# ==============================================================================
ANNUALIZATION = 252.0
EPS = 1e-9

# ==============================================================================
# Paths (adjust if needed)
# ==============================================================================
PATH_TRAIN_VALID = "Quanta Fellowship QQQ Train and Validation Data - Sheet1.csv"
PATH_HOLDOUT = "Quanta Fellowship QQQ Blind Out of Sample - Sheet1.csv"
PATH_VXN = "VXNCLS.csv"
PATH_VIX = "VIXCLS.csv"
PATH_VXV = "VXVCLS.csv"

# ==============================================================================
# Data loading & preparation
# ==============================================================================


def clean_price_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a Quanta-style OHLCV dataframe:
      - Drop a possible trailing 'Downloaded ...' footer row
      - Normalize column names to lowercase
      - Rename 'latest' -> 'close', 'time' -> 'date'
      - Parse dates using Quanta format: %m/%d/%y
      - Convert OHLCV columns to numeric
    """
    df = df.copy()

    if len(df) > 0 and isinstance(df.iloc[-1, 0], str) and df.iloc[-1, 0].startswith("Downloaded"):
        df = df.iloc[:-1]

    df.columns = [c.lower().replace("%", "pct_") for c in df.columns]
    df = df.rename(columns={"latest": "close", "time": "date"})

    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%y", errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()

    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def load_fred_series(filepath: str, value_col: str, out_col: str) -> pd.Series:
    """
    Load a FRED-style CSV with columns: observation_date, <value_col>.
    Returns a forward-filled Series indexed by observation_date.
    """
    df = pd.read_csv(filepath)
    df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    df = df.dropna(subset=["observation_date"]).set_index("observation_date").sort_index()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.ffill()
    return df[value_col].rename(out_col)


def load_main_dataframe() -> Optional[pd.DataFrame]:
    """
    Load:
      - QQQ (train+valid)
      - QQQ (holdout)
      - VXN (Nasdaq-100 implied vol)
      - VIX (1M)
      - VXV (3M)
    Build df_main with:
      - QQQ OHLCV
      - vxn / vix / vix3m
      - term structure features
      - calendar features (trading day in month forward/reverse)
    """
    print("Step 1: Loading and preparing data...")
    try:
        df_tv_raw = pd.read_csv(PATH_TRAIN_VALID)
        df_ho_raw = pd.read_csv(PATH_HOLDOUT)

        df_tv = clean_price_df(df_tv_raw)
        df_ho = clean_price_df(df_ho_raw)

        df_ho = df_ho[df_ho.index > df_tv.index.max()]
        df_main = pd.concat([df_tv, df_ho]).sort_index()

        # VXN
        try:
            vxn = load_fred_series(PATH_VXN, value_col="VXNCLS", out_col="vxn")
            df_main = df_main.join(vxn, how="left")
        except Exception as e:
            print(f"Warning: could not load VXNCLS: {e}")
            df_main["vxn"] = np.nan

        # Realized-vol proxy fallback for missing VXN (unchanged)
        rv_proxy = df_main["close"].pct_change().rolling(20).std() * np.sqrt(ANNUALIZATION) * 100.0 * 1.2
        df_main["vxn"] = df_main["vxn"].fillna(rv_proxy).fillna(25.0)

        # VIX 1M
        try:
            vix = load_fred_series(PATH_VIX, value_col="VIXCLS", out_col="vix")
            df_main = df_main.join(vix, how="left")
        except Exception as e:
            print(f"Warning: could not load VIXCLS: {e}")
            df_main["vix"] = np.nan

        # VIX 3M (VXV)
        try:
            vix3m = load_fred_series(PATH_VXV, value_col="VXVCLS", out_col="vix3m")
            df_main = df_main.join(vix3m, how="left")
        except Exception as e:
            print(f"Warning: could not load VXVCLS: {e}")
            df_main["vix3m"] = np.nan

        df_main["vix"] = df_main["vix"].ffill()
        df_main["vix3m"] = df_main["vix3m"].ffill()

        # Term structure features (unchanged)
        df_main["vix_ts_spread"] = df_main["vix3m"] - df_main["vix"]
        df_main["vix_ts_ratio"] = df_main["vix3m"] / (df_main["vix"] + EPS)
        df_main["vix_rel_vxn"] = df_main["vix"] / (df_main["vxn"] + EPS)

        # Calendar features (unchanged)
        df_main["year"] = df_main.index.year
        df_main["month"] = df_main.index.month
        df_main["tdm_fwd"] = df_main.groupby([df_main.index.year, df_main.index.month]).cumcount() + 1
        df_main["tdm_rev"] = df_main.groupby([df_main.index.year, df_main.index.month]).cumcount(ascending=False) + 1

        print("Data loaded successfully.")
        return df_main

    except Exception as e:
        print(f"Data error: {e}")
        return None


# ==============================================================================
# Core math helpers
# ==============================================================================


def calc_rsi(s: pd.Series, n: int) -> pd.Series:
    d = s.diff()
    g = (d.where(d > 0, 0.0)).ewm(alpha=1 / n, adjust=False).mean()
    l = (-d.where(d < 0, 0.0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100.0 - (100.0 / (1.0 + g / (l + EPS)))


def calc_rv(s: pd.Series, n: int) -> pd.Series:
    return s.pct_change().rolling(n).std() * np.sqrt(ANNUALIZATION)


def efficiency_ratio(s: pd.Series, n: int) -> pd.Series:
    return s.diff(n).abs() / (s.diff().abs().rolling(n).sum() + EPS)


# ==============================================================================
# Signal libraries
# ==============================================================================


def generate_full_signal_library(df_main: pd.DataFrame) -> pd.DataFrame:
    """
    Generate the full signal library (trend, mean-reversion, seasonality, vol/risk).
    IMPORTANT: No shifting is applied here; shift(1) is applied at backtest time.
    """
    print("Step 2: Generating full signal library...")

    idx = df_main.index
    s = pd.DataFrame(index=idx)

    c = df_main["close"]
    h = df_main["high"]
    l = df_main["low"]
    o = df_main["open"]
    vxn = df_main["vxn"]

    # --- Trend
    s["T_EMA"] = np.sign(c.ewm(span=50).mean() - c.ewm(span=200).mean())
    s["T_PriceVsMA"] = np.sign(c - c.rolling(200).mean())
    s["T_MACD"] = np.sign(c.ewm(span=12).mean() - c.ewm(span=26).mean())
    s["T_Mom12m"] = np.sign(c.diff(252)).rolling(21).mean()
    s["T_Donchian50"] = np.sign(c - c.rolling(50).mean())
    s["T_Vortex14"] = np.sign((h - l.shift(1)).abs().rolling(14).sum() - (l - h.shift(1)).abs().rolling(14).sum())
    s["T_Accel50"] = np.sign(c.diff(50).diff(20))
    s["T_Hurst"] = (efficiency_ratio(c, 126) > 0.3).astype(float) * 2.0 - 1.0

    # --- Mean reversion
    ibs = (c - l) / (h - l + EPS)
    s["M_IBS"] = np.where(ibs < 0.2, 1.0, np.where(ibs > 0.8, -1.0, 0.0))

    rsi2 = calc_rsi(c, 2)
    s["M_RSI2"] = np.where(rsi2 < 10.0, 1.0, np.where(rsi2 > 90.0, -1.0, 0.0))

    s["M_BBands"] = np.where(c < c.rolling(20).mean() - 2.0 * c.rolling(20).std(), 1.0, 0.0)
    s["M_5dLow"] = (c == c.rolling(5).min()).astype(float)
    s["M_VolRatio"] = (c.pct_change().rolling(5).std() > 2.5 * c.pct_change().rolling(100).std()).astype(float)

    gap = o / c.shift(1) - 1.0
    s["M_GapRev"] = np.where(gap > 0.005, -0.5, np.where(gap < -0.005, 0.5, 0.0))

    s["M_Streak3"] = np.where(
        (c.pct_change() < 0) & (c.pct_change().shift(1) < 0) & (c.pct_change().shift(2) < 0),
        1.0,
        0.0,
    )

    # --- Seasonality
    s["S_TOM"] = ((df_main["tdm_rev"] == 1) | (df_main["tdm_fwd"] <= 3)).astype(float)
    s["S_YearMom"] = (c > o.groupby(df_main["year"]).transform("first")).astype(float)

    s["S_Santa"] = (((df_main["month"] == 12) & (df_main["tdm_rev"] <= 5)) | ((df_main["month"] == 1) & (df_main["tdm_fwd"] <= 2))).astype(
        float
    )

    s["S_SellMay"] = np.where(df_main["month"].isin([5, 6, 7, 8, 9, 10]), -0.5, 1.0)

    s["S_WinDress"] = (
        (df_main["tdm_rev"] <= 3) & (df_main["month"].isin([3, 6, 9, 12])) & (c.pct_change(60) > 0)
    ).astype(float)

    # --- Volatility & risk
    rv10 = calc_rv(c, 10) * 100.0
    ratio_ivrv = vxn / (rv10 + EPS)
    s["V_IVRV"] = np.where(ratio_ivrv > 1.5, 1.0, np.where(ratio_ivrv < 0.8, -0.5, 0.0))

    def asym_ratio(x: np.ndarray) -> float:
        x = np.asarray(x)
        neg = x[x < 0]
        pos = x[x > 0]
        if len(pos) == 0:
            return 999.0
        return float(np.std(neg) / (np.std(pos) + EPS))

    s["V_Asym"] = c.pct_change().rolling(60).apply(asym_ratio, raw=True).le(1.5).astype(float)
    s["V_VXN_Trend"] = np.sign(vxn - vxn.rolling(50).mean()) * -1.0
    s["V_LowVol"] = (vxn < 18.0).astype(float)
    s["V_ROC"] = (vxn.pct_change(5) <= 0.20).astype(float)

    s["V_BearRally"] = ((c < c.rolling(200).mean()) & (calc_rsi(c, 4) < 20)).astype(float)
    s["V_BearShort"] = ((c < c.rolling(50).mean()) & (calc_rsi(c, 4) > 65)).astype(float) * -1.0

    s["V_VolCrush"] = (vxn.pct_change(5) < -0.20).astype(float)
    s["V_PanicBuy"] = np.where((vxn > 40.0) & (c > o), 1.5, 0.0)
    s["V_ER_Regime"] = (efficiency_ratio(c, 30) > 0.25).astype(float)

    ma200 = c.rolling(200).mean()
    ma20 = c.rolling(20).mean()
    er20 = efficiency_ratio(c, 20)

    cond_bull_chop = (c > ma200) & (vxn >= 18.0) & (er20 < 0.20)
    s["V_BullChopShort"] = np.where(cond_bull_chop, -0.5, 0.0)

    cond_bull_hv_break = (c > ma200) & (vxn > 27.0) & (c < ma20)
    s["V_BullHighVolBreak"] = np.where(cond_bull_hv_break, -1.0, 0.0)

    # Experimental hedges / vol structure filters
    ma50 = c.rolling(50).mean()
    er40 = efficiency_ratio(c, 40)
    er60 = efficiency_ratio(c, 60)
    vxn_ma20 = vxn.rolling(20).mean()
    roll_max_60 = c.rolling(60).max()

    s["X_BullChopFilter2"] = np.where(
        (c > ma200) & (vxn >= 18.0) & (er40 < 0.20) & ((c / (ma50 + EPS) - 1.0).abs() < 0.01),
        -0.5,
        0.0,
    )

    intraday_mid = (h + l) / 2.0
    s["X_TopFailure"] = np.where(
        (c < roll_max_60 * 0.99) & (c < ma50) & (vxn >= 20.0) & ((roll_max_60 / (c + EPS) - 1.0) > 0.03),
        -0.5,
        0.0,
    )

    s["X_VolUptrendDanger"] = np.where(
        (c > ma200) & (vxn > 20.0) & (vxn > vxn_ma20) & (vxn > vxn.shift(10)),
        -0.5,
        0.0,
    )

    s["X_BearTrendFollow"] = np.where((c < ma200) & (vxn > 25.0) & (er60 < 0.25), -1.0, 0.0)
    s["X_GapDownFollow"] = np.where((vxn > 27.0) & (gap < -0.02) & (c < intraday_mid), -0.5, 0.0)

    rsi4 = calc_rsi(c, 4)
    s["X_VolClusterLong"] = np.where((vxn > 35.0) & (vxn.rolling(5).min() > 30.0) & (rsi4 < 30.0), 1.0, 0.0)

    cond_low_vol_regime = vxn.rolling(30).max() < 18.0
    s["X_VolBreakoutFilter"] = np.where(
        cond_low_vol_regime & (vxn > 20.0) & (vxn.pct_change(5) > 0.20) & (c > ma200),
        -0.5,
        0.0,
    )

    vxn_range20 = vxn.rolling(20).max() - vxn.rolling(20).min()
    s["X_VolPlateau"] = np.where(
        (vxn >= 25.0) & (vxn <= 35.0) & (vxn_range20 < 5.0) & (c > ma200) & (er60 > 0.30),
        0.5,
        0.0,
    )

    corr20 = c.pct_change().rolling(20).corr(vxn.diff())
    s["X_PriceVolCorr"] = np.where(corr20 > 0.30, -0.5, np.where(corr20 < -0.30, 0.3, 0.0))

    df_signals = s.fillna(0.0)
    print(f"Generated {df_signals.shape[1]} signals.")
    return df_signals


def generate_vol_term_structure_signals(df_main: pd.DataFrame, mask_fit: pd.Series) -> pd.DataFrame:
    """Build candidate signals related to vol level and term structure.
    
    Args:
        df_main: Main dataframe with price and volatility data
        mask_fit: Boolean mask for fit period (train or train+valid) - quantiles computed only on this period
    """
    s = pd.DataFrame(index=df_main.index)

    vix = df_main["vix"].astype(float)
    vix3m = df_main["vix3m"].astype(float)
    vxn = df_main.get("vxn", pd.Series(index=df_main.index, dtype=float)).astype(float)

    # FIXED: Calculate quantiles only on FIT period to avoid lookahead bias
    vix_fit = vix[mask_fit].dropna()
    vix3m_fit = vix3m[mask_fit].dropna()
    if len(vix_fit) == 0:
        return s.fillna(0.0)

    vix_q20, vix_q50, vix_q70, vix_q80 = (
        vix_fit.quantile(0.20),
        vix_fit.quantile(0.50),
        vix_fit.quantile(0.70),
        vix_fit.quantile(0.80),
    )

    vix3m_q20 = vix3m_fit.quantile(0.20) if len(vix3m_fit) else np.nan
    vix3m_q80 = vix3m_fit.quantile(0.80) if len(vix3m_fit) else np.nan

    slope = vix3m - vix
    slope_fit = slope[mask_fit].dropna()
    slope_q20 = slope_fit.quantile(0.20) if len(slope_fit) else np.nan
    slope_q80 = slope_fit.quantile(0.80) if len(slope_fit) else np.nan

    slope_z20 = (slope - slope.rolling(20).mean()) / (slope.rolling(20).std() + 1e-8)

    slope_chg10 = slope - slope.shift(10)
    slope_chg_fit = slope_chg10[mask_fit].dropna()
    slope_chg_q20 = slope_chg_fit.quantile(0.20) if len(slope_chg_fit) else np.nan
    slope_chg_q80 = slope_chg_fit.quantile(0.80) if len(slope_chg_fit) else np.nan

    vix_z20 = (vix - vix.rolling(20).mean()) / (vix.rolling(20).std() + 1e-8)
    vix_pct_chg5 = vix.pct_change(5)
    vix_pct_chg20 = vix.pct_change(20)
    vix_diff_3d = vix - vix.shift(3)

    vol_gap = vxn - vix
    gap_fit = vol_gap[mask_fit].dropna()
    gap_q20 = gap_fit.quantile(0.20) if len(gap_fit) else np.nan
    gap_q80 = gap_fit.quantile(0.80) if len(gap_fit) else np.nan

    # VIX level
    s["VT_VIX_Low"] = np.where(vix <= vix_q20, 1.0, 0.0)
    s["VT_VIX_High"] = np.where(vix >= vix_q80, -1.0, 0.0)
    s["VT_VIX_Crush"] = np.where(vix_pct_chg5 <= -0.30, 1.0, 0.0)
    s["VT_VIX_SlowUp"] = np.where((vix_pct_chg20 >= 0.05) & (vix_pct_chg20 <= 0.25), -0.5, 0.0)
    s["VT_VIX_SlowDown"] = np.where((vix_pct_chg20 <= -0.05) & (vix_pct_chg20 >= -0.25), 0.5, 0.0)
    s["VT_VIX_ZHigh_MeanRev"] = np.where((vix_z20 >= 1.5) & (vix_diff_3d < 0), 1.0, 0.0)
    s["VT_VIX_ZLow_MeanRev"] = np.where((vix_z20 <= -1.5) & (vix_diff_3d > 0), -1.0, 0.0)

    # Term structure (static)
    s["VT_Contango_Steep"] = np.where(slope >= slope_q80, 1.0, 0.0)
    s["VT_Backwardation_Deep"] = np.where(slope <= slope_q20, -1.0, 0.0)
    s["VT_Contango_Carry"] = np.where((slope > 0) & (vix <= vix_q50), 1.0, 0.0)
    s["VT_Backwardation_Stress"] = np.where((slope < 0) & (vix >= vix_q70), -1.0, 0.0)

    # Term structure (dynamic)
    s["VT_TermSlope_ZHigh"] = np.where(slope_z20 >= 1.5, -0.5, 0.0)
    s["VT_TermSlope_ZLow"] = np.where(slope_z20 <= -1.5, 0.5, 0.0)
    s["VT_TermSlope_Flat"] = np.where(slope_z20.abs() <= 0.25, -0.3, 0.0)
    s["VT_TermSlope_Steepening"] = np.where((slope_chg10 >= slope_chg_q80) & (vix_z20 > 0), -0.5, 0.0)
    s["VT_TermSlope_Flattening"] = np.where((slope_chg10 <= slope_chg_q20) & (vix_z20 < 0), 0.5, 0.0)

    # VIX3M level
    s["VT_VIX3M_Level_High"] = np.where(vix3m >= vix3m_q80, -0.5, 0.0)
    s["VT_VIX3M_Level_Low"] = np.where(vix3m <= vix3m_q20, 0.5, 0.0)

    # VXN - VIX gap
    s["VT_VolGap_QQQRich"] = np.where(vol_gap >= gap_q80, -0.5, 0.0)
    s["VT_VolGap_QQQCheap"] = np.where(vol_gap <= gap_q20, 0.5, 0.0)

    s = s.fillna(0.0)
    print(f"[VOL-TS] Generated {s.shape[1]} vol/term-structure candidate signals.")
    return s


# ==============================================================================
# Candidate signal set (kept identical)
# ==============================================================================


def generate_candidate_signals(df_main: pd.DataFrame) -> pd.DataFrame:
    """Bull x Medium-vol focused candidate signals (kept identical to your logic)."""
    s = pd.DataFrame(index=df_main.index)

    c = df_main["close"]
    h = df_main["high"]
    l = df_main["low"]
    o = df_main["open"]
    vxn = df_main["vxn"]
    vix = df_main.get("vix", pd.Series(index=df_main.index, dtype=float))
    vix3 = df_main.get("vix3m", pd.Series(index=df_main.index, dtype=float))

    ma20 = c.rolling(20).mean()
    ma50 = c.rolling(50).mean()
    ma200 = c.rolling(200).mean()
    roll_max_60 = c.rolling(60).max()

    rv20 = c.pct_change().rolling(20).std() * np.sqrt(ANNUALIZATION)
    rv60 = c.pct_change().rolling(60).std() * np.sqrt(ANNUALIZATION)

    dd_60 = c / (roll_max_60 + EPS) - 1.0

    rsi4 = calc_rsi(c, 4)
    rsi8 = calc_rsi(c, 8)
    ibs = (c - l) / (h - l + EPS)

    er20 = efficiency_ratio(c, 20)
    er40 = efficiency_ratio(c, 40)

    cond_bull = c > ma200
    cond_bear = c <= ma200

    cond_lowV = vxn < 18.0
    cond_medV = (vxn >= 18.0) & (vxn <= 27.0)
    cond_highV = vxn > 27.0

    # A) Mean reversion
    cond = cond_bull & cond_medV
    s["M_BullMed_DipRSI"] = np.where(
        cond & (c < ma20 * 0.985) & (rsi4 < 35),
        1.0,
        np.where(cond & (c > ma20 * 1.01) & (rsi4 > 65), -0.5, 0.0),
    )

    cond = cond_bull & cond_medV
    s["M_BullMed_IBS_RSI_Combo"] = np.where(
        cond & (ibs < 0.25) & (rsi4 < 35),
        1.0,
        np.where(cond & (ibs > 0.75) & (rsi4 > 65), -0.5, 0.0),
    )

    cond = cond_bull & cond_medV
    s["M_BullMed_DrawdownFader"] = np.where(cond & (dd_60 < -0.15) & (dd_60 > -0.30) & (rsi4 < 35), 1.0, 0.0)

    cond = cond_bull & cond_medV
    s["M_BullMed_ShortDip"] = np.where(cond & (dd_60 < -0.07) & (dd_60 > -0.20) & (rsi8 < 40), 0.7, 0.0)

    gap = o / c.shift(1) - 1.0
    mid = (h + l) / 2.0
    cond = cond_bull & cond_medV
    s["M_BullMed_SmallGapFade"] = np.where(
        cond & (gap > 0.004) & (gap < 0.015) & (c < mid),
        -0.7,
        np.where(cond & (gap < -0.004) & (gap > -0.015) & (c > mid), 0.7, 0.0),
    )

    cond = cond_bull & cond_lowV
    s["M_BullLow_PullbackBuy"] = np.where(
        cond & (c < ma20 * 0.99) & (rsi4 < 40),
        0.8,
        np.where(cond & (c > ma20 * 1.01) & (rsi4 > 70), -0.3, 0.0),
    )

    big_down = c.pct_change() < -0.04
    rebound = c.pct_change(2) > 0.02
    s["M_CrashBounce_VolHigh"] = np.where(cond_highV & big_down.shift(1) & rebound & (rsi4 < 45), 0.8, 0.0)

    cond = cond_bear & cond_highV
    s["M_BearHigh_ShortRallyFade"] = np.where(cond & (c.pct_change(3) > 0.04) & (rsi4 > 60), -1.0, 0.0)

    # B) Trend
    ema10 = c.ewm(span=10).mean()
    ema30 = c.ewm(span=30).mean()

    cond = cond_bull & cond_medV
    s["T_BullMed_ShortTermTrend"] = np.where(
        cond & (ema10 > ema30) & (c > ma50),
        1.0,
        np.where(cond & (ema10 < ema30) & (c < ma50), -0.5, 0.0),
    )

    cond = cond_bull & cond_medV
    s["T_BullMed_ER_Filter"] = np.where(cond & (er20 < 0.15), -0.7, 0.0)

    cond = cond_bull & cond_medV
    s["T_BullMed_TopFilter"] = np.where(cond & (c > roll_max_60 * 0.995) & (rsi8 > 65), -0.7, 0.0)

    cond = cond_bull & cond_lowV
    s["T_BullLow_StrongTrendBoost"] = np.where(
        cond & (er40 > 0.35) & (c > ma50) & (c > roll_max_60 * 0.98), 1.0, 0.0
    )

    breakout = (c.shift(5) < roll_max_60.shift(5) * 0.99) & (c > roll_max_60.shift(5) * 1.01)
    pullback = c < c.rolling(5).max() * 0.98

    cond = cond_bull & cond_medV
    s["T_BullMed_PullbackAfterBreakout"] = np.where(cond & breakout & pullback & (rsi4 < 55), 0.8, 0.0)

    s["T_BearMed_TrendOff"] = np.where(cond_bear & cond_medV, -0.5, 0.0)

    # C) Vol / term structure
    ivrv = vxn / (rv20 * 100.0 + EPS)
    cond = cond_bull & cond_medV
    s["V_BullMed_IVRV_SweetSpot"] = np.where(
        cond & (ivrv > 1.1) & (ivrv < 1.8),
        1.0,
        np.where(cond & (ivrv > 2.2), -0.7, 0.0),
    )

    slope = df_main.get("vix_ts_spread", vix3 - vix)
    ratio = df_main.get("vix_ts_ratio", vix3 / (vix + EPS))

    cond = cond_bull & cond_medV
    s["V_BullMed_TermSlope_Friendly"] = np.where(
        cond & (slope > 0) & (slope < 3) & (ratio > 1.0) & (ratio < 1.2), 0.7, 0.0
    )

    vxn_chg5 = vxn.pct_change(5)
    s["V_BullMed_VolSpikeKill"] = np.where((cond_bull & cond_medV) & (vxn_chg5 > 0.25), -1.0, 0.0)

    rv_ratio = rv20 / (rv60 + EPS)
    cond = cond_bull & cond_medV
    s["V_BullMed_RVCluster"] = np.where(
        cond & (rv_ratio > 0.7) & (rv_ratio < 1.3) & (rv20 < 0.25),
        0.5,
        np.where(cond & (rv20 > 0.35), -0.5, 0.0),
    )

    s["V_BearHigh_TS_StressOff"] = np.where((cond_bear & cond_highV) & (slope < -3), -0.7, 0.0)

    vxn_range_20 = vxn.rolling(20).max() - vxn.rolling(20).min()
    vxn_chg_3 = vxn.pct_change(3)
    s["V_Any_VolCompressionBreakout"] = np.where((vxn_range_20 < 3.0) & (vxn_chg_3 > 0.20), -0.5, 0.0)

    return s.fillna(0.0)


# ==============================================================================
# Signal sets, groups
# ==============================================================================

VOLTS_CORE_SIGNALS = [
    "VT_VolGap_QQQCheap",
    "VT_VIX_SlowDown",
    "VT_TermSlope_ZLow",
    "VT_VIX_Low",
    "VT_VIX3M_Level_Low",
]

CORE_SIGNALS = [
    # Trend
    "T_EMA",
    "T_Mom12m",
    "T_PriceVsMA",
    "T_MACD",
    "T_Donchian50",
    "T_Vortex14",
    # Mean reversion
    "M_IBS",
    "M_RSI2",
    "M_BBands",
    "M_5dLow",
    "M_Streak3",
    "M_GapRev",
    # Seasonality
    "S_TOM",
    "S_YearMom",
    "S_Santa",
    "S_SellMay",
    # Vol / risk
    "V_IVRV",
    "V_LowVol",
    "V_BearRally",
    "V_Asym",
    "V_ER_Regime",
    "V_PanicBuy",
    "V_ROC",
    "X_PriceVolCorr",
    "X_VolClusterLong",
] + VOLTS_CORE_SIGNALS

GROUPS: Dict[str, List[str]] = {
    "Trend": [
        "T_EMA",
        "T_Mom12m",
        "T_PriceVsMA",
        "T_MACD",
        "T_Donchian50",
        "T_Vortex14",
        "T_BullMed_ER_Filter",
    ],
    "MeanRev": [
        "M_IBS",
        "M_RSI2",
        "M_BBands",
        "M_5dLow",
        "M_Streak3",
        "M_GapRev",
        "M_BearHigh_ShortRallyFade",
    ],
    "Seasonality": ["S_TOM", "S_YearMom", "S_Santa", "S_SellMay"],
    "VolRisk": [
        "V_IVRV",
        "V_LowVol",
        "V_BearRally",
        "V_Asym",
        "V_ER_Regime",
        "V_PanicBuy",
        "V_ROC",
        "X_PriceVolCorr",
        "X_VolClusterLong",
        "V_BearHigh_TS_StressOff",
    ]
    + VOLTS_CORE_SIGNALS,
}


def get_group_signal_list(groups: Dict[str, List[str]], families: Optional[List[str]] = None) -> List[str]:
    if families is None:
        families = list(groups.keys())
    seen = set()
    out: List[str] = []
    for fam in families:
        for col in groups.get(fam, []):
            if col not in seen:
                seen.add(col)
                out.append(col)
    return out


# ==============================================================================
# Backtest helpers: returns, masks, stats, normalization, aggregation
# ==============================================================================


def get_returns(df_main: pd.DataFrame) -> pd.Series:
    return df_main["close"].pct_change().fillna(0.0)


def get_masks(index: pd.DatetimeIndex):
    idx = pd.DatetimeIndex(index)
    mask_train = (idx >= pd.Timestamp("2000-01-01")) & (idx <= pd.Timestamp("2015-12-31"))
    mask_valid = (idx >= pd.Timestamp("2016-01-01")) & (idx <= pd.Timestamp("2021-12-31"))
    mask_holdout = (idx >= pd.Timestamp("2022-01-01")) & (idx <= pd.Timestamp("2025-06-30"))
    return mask_train, mask_valid, mask_holdout


def get_fit_mask(index: pd.DatetimeIndex, fit_on: str = "train"):
    mask_train, mask_valid, mask_holdout = get_masks(index)
    mask_fit = (mask_train | mask_valid) if fit_on == "trainval" else mask_train
    return mask_train, mask_valid, mask_holdout, mask_fit


def compute_perf_stats(r: pd.Series, mask: pd.Series) -> dict:
    r_sub = r[mask].dropna()
    if len(r_sub) == 0:
        return {"ann_ret": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_dd": np.nan, "n_days": 0}

    mu = r_sub.mean()
    sigma = r_sub.std()

    ann_ret = mu * ANNUALIZATION
    ann_vol = sigma * np.sqrt(ANNUALIZATION)
    sharpe = 0.0 if sigma < 1e-8 else (mu / sigma) * np.sqrt(ANNUALIZATION)

    r_eq = r_sub.clip(lower=-0.99)
    equity = (1.0 + r_eq).cumprod()
    dd = equity / equity.cummax() - 1.0
    max_dd = dd.min()

    return {"ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "max_dd": max_dd, "n_days": len(r_sub)}


def normalize_signals_by_pnl_vol(
    df_signals: pd.DataFrame,
    returns: pd.Series,
    mask_fit: pd.Series,
    min_sigma: float = 1e-8,
) -> pd.DataFrame:
    """
    Per-signal scaling using the volatility of (signal.shift(1) * returns) on the fit sample.
    """
    df_norm = pd.DataFrame(index=df_signals.index)
    for col in df_signals.columns:
        expo_raw = df_signals[col].shift(1)
        pnl = expo_raw * returns
        sigma_fit = pnl[mask_fit].std()
        scale = 0.0 if (sigma_fit is None or np.isnan(sigma_fit) or sigma_fit < min_sigma) else (1.0 / sigma_fit)
        df_norm[col] = df_signals[col] * scale
    return df_norm


def compute_signal_sharpes(
    df_signals_norm: pd.DataFrame,
    returns: pd.Series,
    mask_fit: pd.Series,
    min_sigma: float = 1e-8,
) -> dict:
    sharpes = {}
    for col in df_signals_norm.columns:
        expo = df_signals_norm[col].shift(1)
        pnl = expo * returns
        r_fit = pnl[mask_fit].dropna()
        if len(r_fit) == 0:
            sharpes[col] = 0.0
            continue
        mu = r_fit.mean()
        sigma = r_fit.std()
        sharpes[col] = 0.0 if sigma < min_sigma else (mu / sigma) * np.sqrt(ANNUALIZATION)
    return sharpes


def compute_group_exposures(
    df_signals_norm: pd.DataFrame,
    sharpes: dict,
    groups: dict,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """
    Build family exposures as weighted sums of normalized signal columns,
    where within-family weights are proportional to max(Sharpe, 0)^alpha.
    """
    df_groups = pd.DataFrame(index=df_signals_norm.index)
    for g, cols in groups.items():
        sh = np.array([max(sharpes.get(c, 0.0), 0.0) for c in cols], dtype=float)
        if np.all(sh <= 0):
            w = np.ones(len(cols), dtype=float) / max(len(cols), 1)
        else:
            w = np.power(sh, alpha)
            w = w / (w.sum() + EPS)

        group_expo = np.zeros(len(df_signals_norm), dtype=float)
        for w_i, col in zip(w, cols):
            group_expo += w_i * df_signals_norm[col].values
        df_groups[g] = group_expo

    return df_groups


def backtest_portfolio_with_vol_target(
    exposures: pd.Series,
    returns: pd.Series,
    target_vol_annual: float = 0.15,
    vol_window: int = 60,
    expo_min: float = -1.0,
    expo_max: float = 1.5,
    max_scale: float = 10.0,
    cap_series: Optional[pd.Series] = None,
) -> Tuple[pd.Series, pd.Series]:
    """
    Apply vol targeting on the *portfolio* PnL, then clip exposure (optionally with a
    time-varying symmetric cap per day).
    """
    e_base = exposures.shift(1).fillna(0.0)
    r_base = e_base * returns

    daily_target_vol = target_vol_annual / np.sqrt(ANNUALIZATION)
    vol_prev = r_base.rolling(vol_window).std().shift(1)
    scale = (daily_target_vol / (vol_prev + 1e-8)).clip(0.0, max_scale)

    e_scaled = e_base * scale

    if cap_series is not None:
        cap_s = pd.Series(cap_series, index=e_scaled.index).reindex(e_scaled.index).fillna(expo_max).astype(float)
        e_final = e_scaled.clip(lower=-cap_s, upper=cap_s).clip(lower=expo_min, upper=expo_max)
    else:
        e_final = e_scaled.clip(lower=expo_min, upper=expo_max)

    r_port = e_final * returns
    return r_port, e_final


def print_subperiod_stats(r_port: pd.Series):
    subperiods = [
        ("2000-01-01", "2007-12-31"),
        ("2008-01-01", "2012-12-31"),
        ("2013-01-01", "2016-12-31"),
        ("2017-01-01", "2020-12-31"),
        ("2021-01-01", "2025-06-30"),
    ]
    print("\n=== SUBPERIOD STATS (PORTFOLIO RETURNS) ===")
    for start, end in subperiods:
        mask = (r_port.index >= start) & (r_port.index <= end)
        st = compute_perf_stats(r_port, mask)
        print(
            f"{start} -> {end} | Sharpe: {st['sharpe']:.2f} | "
            f"Ret: {100*st['ann_ret']:.1f}% | Vol: {100*st['ann_vol']:.1f}% | "
            f"MaxDD: {100*st['max_dd']:.1f}% | N={st['n_days']}"
        )


# ==============================================================================
# VolTS gate (throttle) — unchanged behavior when columns exist
# ==============================================================================


def build_volts_gate_series(df_signals: pd.DataFrame) -> pd.Series:
    # These columns exist in your pipeline (df_volts + joins); keep logic identical.
    cheap = (df_signals["VT_VolGap_QQQCheap"] > 0).astype(int)
    vix_low = (df_signals["VT_VIX_Low"] > 0).astype(int)
    vix3m_low = (df_signals["VT_VIX3M_Level_Low"] > 0).astype(int)
    slowdown = (df_signals["VT_VIX_SlowDown"] > 0).astype(int)

    score = cheap + vix_low + vix3m_low + slowdown

    gate = pd.Series(1.0, index=df_signals.index, dtype=float)
    gate[score <= 0] = 0.60
    gate[score == 1] = 0.80
    gate[score == 2] = 1.00
    gate[score == 3] = 1.20
    gate[score >= 4] = 1.40
    return gate


# ==============================================================================
# Buckets, rolling Sharpe, diagnostics
# ==============================================================================

BUCKET_KEYS = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]


def build_bull_vol_buckets(df_main: pd.DataFrame, mask_fit: pd.Series):
    c = df_main["close"]
    vxn = df_main["vxn"]
    ma200 = c.rolling(200).mean()

    trend_state = (c > ma200).astype(int)

    v_fit = vxn[mask_fit & ma200.notna()]
    q1 = v_fit.quantile(1 / 3) if len(v_fit) else np.nan
    q2 = v_fit.quantile(2 / 3) if len(v_fit) else np.nan

    def bucketize(x: float) -> int:
        if np.isnan(x) or np.isnan(q1) or np.isnan(q2):
            return -1
        if x <= q1:
            return 0
        if x <= q2:
            return 1
        return 2

    v_bucket = vxn.apply(bucketize)
    return {"trend_state": trend_state, "v_bucket": v_bucket, "q1": q1, "q2": q2}


def compute_rolling_sharpe(r: pd.Series, window: int = 756) -> pd.Series:
    r = r.dropna()
    if len(r) < window:
        return pd.Series(index=r.index, dtype=float)
    mu_roll = r.rolling(window).mean()
    sigma_roll = r.rolling(window).std()
    return (mu_roll / (sigma_roll + 1e-8)) * np.sqrt(ANNUALIZATION)


def print_rolling_sharpe_summary(r_port: pd.Series, r_bh: pd.Series, window: int = 756):
    rs_port = compute_rolling_sharpe(r_port, window=window)
    rs_bh = compute_rolling_sharpe(r_bh, window=window)

    def summarize(name: str, rs: pd.Series):
        rs = rs.dropna()
        if len(rs) == 0:
            print(f"{name}: rolling window too short for stats.")
            return
        print(
            f"{name:10s} rolling Sharpe (window={window}d) | "
            f"median={rs.median():5.2f} | 10%={rs.quantile(0.10):5.2f} | "
            f"min={rs.min():5.2f} | max={rs.max():5.2f}"
        )

    print("\n=== ROLLING 3Y SHARPE (DAILY RETURNS) ===")
    summarize("Portfolio", rs_port)
    summarize("Buy&Hold", rs_bh)


def analyze_portfolio_weak_zones(
    df_main: pd.DataFrame,
    r_port: pd.Series,
    fit_on: str = "train",
    model_name: str = "M2",
    align_with_signal_shift: bool = True,
):
    idx = df_main.index
    mask_train, mask_valid, _, mask_fit = get_fit_mask(idx, fit_on)

    buckets = build_bull_vol_buckets(df_main, mask_fit)
    trend_state = buckets["trend_state"]
    v_bucket = buckets["v_bucket"]
    q1, q2 = buckets["q1"], buckets["q2"]

    if align_with_signal_shift:
        tr_ref = trend_state.shift(1)
        vb_ref = v_bucket.shift(1)
    else:
        tr_ref = trend_state
        vb_ref = v_bucket

    print(f"\n=== {model_name}: CONDITIONAL STATS BY TREND x VXN BUCKET ===")
    print(f"(fit_on={fit_on}, VXN tertiles on FIT: q1={q1:.2f}, q2={q2:.2f}, align_shift={align_with_signal_shift})\n")

    def fmt(tr_state: int) -> str:
        return "Bull" if tr_state == 1 else "Bear/Side"

    for period_name, mask_period in [("TRAIN", mask_train), ("VALID", mask_valid)]:
        print(f"--- {model_name} {period_name} ---")
        for tr_state in [0, 1]:
            for b in [0, 1, 2]:
                m = mask_period & (tr_ref == tr_state) & (vb_ref == b)
                st = compute_perf_stats(r_port, m)
                if st["n_days"] > 50:
                    vlabel = {0: "LowV", 1: "MedV", 2: "HighV"}[b]
                    print(
                        f"  {fmt(tr_state):9s} x {vlabel:5s} | "
                        f"Sharpe: {st['sharpe']:5.2f} | Ret: {st['ann_ret']*100:5.1f}% | "
                        f"Vol: {st['ann_vol']*100:5.1f}% | MaxDD: {st['max_dd']*100:5.1f}% | N={st['n_days']}"
                    )
        print()


def compute_family_pnl(df_groups: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
    pnl = pd.DataFrame(index=df_groups.index)
    for g in df_groups.columns:
        pnl[g] = df_groups[g].shift(1).fillna(0.0) * returns
    return pnl


def print_family_perf_stats(df_groups: pd.DataFrame, returns: pd.Series, fit_on: str = "train"):
    mask_train, mask_valid, mask_holdout, _ = get_fit_mask(df_groups.index, fit_on)
    pnl_fam = compute_family_pnl(df_groups, returns)

    print("\n=== FAMILY-LEVEL PERFORMANCE (RAW, NO VOL TARGET) ===")
    for period_name, mask in [("TRAIN", mask_train), ("VALID", mask_valid), ("HOLDOUT", mask_holdout)]:
        print(f"\n--- {period_name} ---")
        for g in pnl_fam.columns:
            st = compute_perf_stats(pnl_fam[g], mask)
            print(
                f"  {g:12s} | Sharpe: {st['sharpe']:5.2f} | Ret: {st['ann_ret']*100:6.2f}% | "
                f"Vol: {st['ann_vol']*100:6.2f}% | MaxDD: {st['max_dd']*100:6.2f}% | N={st['n_days']}"
            )


# ==============================================================================
# Family scaling, hedge overlay, regimes & bucket weights
# ==============================================================================


def apply_family_bucket_scaling(
    df_main: pd.DataFrame,
    df_groups: pd.DataFrame,
    mask_fit: pd.Series,
    scaling_rules: dict,
) -> pd.DataFrame:
    buckets = build_bull_vol_buckets(df_main, mask_fit)
    trend_state = buckets["trend_state"]
    v_bucket = buckets["v_bucket"]

    df_scaled = df_groups.copy()
    for (ts, vb), fam_scales in scaling_rules.items():
        m = (trend_state == ts) & (v_bucket == vb)
        if m.sum() == 0:
            continue
        for fam, scale in fam_scales.items():
            if fam in df_scaled.columns:
                df_scaled.loc[m, fam] *= scale
    return df_scaled


def build_hedge_overlay(
    df_main: pd.DataFrame,
    df_hedge_norm: pd.DataFrame,
    mask_fit: pd.Series,
    returns: pd.Series,
    beta_map: Optional[dict] = None,
):
    beta_map = {} if beta_map is None else beta_map

    buckets = build_bull_vol_buckets(df_main, mask_fit)
    trend_state = buckets["trend_state"]
    v_bucket = buckets["v_bucket"]

    mask_hedge_zone = (((trend_state == 1) & (v_bucket == 1)) | ((trend_state == 1) & (v_bucket == 2)) | ((trend_state == 0) & (v_bucket == 2)))

    hedge_raw = pd.Series(0.0, index=df_main.index)
    for col in df_hedge_norm.columns:
        hedge_raw += beta_map.get(col, 1.0) * df_hedge_norm[col]

    mask_target_fit = mask_fit & mask_hedge_zone
    r_hedge = hedge_raw.shift(1) * returns
    r_sub = r_hedge[mask_target_fit].dropna()

    if len(r_sub) == 0 or r_sub.std() < 1e-8:
        sign_dir = 0.0
    else:
        s_plus = (r_sub.mean() / r_sub.std()) * np.sqrt(ANNUALIZATION)
        s_minus = (-r_sub.mean() / r_sub.std()) * np.sqrt(ANNUALIZATION)
        sign_dir = 1.0 if s_plus >= s_minus else -1.0

    hedge_expo = (sign_dir * hedge_raw).where(mask_hedge_zone, 0.0)
    return hedge_expo, mask_hedge_zone, sign_dir


def build_two_regime_series(df_main: pd.DataFrame, mask_fit: pd.Series, vxn_quantile: float = 0.75) -> pd.Series:
    c = df_main["close"]
    vxn = df_main["vxn"]
    ma200 = c.rolling(200).mean()

    vxn_fit = vxn[mask_fit & ma200.notna()]
    thresh = np.nan if len(vxn_fit) == 0 else np.nanquantile(vxn_fit, vxn_quantile)

    bull = c > ma200
    low_vol = vxn <= thresh
    regime_normal = bull & low_vol

    regime = pd.Series(index=df_main.index, dtype=int)
    regime[regime_normal] = 0
    regime[~regime_normal] = 1

    print(f"\nRegime thresholds: VXN quantile {vxn_quantile:.2f} on FIT = {thresh:.2f}")
    print("Regime distribution (all data):")
    print(regime.value_counts().sort_index())

    return regime


def compute_family_sharpes_by_regime(
    df_groups: pd.DataFrame,
    returns: pd.Series,
    mask_fit: pd.Series,
    regime: pd.Series,
    min_sigma: float = 1e-8,
) -> dict:
    sharpes = {}
    regime_used = regime.shift(1)

    for fam in df_groups.columns:
        pnl = df_groups[fam].shift(1) * returns
        for r in [0, 1]:
            m = mask_fit & (regime_used == r)
            r_sub = pnl[m].dropna()
            if len(r_sub) == 0:
                sharpes[(fam, r)] = 0.0
                continue
            mu = r_sub.mean()
            sigma = r_sub.std()
            sharpes[(fam, r)] = 0.0 if sigma < min_sigma else (mu / sigma) * np.sqrt(ANNUALIZATION)

    return sharpes


def derive_family_weights_per_regime_from_sharpes(
    sharpes_fr: dict,
    family_names: List[str],
    alpha: float = 1.0,
    min_w: float = 0.05,
    max_w: float = 0.60,
) -> dict:
    weights_regime = {}
    for r in [0, 1]:
        scores = np.array([max(sharpes_fr.get((g, r), 0.0), 0.10) for g in family_names], dtype=float)
        w_raw = np.power(scores, alpha)
        if w_raw.sum() <= 0:
            w_raw = np.ones_like(w_raw) / len(w_raw)
        else:
            w_raw = w_raw / w_raw.sum()

        w_clamped = np.clip(w_raw, min_w, max_w)
        if w_clamped.sum() <= 0:
            w_final = np.ones_like(w_clamped) / len(w_clamped)
        else:
            w_final = w_clamped / w_clamped.sum()

        weights_regime[r] = w_final
    return weights_regime


def compute_family_sharpes_by_bucket(
    df_groups: pd.DataFrame,
    returns: pd.Series,
    mask_fit: pd.Series,
    trend_state: pd.Series,
    v_bucket: pd.Series,
    align_with_signal_shift: bool = True,
) -> dict:
    m = pd.Series(mask_fit, index=df_groups.index).fillna(False).astype(bool)

    expo = df_groups.copy()
    ts = trend_state.copy()
    vb = v_bucket.copy()

    if align_with_signal_shift:
        expo = expo.shift(1)
        ts = ts.shift(1)
        vb = vb.shift(1)

    out = {}
    for fam in expo.columns:
        r_fam = expo[fam] * returns
        for ts_k, vb_k in BUCKET_KEYS:
            mb = m & (ts == ts_k) & (vb == vb_k)
            out[(fam, ts_k, vb_k)] = compute_perf_stats(r_fam, mb)["sharpe"]

    return out


def derive_family_weights_per_bucket_from_sharpes(
    sharpes_fb: dict,
    family_names: List[str],
    alpha: float = 1.0,
    min_w: float = 0.05,
    max_w: float = 0.70,
    score_floor: float = 0.10,
) -> dict:
    """Convert conditional Sharpes into per-bucket family weights."""
    weights_bucket = {}
    for ts, vb in BUCKET_KEYS:
        scores = np.asarray([max(sharpes_fb.get((fam, ts, vb), 0.0), score_floor) for fam in family_names], dtype=float)
        w_raw = np.power(scores, alpha)
        if w_raw.sum() <= 0:
            w_raw = np.ones_like(w_raw) / len(w_raw)
        else:
            w_raw = w_raw / w_raw.sum()

        w_clamped = np.clip(w_raw, min_w, max_w)
        if w_clamped.sum() <= 0:
            w_final = np.ones_like(w_clamped) / len(w_clamped)
        else:
            w_final = w_clamped / w_clamped.sum()

        weights_bucket[(ts, vb)] = w_final

    return weights_bucket


# ==============================================================================
# M2 runner (model stays the same; reporting removed from inside)
# ==============================================================================


def run_M2_two_regimes_with_hedge(
    df_main: pd.DataFrame,
    df_signals_full: pd.DataFrame,
    target_vol_annual: float = 0.15,
    vol_window: int = 60,
    fit_on: str = "train",
    lambda_grid: Optional[List[float]] = None,
    family_keep: Optional[List[str]] = None,
    scaling_rules_override: Optional[dict] = None,
    scale_map_override: Optional[dict] = None,
    quiet: bool = False,
    scale_bull_medv: Optional[float] = None,
    use_bucket_family_weights: bool = True,
    bucket_weight_alpha: float = 1.0,
    bucket_min_w: float = 0.05,
    bucket_max_w: float = 0.70,
    bucket_score_floor: float = 0.10,
) -> dict:
    """
    M2:
      - Family groups are controlled by GROUPS
      - Bucket-specific intra-family scaling (scaling_rules)
      - Bucket throttle (scale_map) + VolTS gate
      - Hedge overlay + grid search on lambda_hedge (validated on TRAIN/VALID)
    """
    lambda_grid = [0.0] if lambda_grid is None else lambda_grid
    family_keep = ["Trend", "MeanRev", "VolRisk"] if family_keep is None else family_keep

    returns = get_returns(df_main)
    mask_train, mask_valid, mask_holdout, mask_fit = get_fit_mask(df_main.index, fit_on)

    # --- Core signals -> normalized -> group exposures
    families_all = list(GROUPS.keys())
    core_cols = [c for c in get_group_signal_list(GROUPS, families_all) if c in df_signals_full.columns]
    df_signals_core = df_signals_full[core_cols].copy()

    df_signals_norm = normalize_signals_by_pnl_vol(df_signals_core, returns, mask_fit)
    sharpes_fit = compute_signal_sharpes(df_signals_norm, returns, mask_fit)

    df_groups = compute_group_exposures(df_signals_norm, sharpes_fit, GROUPS, alpha=1.0)[family_keep]
    family_names = list(df_groups.columns)

    gate_series = build_volts_gate_series(df_signals_full).reindex(df_main.index).fillna(1.0)

    # --- Bucket-specific family scaling
    scaling_rules = (
        {k: dict(v) for k, v in scaling_rules_override.items()}
        if scaling_rules_override is not None
        else {
            (0, 1): {"Trend": 0.8, "VolRisk": 0.2},
            (1, 1): {"VolRisk": 0.7},
            (1, 2): {"VolRisk": 1.1},
        }
    )
    df_groups = apply_family_bucket_scaling(df_main, df_groups, mask_fit, scaling_rules)

    # --- Regimes and regime weights
    regime = build_two_regime_series(df_main, mask_fit, vxn_quantile=0.75)
    sharpes_fr = compute_family_sharpes_by_regime(df_groups, returns, mask_fit, regime)
    weights_regimes = derive_family_weights_per_regime_from_sharpes(sharpes_fr, family_names, alpha=1.0)

    # --- Buckets for throttle & caps
    buckets = build_bull_vol_buckets(df_main, mask_fit)
    trend_state = buckets["trend_state"]
    v_bucket = buckets["v_bucket"]

    # Bucket sample size (fit) for shrinkage
    ts_fit = trend_state.shift(1)
    vb_fit = v_bucket.shift(1)
    bucket_counts = {(ts, vb): int((mask_fit & (ts_fit == ts) & (vb_fit == vb)).sum()) for ts, vb in BUCKET_KEYS}

    # Exposure caps per bucket (aligned to exposure shift)
    cap_default = 1.5
    cap_map = {
        (1, 0): 1.5,  # Bull LowV
        (1, 1): 1.5,  # Bull MedV
        (1, 2): 1.2,  # Bull HighV
        (0, 0): 1.0,  # Bear LowV
        (0, 1): 0.8,  # Bear MedV
        (0, 2): 0.6,  # Bear HighV
    }
    cap_series = pd.Series(index=df_main.index, dtype=float)
    for t in df_main.index:
        ts = int(trend_state.loc[t]) if not pd.isna(trend_state.loc[t]) else 0
        vb = int(v_bucket.loc[t]) if not pd.isna(v_bucket.loc[t]) else 0
        cap_series.loc[t] = cap_map.get((ts, vb), cap_default)
    cap_series = cap_series.shift(1).fillna(cap_default)

    # Learn bucket family weights (optional)
    weights_bucket = None
    if use_bucket_family_weights:
        sharpes_fb = compute_family_sharpes_by_bucket(
            df_groups=df_groups,
            returns=returns,
            mask_fit=mask_fit,
            trend_state=trend_state,
            v_bucket=v_bucket,
            align_with_signal_shift=True,
        )
        weights_bucket = derive_family_weights_per_bucket_from_sharpes(
            sharpes_fb=sharpes_fb,
            family_names=family_names,
            alpha=bucket_weight_alpha,
            min_w=bucket_min_w,
            max_w=bucket_max_w,
            score_floor=bucket_score_floor,
        )

    # Bucket throttle (scale_map)
    scale_map = (
        dict(scale_map_override)
        if scale_map_override is not None
        else {
            (1, 0): 1.10,  # Bull LowV
            (1, 1): 0.35,  # Bull MedV
            (1, 2): 0.50,  # Bull HighV
            (0, 0): 0.60,  # Bear LowV
            (0, 1): 0.35,  # Bear MedV
            (0, 2): 0.15,  # Bear HighV
        }
    )
    if scale_bull_medv is not None:
        scale_map[(1, 1)] = float(scale_bull_medv)

    # --- Build base exposure time series
    regime_used = regime.shift(1).fillna(0).astype(int)
    base_expo_raw = pd.Series(index=df_groups.index, dtype=float)
    weights_used_daily = pd.DataFrame(index=df_groups.index, columns=family_names, dtype=float)

    for t in df_groups.index:
        expo_t = df_groups.loc[t, family_names].values

        ts = int(trend_state.loc[t]) if not pd.isna(trend_state.loc[t]) else 0
        vb = int(v_bucket.loc[t]) if not pd.isna(v_bucket.loc[t]) else 0

        # Choose family weights: bucket weights with shrinkage toward regime weights
        r = int(regime_used.loc[t])
        w_r = weights_regimes.get(r, np.ones(len(family_names)) / len(family_names))

        if use_bucket_family_weights and (weights_bucket is not None) and ((ts, vb) in weights_bucket):
            w_b = weights_bucket[(ts, vb)]
            n = bucket_counts.get((ts, vb), 0)
            gamma = 200.0 / (n + 200.0)  # more shrinkage for smaller buckets
            w_fam = (1.0 - gamma) * w_b + gamma * w_r
            w_fam = w_fam / (w_fam.sum() + EPS)
        else:
            w_fam = w_r

        weights_used_daily.loc[t, family_names] = w_fam
        e_raw = float(np.dot(expo_t, w_fam))

        # Extra cutoffs / stress filters (unchanged)
        if (ts == 0) and (vb == 2) and (e_raw > 0):
            e_raw *= 0.40
        if df_main["vxn"].loc[t] > 50:
            e_raw *= 0.20
        if df_main["vix_ts_spread"].loc[t] < -2:
            e_raw *= 0.50

        base_expo_raw.loc[t] = scale_map.get((ts, vb), 1.0) * float(gate_series.loc[t]) * e_raw

    if not quiet:
        print("\n[GATE] gate_series summary:\n", gate_series.describe())
        print("[GATE] value counts:\n", gate_series.round(2).value_counts().head(10))

    # --- Hedge overlay
    hedge_candidates = ["X_BearTrendFollow", "V_BearShort", "VT_Contango_Carry", "VT_VolGap_QQQRich"]
    hedge_cols = [c for c in hedge_candidates if c in df_signals_full.columns]

    if len(hedge_cols) == 0:
        hedge_expo = pd.Series(0.0, index=df_main.index)
        mask_hedge_zone = pd.Series(False, index=df_main.index)
        sign_dir = 0.0
        r_hedge = pd.Series(0.0, index=df_main.index)
    else:
        df_hedge = df_signals_full[hedge_cols].copy()
        df_hedge_norm = normalize_signals_by_pnl_vol(df_hedge, returns, mask_fit)
        beta_map = {"X_BearTrendFollow": 0.7, "VT_Contango_Carry": 0.4, "VT_VolGap_QQQRich": 0.6}
        hedge_expo, mask_hedge_zone, sign_dir = build_hedge_overlay(df_main, df_hedge_norm, mask_fit, returns, beta_map=beta_map)
        r_hedge = hedge_expo.shift(1).fillna(0.0) * returns

    # Baseline (lambda=0)
    base_r_port, _ = backtest_portfolio_with_vol_target(
        base_expo_raw, returns, target_vol_annual=target_vol_annual, vol_window=vol_window, cap_series=cap_series
    )
    ref_valid = compute_perf_stats(base_r_port, mask_valid)["sharpe"]

    best_lambda = 0.0
    best_score = -1e9

    if not quiet and mask_hedge_zone.any():
        corr_zone = pd.concat([base_r_port, r_hedge], axis=1, keys=["port", "hedge"]).loc[mask_train & mask_hedge_zone].corr().iloc[0, 1]
        print(f"\nHedge corr with portfolio (TRAIN & hedge zone): {corr_zone:.3f}")
        print(f"[HEDGE] hedge_expo abs mean: {hedge_expo.abs().mean():.4f}")
        print(f"[HEDGE] hedge_expo nonzero freq: {(hedge_expo.abs() > 1e-12).mean():.4f}")
        print(f"[HEDGE] hedge_zone freq: {mask_hedge_zone.mean():.4f}")
        print(f"[M2] Hedge direction sign_dir = {sign_dir}")

    # Grid search lambda_hedge (constraint: do not degrade VALID too much)
    for lam in lambda_grid:
        expo_raw = base_expo_raw + lam * hedge_expo
        r_port_lam, _ = backtest_portfolio_with_vol_target(
            expo_raw, returns, target_vol_annual=target_vol_annual, vol_window=vol_window, cap_series=cap_series
        )

        st_tr_zone = compute_perf_stats(r_port_lam, mask_train & mask_hedge_zone)
        st_va_zone = compute_perf_stats(r_port_lam, mask_valid & mask_hedge_zone)
        st_valid = compute_perf_stats(r_port_lam, mask_valid)

        score = 0.5 * st_tr_zone["sharpe"] + 0.5 * st_va_zone["sharpe"]

        if not quiet:
            st_train = compute_perf_stats(r_port_lam, mask_train)
            print(
                f"  lambda={lam: .2f} | "
                f"Sharpe(TR_zone)={st_tr_zone['sharpe']:.3f} | Sharpe(VA_zone)={st_va_zone['sharpe']:.3f} | "
                f"Sharpe(TR)={st_train['sharpe']:.3f} | Sharpe(VA)={st_valid['sharpe']:.3f}"
            )

        if st_valid["sharpe"] >= ref_valid - 0.02 and score > best_score:
            best_score = score
            best_lambda = lam

    # Final backtest
    expo_raw_final = base_expo_raw + best_lambda * hedge_expo
    r_port, e_final = backtest_portfolio_with_vol_target(
        expo_raw_final, returns, target_vol_annual=target_vol_annual, vol_window=vol_window, cap_series=cap_series
    )

    r_bh = returns.copy()

    stats_port = {
        "train": compute_perf_stats(r_port, mask_train),
        "valid": compute_perf_stats(r_port, mask_valid),
        "holdout": compute_perf_stats(r_port, mask_holdout),
    }
    stats_bh = {
        "train": compute_perf_stats(r_bh, mask_train),
        "valid": compute_perf_stats(r_bh, mask_valid),
        "holdout": compute_perf_stats(r_bh, mask_holdout),
    }

    if not quiet:
        print("\n=== PERFORMANCE PORTFOLIO M2 (M1 + hedge overlay) ===")
        print(f"lambda_hedge = {best_lambda:.2f}")
        for period in ["train", "valid", "holdout"]:
            st = stats_port[period]
            print(
                f"{period.upper():8s} | Sharpe: {st['sharpe']:.2f} | "
                f"Ret: {100*st['ann_ret']:.1f}% | Vol: {100*st['ann_vol']:.1f}% | "
                f"MaxDD: {100*st['max_dd']:.1f}% | N={st['n_days']}"
            )

        print("\n=== BUY & HOLD QQQ (REFERENCE) ===")
        for period in ["train", "valid", "holdout"]:
            st = stats_bh[period]
            print(
                f"{period.upper():8s} | Sharpe: {st['sharpe']:.2f} | "
                f"Ret: {100*st['ann_ret']:.1f}% | Vol: {100*st['ann_vol']:.1f}% | "
                f"MaxDD: {100*st['max_dd']:.1f}% | N={st['n_days']}"
            )

        print_subperiod_stats(r_port)
        print_rolling_sharpe_summary(r_port, r_bh, window=756)
        print_family_perf_stats(df_groups, returns, fit_on=fit_on)
        analyze_portfolio_weak_zones(df_main, r_port, fit_on=fit_on, model_name="M2", align_with_signal_shift=True)

    return {
        "stats_portfolio": stats_port,
        "stats_buyhold": stats_bh,
        "lambda_hedge": best_lambda,
        "family_weights_bucket": (
            {k: dict(zip(family_names, v)) for k, v in weights_bucket.items()}
            if (use_bucket_family_weights and weights_bucket is not None)
            else None
        ),
        "family_weights_regimes": {r: dict(zip(family_names, weights_regimes[r])) for r in [0, 1]},
        "regime_series": regime,
        "groups_exposures": df_groups,
        "exposure_base_raw": base_expo_raw,
        "hedge_exposure": hedge_expo,
        "final_exposure": e_final,
        "portfolio_returns": r_port,
        "weights_used_daily": weights_used_daily,
    }


def run_M2_with_groups_override(
    df_main: pd.DataFrame,
    df_signals_full: pd.DataFrame,
    groups_override: Optional[dict] = None,
    **kwargs,
):
    global GROUPS
    if groups_override is None:
        return run_M2_two_regimes_with_hedge(df_main, df_signals_full, **kwargs)

    groups_saved = copy.deepcopy(GROUPS)
    try:
        GROUPS = groups_override
        res = run_M2_two_regimes_with_hedge(df_main, df_signals_full, **kwargs)
    finally:
        GROUPS = groups_saved
    return res


# ==============================================================================
# REPORTING (PLOTS + LaTeX tables) — NO saturation, NO bucket heatmap
# ==============================================================================


@dataclass(frozen=True)
class ReportConfig:
    out_dir: str = "./report_artifacts"
    rolling_sharpe_window: int = 756  # ~3y trading days
    dpi: int = 160


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _mask_dict(df_main: pd.DataFrame, fit_on: str = "train") -> dict:
    mask_train, mask_valid, mask_holdout, _ = get_fit_mask(df_main.index, fit_on=fit_on)
    return {
        "TRAIN": mask_train,
        "VALID": mask_valid,
        "HOLDOUT": mask_holdout,
        "ALL": (mask_train | mask_valid | mask_holdout),
    }


def df_to_latex_table_env(
    df: pd.DataFrame,
    caption: str,
    label: str,
    index: bool = False,
    float_fmt: str = "%.2f",
) -> str:
    latex_inner = df.to_latex(
        index=index,
        escape=True,
        float_format=(lambda x: float_fmt % x) if float_fmt else None,
    )
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{latex_inner}\n"
        "\\end{table}\n"
    )


def build_performance_summary_table(stats_port: dict, stats_bh: dict) -> pd.DataFrame:
    rows = []
    for period in ["train", "valid", "holdout"]:
        sp = stats_port[period]
        sb = stats_bh[period]
        rows.append(
            {
                "Period": period.upper(),
                "Port Sharpe": sp["sharpe"],
                "Port AnnRet %": sp["ann_ret"] * 100,
                "Port AnnVol %": sp["ann_vol"] * 100,
                "Port MaxDD %": sp["max_dd"] * 100,
                "BH Sharpe": sb["sharpe"],
                "BH AnnRet %": sb["ann_ret"] * 100,
                "BH AnnVol %": sb["ann_vol"] * 100,
                "BH MaxDD %": sb["max_dd"] * 100,
                "N_days": sp["n_days"],
            }
        )
    return pd.DataFrame(rows)


def _bucket_label(ts: int, vb: int) -> str:
    ts_name = "Bear/Side" if int(ts) == 0 else "Bull"
    vb_name = {0: "LowV", 1: "MedV", 2: "HighV"}.get(int(vb), f"V{vb}")
    return f"{ts_name} x {vb_name}"


def build_bucket_diagnostics_table(
    df_main: pd.DataFrame,
    r_port: pd.Series,
    fit_on: str = "train",
    align_with_signal_shift: bool = True,
    min_days: int = 50,
) -> pd.DataFrame:
    idx = df_main.index
    mask_train, mask_valid, mask_holdout, mask_fit = get_fit_mask(idx, fit_on)

    masks = {
        "TRAIN": mask_train,
        "VALID": mask_valid,
        "HOLDOUT": mask_holdout,
        "ALL": (mask_train | mask_valid | mask_holdout),
    }

    buckets = build_bull_vol_buckets(df_main, mask_fit)
    trend_state = buckets["trend_state"]
    v_bucket = buckets["v_bucket"]

    if align_with_signal_shift:
        ts_ref = trend_state.shift(1)
        vb_ref = v_bucket.shift(1)
    else:
        ts_ref = trend_state
        vb_ref = v_bucket

    rows = []
    for period_name, m_period in masks.items():
        n_total = int(m_period.sum())
        if n_total == 0:
            continue

        for ts, vb in BUCKET_KEYS:
            m_bucket = m_period & (ts_ref == ts) & (vb_ref == vb)
            n_bucket = int(m_bucket.sum())
            if n_bucket < min_days:
                continue

            st = compute_perf_stats(r_port, m_bucket)
            w_days = n_bucket / n_total
            ann_contrib = st["ann_ret"] * w_days

            rows.append(
                {
                    "Period": period_name,
                    "Bucket": _bucket_label(ts, vb),
                    "N_days": n_bucket,
                    "W_days": w_days,
                    "Sharpe": st["sharpe"],
                    "AnnRet %": st["ann_ret"] * 100,
                    "AnnVol %": st["ann_vol"] * 100,
                    "MaxDD %": st["max_dd"] * 100,
                    "AnnContr %": ann_contrib * 100,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    bucket_order = [_bucket_label(ts, vb) for (ts, vb) in BUCKET_KEYS]
    df["Bucket"] = pd.Categorical(df["Bucket"], categories=bucket_order, ordered=True)
    df = df.sort_values(["Period", "Bucket"]).reset_index(drop=True)
    return df


def build_family_weights_bucket_table(res: dict) -> pd.DataFrame:
    fw = res.get("family_weights_bucket", None)
    if not fw:
        return pd.DataFrame()
    rows = []
    for (ts, vb), wmap in fw.items():
        row = {"ts": ts, "vb": vb, "Bucket": _bucket_label(ts, vb)}
        row.update(wmap)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["ts", "vb"]).reset_index(drop=True)


def build_family_weights_regime_table(res: dict) -> pd.DataFrame:
    fw = res.get("family_weights_regimes", None)
    if not fw:
        return pd.DataFrame()
    rows = []
    for regime_id, wmap in fw.items():
        row = {"Regime": int(regime_id)}
        row.update(wmap)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Regime").reset_index(drop=True)


def _equity_curve(r: pd.Series) -> pd.Series:
    r = r.fillna(0.0).clip(lower=-0.99)
    return (1.0 + r).cumprod()


def _underwater_curve(r: pd.Series) -> pd.Series:
    eq = _equity_curve(r)
    peak = eq.cummax()
    return eq / peak - 1.0


def plot_equity_curve_with_splits(
    df_main: pd.DataFrame,
    r_port: pd.Series,
    r_bh: pd.Series,
    masks: dict,
    title: str = "Equity curve: Portfolio vs Buy&Hold",
):
    r_port = r_port.reindex(df_main.index).fillna(0.0)
    r_bh = r_bh.reindex(df_main.index).fillna(0.0)

    eq_port = _equity_curve(r_port)
    eq_bh = _equity_curve(r_bh)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(eq_port.index, eq_port.values, label="Portfolio")
    ax.plot(eq_bh.index, eq_bh.values, label="Buy&Hold QQQ", alpha=0.8)

    def _shade(mask, label, alpha=0.06):
        idx = df_main.index[mask]
        if len(idx) == 0:
            return
        ax.axvspan(idx.min(), idx.max(), alpha=alpha, label=label)

    _shade(masks["TRAIN"], "TRAIN", alpha=0.06)
    _shade(masks["VALID"], "VALID", alpha=0.08)
    _shade(masks["HOLDOUT"], "HOLDOUT", alpha=0.10)

    ax.set_title(title)
    ax.set_ylabel("Equity (normalized)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_rolling_sharpe_comparison(
    df_main: pd.DataFrame,
    r_port: pd.Series,
    r_bh: pd.Series,
    window: int = 756,
    title: str = "Rolling Sharpe (3Y): Portfolio vs Buy&Hold",
):
    r_port = r_port.reindex(df_main.index).fillna(0.0)
    r_bh = r_bh.reindex(df_main.index).fillna(0.0)

    rs_port = compute_rolling_sharpe(r_port, window=window)
    rs_bh = compute_rolling_sharpe(r_bh, window=window)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(rs_port.index, rs_port.values, label="Portfolio")
    ax.plot(rs_bh.index, rs_bh.values, label="Buy&Hold", alpha=0.8)
    ax.axhline(0.0, linewidth=1.0, alpha=0.4)
    ax.set_title(title)
    ax.set_ylabel("Rolling Sharpe")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_underwater_comparison(
    df_main: pd.DataFrame,
    r_port: pd.Series,
    r_bh: pd.Series,
    title: str = "Underwater (Drawdown): Portfolio vs Buy&Hold",
):
    r_port = r_port.reindex(df_main.index).fillna(0.0)
    r_bh = r_bh.reindex(df_main.index).fillna(0.0)

    uw_p = _underwater_curve(r_port)
    uw_b = _underwater_curve(r_bh)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(uw_p.index, uw_p.values, label="Portfolio")
    ax.plot(uw_b.index, uw_b.values, label="Buy&Hold", alpha=0.8)
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def generate_report_artifacts(
    df_main: pd.DataFrame,
    res: dict,
    fit_on: str = "train",
    cfg: ReportConfig = ReportConfig(),
) -> dict:
    """
    Generate the 3 report plots + LaTeX tables for the submission.

    Plots:
      1) Equity curve with TRAIN/VALID/HOLDOUT shading
      2) Rolling 3Y Sharpe (portfolio vs buy&hold)
      3) Underwater / drawdown (portfolio vs buy&hold)

    Tables (LaTeX):
      - Performance summary (TRAIN/VALID/HOLDOUT)
      - Bucket diagnostics (Trend x VXN tertiles)
      - Family weights by bucket (if available)
      - Family weights by regime
    """
    out_dir = _ensure_dir(cfg.out_dir)
    masks = _mask_dict(df_main, fit_on=fit_on)

    r_port = res["portfolio_returns"].reindex(df_main.index).fillna(0.0)
    r_bh = get_returns(df_main).reindex(df_main.index).fillna(0.0)

    # ---- PLOTS ----
    plot_paths: dict[str, str] = {}

    fig1 = plot_equity_curve_with_splits(df_main, r_port, r_bh, masks)
    p1 = out_dir / "plot_equity_curve.png"
    fig1.savefig(p1, dpi=cfg.dpi)
    plt.close(fig1)
    plot_paths["equity_curve"] = str(p1)

    fig2 = plot_rolling_sharpe_comparison(df_main, r_port, r_bh, window=cfg.rolling_sharpe_window)
    p2 = out_dir / "plot_rolling_sharpe.png"
    fig2.savefig(p2, dpi=cfg.dpi)
    plt.close(fig2)
    plot_paths["rolling_sharpe"] = str(p2)

    fig3 = plot_underwater_comparison(df_main, r_port, r_bh)
    p3 = out_dir / "plot_underwater.png"
    fig3.savefig(p3, dpi=cfg.dpi)
    plt.close(fig3)
    plot_paths["underwater"] = str(p3)

    # ---- TABLES (DataFrames) ----
    perf_df = build_performance_summary_table(res["stats_portfolio"], res["stats_buyhold"])
    bucket_df = build_bucket_diagnostics_table(df_main, r_port, fit_on=fit_on, align_with_signal_shift=True, min_days=50)
    w_bucket_df = build_family_weights_bucket_table(res)
    w_regime_df = build_family_weights_regime_table(res)

    # ---- EXPORT TABLES TO .tex ----
    table_paths: dict[str, str] = {}
    latex_strings: dict[str, str] = {}

    def _write_tex(name: str, latex: str):
        path = out_dir / f"table_{name}.tex"
        path.write_text(latex, encoding="utf-8")
        table_paths[name] = str(path)
        latex_strings[name] = latex

    _write_tex(
        "perf_summary",
        df_to_latex_table_env(
            perf_df,
            caption="Performance summary (Portfolio vs Buy\\&Hold) on TRAIN / VALID / HOLDOUT.",
            label="tab:perf_summary",
            index=False,
            float_fmt="%.2f",
        ),
    )

    if bucket_df is not None and not bucket_df.empty:
        _write_tex(
            "bucket_diagnostics",
            df_to_latex_table_env(
                bucket_df,
                caption="Conditional performance by bucket (Trend state \\(\\times\\) VXN tertiles).",
                label="tab:bucket_diag",
                index=False,
                float_fmt="%.2f",
            ),
        )

    if w_bucket_df is not None and not w_bucket_df.empty:
        _write_tex(
            "family_weights_bucket",
            df_to_latex_table_env(
                w_bucket_df,
                caption="Learned family weights by bucket (with shrinkage).",
                label="tab:family_weights_bucket",
                index=False,
                float_fmt="%.3f",
            ),
        )

    if w_regime_df is not None and not w_regime_df.empty:
        _write_tex(
            "family_weights_regime",
            df_to_latex_table_env(
                w_regime_df,
                caption="Family weights by regime (fallback weights).",
                label="tab:family_weights_regime",
                index=False,
                float_fmt="%.3f",
            ),
        )

    return {
        "plots": plot_paths,
        "tables": table_paths,
        "latex": latex_strings,
    }

# ==============================================================================
# SENSITIVITY / ROBUSTNESS TESTS (±10%) — TRAIN/VALID only for decisions
# ==============================================================================
from dataclasses import dataclass

@dataclass(frozen=True)
class SensitivityCase:
    """A single sensitivity perturbation (portfolio params and/or signal overrides)."""
    name: str
    # Portfolio/hyperparams (passed to run_M2_two_regimes_with_hedge)
    vol_window: Optional[int] = None
    target_vol_annual: Optional[float] = None
    bucket_weight_alpha: Optional[float] = None
    bucket_score_floor: Optional[float] = None
    scale_bull_medv: Optional[float] = None
    # Signal overrides (recompute only selected columns; no new signal invention)
    sig_overrides: Optional[dict] = None


def _round_int_perturb(x: int, pct: float) -> int:
    """±pct perturbation but keep a meaningful integer move."""
    if x <= 0:
        return x
    y = int(round(x * (1.0 + pct)))
    if y == x:
        y = x + (1 if pct > 0 else -1)
    return max(2, y)  # avoid tiny windows


def apply_signal_overrides(
    df_main: pd.DataFrame,
    df_signals_all: pd.DataFrame,
    overrides: Optional[dict],
) -> pd.DataFrame:
    """
    Recompute a small subset of signals under perturbed parameters.
    IMPORTANT: This does NOT create new signals; it only redefines existing ones
    with slightly changed parameters (robustness check).
    """
    if not overrides:
        return df_signals_all

    df = df_signals_all.copy()
    c = df_main["close"]
    h = df_main["high"]
    l = df_main["low"]
    o = df_main["open"]
    vxn = df_main["vxn"]

    # ---- Trend: EMA crossover (T_EMA) spans ±10%
    if ("ema_fast" in overrides) or ("ema_slow" in overrides):
        fast = int(overrides.get("ema_fast", 50))
        slow = int(overrides.get("ema_slow", 200))
        df["T_EMA"] = np.sign(c.ewm(span=fast).mean() - c.ewm(span=slow).mean())

    # ---- Trend: Price vs MA (T_PriceVsMA) length ±10%
    if "ma_len" in overrides:
        ma_len = int(overrides["ma_len"])
        df["T_PriceVsMA"] = np.sign(c - c.rolling(ma_len).mean())

    # ---- MeanReversion: RSI2 window (M_RSI2) ±10% (rounded)
    if "rsi2_n" in overrides:
        n = int(overrides["rsi2_n"])
        rsi2 = calc_rsi(c, n)
        df["M_RSI2"] = np.where(rsi2 < 10.0, 1.0, np.where(rsi2 > 90.0, -1.0, 0.0))

    # ---- VolRisk: IV/RV (V_IVRV) realized vol window ±10%
    if "rv10_n" in overrides:
        n = int(overrides["rv10_n"])
        rvN = calc_rv(c, n) * 100.0
        ratio_ivrv = vxn / (rvN + EPS)
        df["V_IVRV"] = np.where(ratio_ivrv > 1.5, 1.0, np.where(ratio_ivrv < 0.8, -0.5, 0.0))

    # ---- MeanReversion: IBS threshold perturb (M_IBS) (optional)
    if ("ibs_lo" in overrides) or ("ibs_hi" in overrides):
        lo = float(overrides.get("ibs_lo", 0.2))
        hi = float(overrides.get("ibs_hi", 0.8))
        ibs = (c - l) / (h - l + EPS)
        df["M_IBS"] = np.where(ibs < lo, 1.0, np.where(ibs > hi, -1.0, 0.0))

    return df.fillna(0.0)


def run_sensitivity_suite(
    df_main: pd.DataFrame,
    df_signals_all: pd.DataFrame,
    base_kwargs: dict,
    fit_on: str = "train",
    quiet: bool = True,
) -> pd.DataFrame:
    """
    Run a set of ±10% perturbations and report performance stability.

    Rules:
      - We evaluate robustness primarily on TRAIN and VALID.
      - HOLDOUT is reported but never used to select variants.
      - No new signal is invented; only small parameter perturbations are applied.
    """
    # Baseline run
    base_res = run_M2_two_regimes_with_hedge(
        df_main=df_main,
        df_signals_full=df_signals_all,
        fit_on=fit_on,
        quiet=quiet,
        **base_kwargs,
    )

    def row_from_res(tag: str, res: dict) -> dict:
        sp = res["stats_portfolio"]
        return {
            "Case": tag,
            "TR_Sharpe": sp["train"]["sharpe"],
            "TR_MaxDD_%": 100 * sp["train"]["max_dd"],
            "VA_Sharpe": sp["valid"]["sharpe"],
            "VA_MaxDD_%": 100 * sp["valid"]["max_dd"],
            "HO_Sharpe": sp["holdout"]["sharpe"],
            "HO_MaxDD_%": 100 * sp["holdout"]["max_dd"],
        }

    rows = [row_from_res("BASE", base_res)]

    # Build sensitivity cases around current baseline params (±10%)
    vol_window0 = int(base_kwargs.get("vol_window", 60))
    target_vol0 = float(base_kwargs.get("target_vol_annual", 0.15))
    bwa0 = float(base_kwargs.get("bucket_weight_alpha", 0.75))
    bsf0 = float(base_kwargs.get("bucket_score_floor", 0.15))
    sbm0 = float(base_kwargs.get("scale_bull_medv", 0.30))

    cases = [
        # ---- Portfolio-level params
        SensitivityCase("VOL_WINDOW_-10%", vol_window=_round_int_perturb(vol_window0, -0.10)),
        SensitivityCase("VOL_WINDOW_+10%", vol_window=_round_int_perturb(vol_window0, +0.10)),
        SensitivityCase("TARGET_VOL_-10%", target_vol_annual=target_vol0 * 0.90),
        SensitivityCase("TARGET_VOL_+10%", target_vol_annual=target_vol0 * 1.10),
        SensitivityCase("BUCKET_ALPHA_-10%", bucket_weight_alpha=bwa0 * 0.90),
        SensitivityCase("BUCKET_ALPHA_+10%", bucket_weight_alpha=bwa0 * 1.10),
        SensitivityCase("SCORE_FLOOR_-10%", bucket_score_floor=bsf0 * 0.90),
        SensitivityCase("SCORE_FLOOR_+10%", bucket_score_floor=bsf0 * 1.10),
        SensitivityCase("SCALE_BULL_MEDV_-10%", scale_bull_medv=sbm0 * 0.90),
        SensitivityCase("SCALE_BULL_MEDV_+10%", scale_bull_medv=sbm0 * 1.10),

        # ---- Signal-level params (key indicators) ±10%
        SensitivityCase("SIG_T_EMA_SPANS_-10%", sig_overrides={"ema_fast": _round_int_perturb(50, -0.10),
                                                              "ema_slow": _round_int_perturb(200, -0.10)}),
        SensitivityCase("SIG_T_EMA_SPANS_+10%", sig_overrides={"ema_fast": _round_int_perturb(50, +0.10),
                                                              "ema_slow": _round_int_perturb(200, +0.10)}),
        SensitivityCase("SIG_T_MA_LEN_-10%", sig_overrides={"ma_len": _round_int_perturb(200, -0.10)}),
        SensitivityCase("SIG_T_MA_LEN_+10%", sig_overrides={"ma_len": _round_int_perturb(200, +0.10)}),
        SensitivityCase("SIG_RSI2_N_-10%", sig_overrides={"rsi2_n": _round_int_perturb(2, -0.10)}),
        SensitivityCase("SIG_RSI2_N_+10%", sig_overrides={"rsi2_n": _round_int_perturb(2, +0.10)}),
        SensitivityCase("SIG_IVRV_RVWIN_-10%", sig_overrides={"rv10_n": _round_int_perturb(10, -0.10)}),
        SensitivityCase("SIG_IVRV_RVWIN_+10%", sig_overrides={"rv10_n": _round_int_perturb(10, +0.10)}),
        SensitivityCase("SIG_IBS_THRESH_-10%", sig_overrides={"ibs_lo": 0.20 * 0.90, "ibs_hi": 0.80 * 0.90}),
        SensitivityCase("SIG_IBS_THRESH_+10%", sig_overrides={"ibs_lo": 0.20 * 1.10, "ibs_hi": 0.80 * 1.10}),
    ]

    for cs in cases:
        # apply signal overrides (if any)
        df_sig = apply_signal_overrides(df_main, df_signals_all, cs.sig_overrides)

        kwargs = dict(base_kwargs)
        if cs.vol_window is not None:
            kwargs["vol_window"] = cs.vol_window
        if cs.target_vol_annual is not None:
            kwargs["target_vol_annual"] = cs.target_vol_annual
        if cs.bucket_weight_alpha is not None:
            kwargs["bucket_weight_alpha"] = cs.bucket_weight_alpha
        if cs.bucket_score_floor is not None:
            kwargs["bucket_score_floor"] = cs.bucket_score_floor
        if cs.scale_bull_medv is not None:
            kwargs["scale_bull_medv"] = cs.scale_bull_medv

        res = run_M2_two_regimes_with_hedge(
            df_main=df_main,
            df_signals_full=df_sig,
            fit_on=fit_on,
            quiet=True,  # keep suite clean
            **kwargs,
        )
        rows.append(row_from_res(cs.name, res))

    df_out = pd.DataFrame(rows)

    # Add deltas vs baseline (robustness view)
    base = df_out.iloc[0]
    for col in ["TR_Sharpe", "VA_Sharpe", "HO_Sharpe"]:
        df_out[f"Δ{col}"] = df_out[col] - float(base[col])

    return df_out


def save_sensitivity_table_tex(df_sens: pd.DataFrame, out_path: str):
    """
    Save a compact LaTeX table for the report.
    Keep it focused on TRAIN/VALID (HOLDOUT reported, but not emphasized).
    """
    show_cols = [
        "Case",
        "TR_Sharpe", "ΔTR_Sharpe", "TR_MaxDD_%",
        "VA_Sharpe", "ΔVA_Sharpe", "VA_MaxDD_%",
        "HO_Sharpe", "HO_MaxDD_%",
    ]
    df = df_sens[show_cols].copy()

    # format
    for c in df.columns:
        if c == "Case":
            continue
        df[c] = df[c].astype(float)

    tex = df.to_latex(index=False, float_format="%.3f", escape=True)
    Path(out_path).write_text(tex, encoding="utf-8")


# ==============================================================================
# Main
# ==============================================================================


def main():
    df_main = load_main_dataframe()
    if df_main is None:
        return

    # Get fit mask for signal generation (use train only to avoid lookahead)
    mask_train, _, _, _ = get_fit_mask(df_main.index, fit_on="train")

    df_core = generate_full_signal_library(df_main)
    df_volts = generate_vol_term_structure_signals(df_main, mask_fit=mask_train)
    df_candidates = generate_candidate_signals(df_main)

    df_signals_base = df_core.join(df_volts, how="left").reindex(df_main.index).fillna(0.0)
    df_signals_all = df_signals_base.join(df_candidates, how="left").reindex(df_main.index).fillna(0.0)

    print("\n##########################")
    print("#   RUNNING M2 (all signals incl. candidates)   #")
    print("##########################")

    res = run_M2_two_regimes_with_hedge(
        df_main=df_main,
        df_signals_full=df_signals_all,
        fit_on="train",
        family_keep=["Trend", "MeanRev", "VolRisk"],
        scale_bull_medv=0.3,
        lambda_grid=[-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
        bucket_weight_alpha=0.75,
        bucket_min_w=0.10,
        bucket_max_w=0.60,
        bucket_score_floor=0.15,
        quiet=False,
    )

    # ------------------------------
    # Sensitivity / Robustness suite (±10%)
    # ------------------------------
    base_kwargs = dict(
        target_vol_annual=0.15,
        vol_window=60,
        family_keep=["Trend", "MeanRev", "VolRisk"],
        scale_bull_medv=0.3,
        lambda_grid=[-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
        bucket_weight_alpha=0.75,
        bucket_min_w=0.10,
        bucket_max_w=0.60,
        bucket_score_floor=0.15,
    )

    df_sens = run_sensitivity_suite(
        df_main=df_main,
        df_signals_all=df_signals_all,
        base_kwargs=base_kwargs,
        fit_on="train",
        quiet=True,
    )
    print("\n=== SENSITIVITY (±10%) SUMMARY ===")
    with pd.option_context("display.width", 180):
        print(df_sens.to_string(index=False))

    out_dir = Path("./report_artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_sensitivity_table_tex(df_sens, str(out_dir / "table_sensitivity.tex"))
    print(f"\n[REPORT] Sensitivity table saved: {out_dir / 'table_sensitivity.tex'}")

    artifacts = generate_report_artifacts(df_main, res, fit_on="train", cfg=ReportConfig(out_dir="./report_artifacts"))
    print("\n[REPORT] Plots saved:")
    for k, v in artifacts["plots"].items():
        print(f"  - {k}: {v}")

    print("\n[REPORT] LaTeX tables saved:")
    for k, v in artifacts["tables"].items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()