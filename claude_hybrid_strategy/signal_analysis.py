"""
Comprehensive Signal Component Analysis
========================================

Backtests each individual signal component from all submissions
to identify the best and worst performers.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DATA LOADING
# =============================================================================

def load_qqq_data():
    """Load QQQ data from available files."""
    paths = [
        'Quanta Fellowship Train & Validate.csv',
        '../minh_nguyen/Quanta Fellowship Train & Validate.csv',
    ]

    blind_paths = [
        'QQQ Fellowship Blind Out of Sample.csv',
        '../minh_nguyen/QQQ Fellowship Blind Out of Sample.csv',
    ]

    df_train = None
    for path in paths:
        try:
            df_train = pd.read_csv(path)
            break
        except:
            continue

    df_blind = None
    for path in blind_paths:
        try:
            df_blind = pd.read_csv(path)
            break
        except:
            continue

    if df_train is None or df_blind is None:
        raise FileNotFoundError("Could not find data files")

    # Process training data
    df_train.columns = df_train.columns.str.strip()
    df_train['Date'] = pd.to_datetime(df_train['Time'], errors='coerce')
    df_train = df_train.dropna(subset=['Date']).set_index('Date').sort_index()
    if 'Latest' in df_train.columns:
        df_train = df_train.rename(columns={'Latest': 'Close'})
    df_train = df_train[['Open', 'High', 'Low', 'Close', 'Volume']]

    # Process blind data
    df_blind.columns = df_blind.columns.str.strip()
    df_blind['Date'] = pd.to_datetime(df_blind['Time'], errors='coerce')
    df_blind = df_blind.dropna(subset=['Date']).set_index('Date').sort_index()
    if 'Latest' in df_blind.columns:
        df_blind = df_blind.rename(columns={'Latest': 'Close'})
    df_blind = df_blind[['Open', 'High', 'Low', 'Close', 'Volume']]

    # Combine
    df = pd.concat([df_train, df_blind])
    df = df[~df.index.duplicated(keep='first')].sort_index()

    return df


def calculate_features(df):
    """Calculate all technical features."""
    df = df.copy()

    # Returns
    df['Returns'] = df['Close'].pct_change()

    # Moving averages
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'SMA_{period}'] = df['Close'].rolling(period).mean()

    # Parkinson Volatility
    park_const = 1.0 / (4.0 * np.log(2.0))
    df['Park_Var'] = park_const * (np.log(df['High'] / df['Low']) ** 2)
    df['Park_Vol_21'] = np.sqrt(df['Park_Var'].rolling(21).mean()) * np.sqrt(252)

    # Realized Vol
    df['RVol_20'] = df['Returns'].rolling(20).std() * np.sqrt(252)
    df['RVol_60'] = df['Returns'].rolling(60).std() * np.sqrt(252)

    # Vol of Vol
    df['Vol_of_Vol'] = df['Park_Vol_21'].rolling(60).std()

    # Vol percentile
    df['Vol_Pct'] = df['Park_Vol_21'].expanding(min_periods=252).rank(pct=True) * 100
    df['Vol_Pct'] = df['Vol_Pct'].fillna(50)

    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI_14'] = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))

    # RSI 2 (short-term)
    gain2 = delta.where(delta > 0, 0).rolling(2).mean()
    loss2 = (-delta.where(delta < 0, 0)).rolling(2).mean()
    df['RSI_2'] = 100 - (100 / (1 + gain2 / loss2.replace(0, 1e-9)))

    # Momentum
    for period in [5, 10, 20, 60, 120]:
        df[f'Mom_{period}'] = df['Close'].pct_change(period)

    # Distance from MAs
    df['Dist_MA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
    df['Dist_MA50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
    df['Dist_MA200'] = (df['Close'] - df['SMA_200']) / df['SMA_200']

    # IBS (Intraday Body Strength)
    range_hl = (df['High'] - df['Low']).replace(0, 1e-9)
    df['IBS'] = (df['Close'] - df['Low']) / range_hl

    # ATR
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            (df['High'] - df['Close'].shift(1)).abs(),
            (df['Low'] - df['Close'].shift(1)).abs()
        )
    )
    df['ATR_14'] = df['TR'].rolling(14).mean()

    # MACD
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()

    # Bollinger Bands
    df['BB_Middle'] = df['SMA_20']
    bb_std = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
    df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower']).replace(0, 1e-9)

    # Trend indicators
    df['Above_200'] = (df['Close'] > df['SMA_200']).astype(int)
    df['Above_50'] = (df['Close'] > df['SMA_50']).astype(int)
    df['Above_20'] = (df['Close'] > df['SMA_20']).astype(int)

    # Golden/Death cross
    df['Golden_Cross'] = (df['SMA_50'] > df['SMA_200']).astype(int)

    return df


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ts_rank(x, window):
    """Rolling percentile rank."""
    return x.rolling(window, min_periods=1).apply(
        lambda a: (a.argsort().argsort()[-1] + 1) / len(a), raw=True
    )


def ts_mad(x, window):
    """Rolling Mean Absolute Deviation."""
    mean = x.rolling(window, min_periods=1).mean()
    return (x - mean).abs().rolling(window, min_periods=1).mean()


def zscore(s, w):
    """Rolling z-score."""
    return ((s - s.rolling(w).mean()) / s.rolling(w).std(ddof=1)).fillna(0)


# =============================================================================
# SIGNAL DEFINITIONS
# =============================================================================

# -------------------- ZHENYU XI SIGNALS --------------------

def signal_hv_momvol_combo(df):
    """HV_MomVol_Combo from Zhenyu Xi."""
    vol_pct = df['Vol_Pct'] / 100
    vol_change_5 = df['Park_Vol_21'].pct_change(5)
    vol_falling = vol_change_5 < 0
    mom_20 = df['Mom_20']
    mom_positive = mom_20 > 0

    high_vol = (vol_pct >= 0.60) & (vol_pct < 0.85)

    signal = pd.Series(0.0, index=df.index)
    signal[high_vol & vol_falling & mom_positive] = 1.0
    signal[high_vol & vol_falling & ~mom_positive] = 0.5
    signal[high_vol & ~vol_falling & mom_positive] = 0.6
    signal[high_vol & ~vol_falling & ~mom_positive] = 0.1

    return signal.fillna(0)


def signal_gamma_momentum_curvature(df):
    """Gamma Momentum Curvature from Zhenyu Xi."""
    vol_pct = df['Vol_Pct'] / 100
    close = df['Close']

    above_200 = df['Above_200'] == 1
    above_50 = df['Above_50'] == 1
    strong_trend = above_200 & above_50

    mom_10 = df['Mom_10']
    mom_20 = df['Mom_20']
    mom_60 = df['Mom_60']

    expected_mom_10 = mom_20 / 2
    mom_curvature = mom_10 - expected_mom_10

    curv_accelerating = mom_curvature > 0.005
    curv_strong_accel = mom_curvature > 0.01

    rsi = df['RSI_14']

    signal = pd.Series(0.0, index=df.index)

    # LOW VOL
    low_vol = vol_pct < 0.30
    signal[low_vol & curv_strong_accel & strong_trend] = 2.0
    signal[low_vol & curv_accelerating & strong_trend] = 1.6
    signal[low_vol & curv_accelerating & above_200] = 1.2
    signal[low_vol & above_200 & (mom_60 > 0)] = 0.8

    # NORMAL VOL
    normal_vol = (vol_pct >= 0.30) & (vol_pct < 0.60)
    dist_20 = df['Dist_MA20']
    dip = dist_20 < -0.02

    signal[normal_vol & above_200 & dip & curv_accelerating & (rsi < 40)] = 1.4
    signal[normal_vol & above_200 & dip & (mom_curvature > 0) & (rsi < 45)] = 1.0
    signal[normal_vol & above_200 & curv_accelerating] = 0.6

    # HIGH/PANIC
    high_vol = vol_pct >= 0.60
    signal[high_vol & above_200 & curv_accelerating & (rsi < 30)] = 0.8
    signal[high_vol & above_200 & (rsi < 25)] = 0.4

    return signal.fillna(0).clip(0, 2.0)


def signal_mean_revert_v2(df):
    """Mean Revert V2 from Zhenyu Xi."""
    vol_pct = df['Vol_Pct'] / 100

    vol_of_vol = df['Park_Vol_21'].rolling(10).std()
    vol_of_vol_pct = vol_of_vol.expanding(min_periods=252).rank(pct=True)
    stable_vol = vol_of_vol_pct < 0.50

    above_200 = df['Above_200'] == 1
    above_50 = df['Above_50'] == 1
    strong_trend = above_200 & above_50

    dist_20 = df['Dist_MA20']
    oversold_mr = dist_20 < -0.025
    deep_oversold = dist_20 < -0.04

    rsi = df['RSI_14']

    signal = pd.Series(0.0, index=df.index)

    # LOW VOL
    low_vol = vol_pct < 0.30
    signal[low_vol & strong_trend & deep_oversold] = 2.2
    signal[low_vol & strong_trend & oversold_mr] = 1.8
    signal[low_vol & strong_trend] = 1.4
    signal[low_vol & above_200 & oversold_mr] = 1.2

    # NORMAL VOL
    normal_vol = (vol_pct >= 0.30) & (vol_pct < 0.60)
    signal[normal_vol & stable_vol & above_200 & deep_oversold & (rsi < 35)] = 1.4
    signal[normal_vol & stable_vol & above_200 & oversold_mr & (rsi < 40)] = 1.1

    # HIGH VOL
    high_vol = (vol_pct >= 0.60) & (vol_pct < 0.85)
    signal[high_vol & above_200 & deep_oversold & (rsi < 30)] = 0.9

    # PANIC
    panic = vol_pct >= 0.85
    signal[panic & (rsi < 20)] = 0.8

    return signal.fillna(0).clip(0, 2.5)


def signal_conservative_mr(df):
    """Conservative MR from Zhenyu Xi."""
    vol_pct = df['Vol_Pct'] / 100

    vol_of_vol = df['Park_Vol_21'].rolling(10).std()
    vol_of_vol_pct = vol_of_vol.expanding(min_periods=252).rank(pct=True)
    very_stable = vol_of_vol_pct < 0.30

    above_200 = df['Above_200'] == 1
    above_50 = df['Above_50'] == 1
    strong_trend = above_200 & above_50

    dist_20 = df['Dist_MA20']
    mr_buy = dist_20 < -0.03
    mr_strong_buy = dist_20 < -0.05

    mom_60 = df['Mom_60']
    mom_confirmed = mom_60 > 0

    rsi = df['RSI_14']

    signal = pd.Series(0.0, index=df.index)

    # LOW VOL
    low_vol = vol_pct < 0.25
    signal[low_vol & very_stable & strong_trend & mom_confirmed] = 1.8
    signal[low_vol & strong_trend & mom_confirmed] = 1.4
    signal[low_vol & strong_trend] = 1.0

    # NORMAL VOL
    normal_vol = (vol_pct >= 0.25) & (vol_pct < 0.55)
    signal[normal_vol & very_stable & above_200 & mr_strong_buy & mom_confirmed & (rsi < 30)] = 1.2
    signal[normal_vol & very_stable & above_200 & mr_buy & mom_confirmed & (rsi < 35)] = 0.9

    # HIGH VOL
    high_vol = (vol_pct >= 0.55) & (vol_pct < 0.85)
    signal[high_vol & very_stable & above_200 & mr_strong_buy & (rsi < 25)] = 0.5

    # PANIC
    panic = vol_pct >= 0.85
    signal[panic & (rsi < 18)] = 0.6

    return signal.fillna(0).clip(0, 2.0)


def signal_panic_fade(df):
    """Panic Fade from Zhenyu Xi."""
    close = df['Close']
    dist_50 = df['Dist_MA50']
    rsi = df['RSI_14']

    mom_5 = df['Mom_5']
    prev_mom_5 = mom_5.shift(3)

    park_vol = df['Park_Vol_21']
    target_vol = 0.20
    vol_scalar = np.clip(target_vol / (park_vol + 0.05), 0.3, 2.5)

    signal = pd.Series(0.0, index=df.index)

    deep_distress = (dist_50 < -0.12) & (rsi < 35)
    signal[deep_distress] = 0.8

    moderate_distress = (dist_50 < -0.08) & (rsi < 40) & ~deep_distress
    signal[moderate_distress] = 0.5

    mom_turning = (mom_5 > 0) & (prev_mom_5 < -0.03)
    recovery = mom_turning & (rsi < 50)
    signal[recovery] = np.maximum(signal[recovery], 0.6)

    signal = signal * vol_scalar

    return signal.fillna(0).clip(0, 1.5)


# -------------------- KIRTIRAJSINH PARMAR SIGNALS --------------------

def signal_trend_state(df):
    """Trend state signal from Kirtirajsinh Parmar."""
    close = df['Close']

    sma200 = df['SMA_200']
    dist = (close - sma200) / sma200
    roc120 = close.pct_change(120)

    W_Z = 504
    trend_state = ((zscore(dist, W_Z) + zscore(roc120, W_Z)) / 2).clip(-3, 3)

    # Convert to signal
    signal = pd.Series(0.0, index=df.index)
    signal[trend_state > 0.5] = 1.0
    signal[trend_state > 1.0] = 1.5
    signal[trend_state < -0.5] = 0.0
    signal[(trend_state >= -0.5) & (trend_state <= 0.5)] = 0.5

    return signal.fillna(0)


def signal_vol_state(df):
    """Volatility state signal from Kirtirajsinh Parmar."""
    ret = df['Returns']
    high = df['High']
    low = df['Low']
    close = df['Close']

    rvol = ret.rolling(20).std(ddof=1) * np.sqrt(252)
    vov = rvol.rolling(60).std(ddof=1)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    shock = tr / tr.rolling(14).mean()

    W_Z = 504
    vol_state = ((zscore(rvol, W_Z) + zscore(vov, W_Z) + zscore(shock, W_Z)) / 3).clip(-3, 3)

    # Low vol = more exposure
    signal = pd.Series(0.5, index=df.index)
    signal[vol_state < -1.0] = 1.5
    signal[vol_state < -0.5] = 1.2
    signal[vol_state > 0.5] = 0.3
    signal[vol_state > 1.0] = 0.1

    return signal.fillna(0.5)


def signal_momentum_state(df):
    """Momentum state signal from Kirtirajsinh Parmar."""
    close = df['Close']

    mom_blend = (close.pct_change(5) + close.pct_change(20) + close.pct_change(60)) / 3
    accel = close.pct_change(20) - close.pct_change(60)

    W_Z = 504
    mom_state = ((zscore(mom_blend, W_Z) + zscore(accel, W_Z)) / 2).clip(-3, 3)

    signal = pd.Series(0.5, index=df.index)
    signal[mom_state > 1.0] = 1.5
    signal[mom_state > 0.5] = 1.2
    signal[mom_state > 0] = 0.8
    signal[mom_state < -0.5] = 0.2
    signal[mom_state < -1.0] = 0.0

    return signal.fillna(0.5)


def signal_ibs_adjustment(df):
    """IBS adjustment signal from Kirtirajsinh Parmar."""
    ibs = df['IBS']
    above_200 = df['Above_200'] == 1

    signal = pd.Series(0.5, index=df.index)

    # Low IBS = closed near low = buy
    signal[ibs < 0.2] = 1.5
    signal[(ibs >= 0.2) & (ibs < 0.3)] = 1.2

    # High IBS = closed near high = reduce
    signal[ibs > 0.8] = 0.3
    signal[(ibs > 0.7) & (ibs <= 0.8)] = 0.4

    # Apply trend filter
    signal[~above_200] *= 0.5

    return signal.fillna(0.5)


def signal_cashfill(df):
    """Cashfill opportunistic signal from Kirtirajsinh Parmar."""
    close = df['Close']
    ret = df['Returns']
    rsi = df['RSI_14']

    # Monday weakness
    mon = (df.index.dayofweek == 0) & (ret.shift(1) < 0)

    # Bollinger band break
    bb = close < df['BB_Lower']

    # Gap down
    gap = df['Open'] < (df['Low'].shift(1) * 0.995)

    signal = pd.Series(0.0, index=df.index)
    signal[mon | (rsi < 30) | bb | gap] = 0.75

    return signal.fillna(0)


# -------------------- HYBRID STRATEGY SIGNALS --------------------

def signal_vol_shock_mr(df):
    """Vol Shock Mean Reversion from Minh Nguyen via hybrid."""
    x = df['Returns'].diff(2)
    x = ts_mad(x, 2)
    x = np.power(1.0 / (x + 1e-9) - 1.0, 0.5)
    x = ts_rank(x, 189) - 0.33

    signal = np.sign(x)
    signal = signal.replace({1.0: 1.5, -1.0: 0.3, 0.0: 0.5})

    # Trend filter
    above_200 = df['Above_200'] == 1
    signal[above_200 & (signal > 0.5)] *= 1.1
    signal[~above_200] *= 0.5

    return signal.fillna(0.5).clip(0, 1.5)


def signal_rsi_mean_reversion(df):
    """Simple RSI mean reversion."""
    rsi = df['RSI_14']
    above_200 = df['Above_200'] == 1

    signal = pd.Series(0.5, index=df.index)

    # Oversold
    signal[(rsi < 30) & above_200] = 1.5
    signal[(rsi < 35) & above_200] = 1.2
    signal[(rsi < 40) & above_200] = 0.9

    # Overbought
    signal[rsi > 70] = 0.3
    signal[rsi > 80] = 0.1

    return signal.fillna(0.5)


def signal_trend_following(df):
    """Simple trend following signal."""
    above_200 = df['Above_200'] == 1
    above_50 = df['Above_50'] == 1
    above_20 = df['Above_20'] == 1

    golden_cross = df['Golden_Cross'] == 1

    mom_20 = df['Mom_20']
    mom_60 = df['Mom_60']

    signal = pd.Series(0.0, index=df.index)

    # Strong uptrend
    signal[above_200 & above_50 & above_20 & golden_cross] = 1.5
    signal[above_200 & above_50 & golden_cross] = 1.2
    signal[above_200 & golden_cross] = 0.9
    signal[above_200] = 0.6

    # Boost with positive momentum
    signal[(mom_20 > 0.02) & (mom_60 > 0.05)] *= 1.2

    return signal.fillna(0).clip(0, 1.5)


def signal_macd_crossover(df):
    """MACD crossover signal."""
    macd = df['MACD']
    macd_signal = df['MACD_Signal']
    above_200 = df['Above_200'] == 1

    signal = pd.Series(0.5, index=df.index)

    # MACD above signal line
    signal[(macd > macd_signal) & above_200] = 1.2
    signal[(macd > macd_signal) & (macd > 0) & above_200] = 1.5

    # MACD below signal
    signal[macd < macd_signal] = 0.3
    signal[(macd < macd_signal) & (macd < 0)] = 0.1

    return signal.fillna(0.5)


def signal_bollinger_mean_reversion(df):
    """Bollinger band mean reversion."""
    bb_pos = df['BB_Position']
    above_200 = df['Above_200'] == 1
    rsi = df['RSI_14']

    signal = pd.Series(0.5, index=df.index)

    # Below lower band = buy
    signal[(bb_pos < 0.1) & above_200 & (rsi < 40)] = 1.5
    signal[(bb_pos < 0.2) & above_200] = 1.2

    # Above upper band = reduce
    signal[bb_pos > 0.9] = 0.2
    signal[bb_pos > 0.8] = 0.3

    return signal.fillna(0.5)


def signal_volume_breakout(df):
    """Volume-based breakout signal."""
    close = df['Close']
    volume = df['Volume']

    vol_avg = volume.rolling(20).mean()
    vol_surge = volume > (vol_avg * 1.5)

    price_breakout = close > close.rolling(20).max().shift(1)

    above_200 = df['Above_200'] == 1

    signal = pd.Series(0.5, index=df.index)
    signal[vol_surge & price_breakout & above_200] = 1.5
    signal[vol_surge & above_200] = 0.9

    return signal.fillna(0.5)


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def backtest_signal(df, signal, period='all'):
    """Backtest a single signal."""

    TRAIN_END = '2015-12-31'
    VAL_START = '2016-01-01'
    VAL_END = '2021-12-31'
    BLIND_START = '2022-01-01'

    if period == 'train':
        mask = df.index <= TRAIN_END
    elif period == 'validation':
        mask = (df.index >= VAL_START) & (df.index <= VAL_END)
    elif period == 'blind':
        mask = df.index >= BLIND_START
    else:
        mask = pd.Series(True, index=df.index)

    returns = df.loc[mask, 'Returns']
    sig = signal.loc[mask]

    # Strategy returns
    strat_ret = (sig.shift(1) * returns).dropna()

    if len(strat_ret) < 10:
        return None

    # Metrics
    total_return = (1 + strat_ret).prod() - 1
    n_years = len(strat_ret) / 252
    ann_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    volatility = strat_ret.std() * np.sqrt(252) if len(strat_ret) > 1 else 0
    sharpe = ann_return / volatility if volatility > 0 else 0

    cum_ret = (1 + strat_ret).cumprod()
    max_dd = (cum_ret / cum_ret.cummax() - 1).min()

    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
    win_rate = (strat_ret > 0).mean()

    # Buy and hold for comparison
    bh_ret = returns.dropna()
    bh_total = (1 + bh_ret).prod() - 1
    bh_ann = (1 + bh_total) ** (1 / n_years) - 1 if n_years > 0 else 0
    bh_vol = bh_ret.std() * np.sqrt(252)
    bh_sharpe = bh_ann / bh_vol if bh_vol > 0 else 0

    return {
        'period': period,
        'sharpe': sharpe,
        'ann_return': ann_return,
        'volatility': volatility,
        'max_drawdown': max_dd,
        'calmar': calmar,
        'win_rate': win_rate,
        'total_return': total_return,
        'n_days': len(strat_ret),
        'bh_sharpe': bh_sharpe,
        'excess_sharpe': sharpe - bh_sharpe,
        'daily_returns': strat_ret,
        'cumulative': cum_ret
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_all_signals():
    """Analyze all signal components."""
    print("=" * 70)
    print("COMPREHENSIVE SIGNAL COMPONENT ANALYSIS")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    df = load_qqq_data()
    df = calculate_features(df)
    print(f"Loaded {len(df)} days of data")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")

    # Define all signals to test
    signals = {
        # Zhenyu Xi signals
        'ZX_HV_MomVol': signal_hv_momvol_combo,
        'ZX_Gamma_MomCurv': signal_gamma_momentum_curvature,
        'ZX_MeanRevert_V2': signal_mean_revert_v2,
        'ZX_Conservative_MR': signal_conservative_mr,
        'ZX_Panic_Fade': signal_panic_fade,

        # Kirtirajsinh Parmar signals
        'KP_Trend_State': signal_trend_state,
        'KP_Vol_State': signal_vol_state,
        'KP_Momentum_State': signal_momentum_state,
        'KP_IBS_Adj': signal_ibs_adjustment,
        'KP_Cashfill': signal_cashfill,

        # Hybrid/General signals
        'HY_Vol_Shock_MR': signal_vol_shock_mr,
        'HY_RSI_MR': signal_rsi_mean_reversion,
        'HY_Trend_Follow': signal_trend_following,
        'HY_MACD_Cross': signal_macd_crossover,
        'HY_Bollinger_MR': signal_bollinger_mean_reversion,
        'HY_Volume_Breakout': signal_volume_breakout,
    }

    # Run backtests
    print("\nBacktesting signals...")
    results = {}

    for name, signal_func in signals.items():
        try:
            sig = signal_func(df)

            results[name] = {
                'signal': sig,
                'train': backtest_signal(df, sig, 'train'),
                'validation': backtest_signal(df, sig, 'validation'),
                'blind': backtest_signal(df, sig, 'blind')
            }
            print(f"  {name}: Blind Sharpe = {results[name]['blind']['sharpe']:.2f}")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")

    return df, results


def print_results_table(results):
    """Print formatted results table."""
    print("\n" + "=" * 100)
    print("SIGNAL PERFORMANCE SUMMARY")
    print("=" * 100)

    # Header
    print(f"\n{'Signal':<25} | {'Train SR':>10} | {'Val SR':>10} | {'Blind SR':>10} | {'Blind DD':>10} | {'Blind Cal':>10}")
    print("-" * 100)

    # Sort by blind Sharpe
    sorted_signals = sorted(results.items(),
                           key=lambda x: x[1]['blind']['sharpe'] if x[1]['blind'] else -999,
                           reverse=True)

    for name, res in sorted_signals:
        train = res['train']
        val = res['validation']
        blind = res['blind']

        if train and val and blind:
            print(f"{name:<25} | {train['sharpe']:>10.3f} | {val['sharpe']:>10.3f} | "
                  f"{blind['sharpe']:>10.3f} | {blind['max_drawdown']:>9.1%} | {blind['calmar']:>10.2f}")

    print("-" * 100)

    # Find best performers
    print("\n" + "=" * 70)
    print("TOP 5 SIGNALS BY BLIND SHARPE RATIO")
    print("=" * 70)

    for i, (name, res) in enumerate(sorted_signals[:5]):
        if res['blind']:
            blind = res['blind']
            print(f"\n{i+1}. {name}")
            print(f"   Sharpe: {blind['sharpe']:.3f} | Return: {blind['ann_return']:.1%} | "
                  f"DD: {blind['max_drawdown']:.1%} | Calmar: {blind['calmar']:.2f}")

    # Find worst performers
    print("\n" + "=" * 70)
    print("BOTTOM 3 SIGNALS (Consider Removing)")
    print("=" * 70)

    for i, (name, res) in enumerate(sorted_signals[-3:]):
        if res['blind']:
            blind = res['blind']
            print(f"\n{len(sorted_signals) - 2 + i}. {name}")
            print(f"   Sharpe: {blind['sharpe']:.3f} | Return: {blind['ann_return']:.1%} | "
                  f"DD: {blind['max_drawdown']:.1%}")


def plot_signal_comparison(df, results):
    """Generate comparison plots."""

    # Filter for valid results
    valid_results = {k: v for k, v in results.items() if v['blind'] is not None}

    # Sort by blind Sharpe
    sorted_signals = sorted(valid_results.items(),
                           key=lambda x: x[1]['blind']['sharpe'],
                           reverse=True)

    # Figure 1: Sharpe comparison across periods
    fig1, ax1 = plt.subplots(figsize=(14, 8))

    names = [s[0] for s in sorted_signals]
    train_sharpes = [s[1]['train']['sharpe'] for s in sorted_signals]
    val_sharpes = [s[1]['validation']['sharpe'] for s in sorted_signals]
    blind_sharpes = [s[1]['blind']['sharpe'] for s in sorted_signals]

    x = np.arange(len(names))
    width = 0.25

    ax1.bar(x - width, train_sharpes, width, label='Train', alpha=0.8)
    ax1.bar(x, val_sharpes, width, label='Validation', alpha=0.8)
    ax1.bar(x + width, blind_sharpes, width, label='Blind', alpha=0.8)

    ax1.set_ylabel('Sharpe Ratio')
    ax1.set_title('Signal Component Sharpe Ratios by Period')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.legend()
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.axhline(y=1, color='green', linestyle='--', linewidth=0.5, alpha=0.5)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('signal_sharpe_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: signal_sharpe_comparison.png")

    # Figure 2: Equity curves for top 5 signals (blind period)
    fig2, ax2 = plt.subplots(figsize=(14, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, min(5, len(sorted_signals))))

    for i, (name, res) in enumerate(sorted_signals[:5]):
        if res['blind'] and 'cumulative' in res['blind']:
            cum = res['blind']['cumulative']
            ax2.plot(cum.index, cum.values, label=f"{name} (SR: {res['blind']['sharpe']:.2f})",
                    color=colors[i], linewidth=1.5)

    # Add buy and hold
    blind_mask = df.index >= '2022-01-01'
    bh = (1 + df.loc[blind_mask, 'Returns'].fillna(0)).cumprod()
    ax2.plot(bh.index, bh.values, label='Buy & Hold', color='gray', linestyle='--', alpha=0.7)

    ax2.set_ylabel('Cumulative Return')
    ax2.set_title('Top 5 Signals - Blind Period Equity Curves (2022-2025)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('signal_equity_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: signal_equity_curves.png")

    # Figure 3: Drawdown comparison
    fig3, ax3 = plt.subplots(figsize=(14, 8))

    names = [s[0] for s in sorted_signals]
    blind_dds = [abs(s[1]['blind']['max_drawdown']) * 100 for s in sorted_signals]

    colors = ['green' if dd < 20 else 'orange' if dd < 35 else 'red' for dd in blind_dds]
    bars = ax3.bar(names, blind_dds, color=colors, edgecolor='black')

    ax3.axhline(y=17, color='red', linestyle='--', label='17% DD Limit')
    ax3.set_ylabel('Max Drawdown (%)')
    ax3.set_title('Signal Component Max Drawdowns (Blind Period)')
    ax3.set_xticklabels(names, rotation=45, ha='right')
    ax3.legend()

    for bar, dd in zip(bars, blind_dds):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{dd:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('signal_drawdown_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: signal_drawdown_comparison.png")

    # Figure 4: Calmar ratio comparison
    fig4, ax4 = plt.subplots(figsize=(14, 8))

    blind_calmars = [s[1]['blind']['calmar'] for s in sorted_signals]

    colors = ['green' if c >= 2.0 else 'orange' if c >= 1.0 else 'red' for c in blind_calmars]
    bars = ax4.bar(names, blind_calmars, color=colors, edgecolor='black')

    ax4.axhline(y=2.0, color='green', linestyle='--', label='Calmar 2.0 Target')
    ax4.axhline(y=1.0, color='orange', linestyle='--', label='Calmar 1.0')
    ax4.set_ylabel('Calmar Ratio')
    ax4.set_title('Signal Component Calmar Ratios (Blind Period)')
    ax4.set_xticklabels(names, rotation=45, ha='right')
    ax4.legend()

    for bar, c in zip(bars, blind_calmars):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{c:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('signal_calmar_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: signal_calmar_comparison.png")


def main():
    """Main execution."""
    df, results = analyze_all_signals()
    print_results_table(results)
    plot_signal_comparison(df, results)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nGenerated plots:")
    print("  - signal_sharpe_comparison.png")
    print("  - signal_equity_curves.png")
    print("  - signal_drawdown_comparison.png")
    print("  - signal_calmar_comparison.png")

    return df, results


if __name__ == '__main__':
    df, results = main()
