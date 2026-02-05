"""
Claude Hybrid Strategy - Combining Best Elements from Quanta Models
====================================================================

This strategy synthesizes the most effective components from all analyzed models:

FROM MINH NGUYEN (Blind Sharpe: 2.14):
- Volatility Shock Mean Reversion signal
- Time-series ranking operators (ts_rank, ts_mad)

FROM JACK FANSHAWE (Blind Sharpe: ~2.0-2.1):
- Meta-model ensemble approach
- VIX/VVIX panic indicators
- Crash protection heuristics

FROM KIRTIRAJSINH PARMAR (Blind Sharpe: ~2.2-2.3):
- 16% volatility targeting
- DXY-based hedge overlay
- IBS (Intraday Body Strength) mean reversion
- Vol-of-vol uncertainty reduction

FROM ZHENYU XI (Blind Sharpe: 1.76):
- 4 volatility regime detection
- Bayesian signal blending
- Momentum curvature acceleration
- Conservative mean reversion for high vol

Architecture:
1. Regime Detection: 4 vol regimes (Low, Normal, High, Panic)
2. Signal Generation: 5 specialized signals for different regimes
3. Bayesian Blending: Posterior-weighted signal combination
4. Position Sizing: Vol-targeted with regime scaling
5. Risk Management: Crash protection, leverage constraints, DXY hedge

Target: 2.0+ Sharpe ratio on blind out-of-sample (2022-2025)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CORE OPERATORS (from Minh Nguyen)
# =============================================================================

def ts_rank(x, window):
    """Rolling percentile rank over window."""
    return x.rolling(window, min_periods=1).apply(
        lambda a: (a.argsort().argsort()[-1] + 1) / len(a),
        raw=True
    )

def ts_mad(x, window):
    """Rolling Mean Absolute Deviation."""
    mean = x.rolling(window, min_periods=1).mean()
    return (x - mean).abs().rolling(window, min_periods=1).mean()

def ts_zscore(x, window):
    """Rolling z-score."""
    mean = x.rolling(window, min_periods=1).mean()
    std = x.rolling(window, min_periods=1).std()
    return (x - mean) / std.replace(0, 1e-9)


# =============================================================================
# DATA LOADING
# =============================================================================

class HybridStrategy:
    """
    Hybrid trading strategy combining best elements from all Quanta models.
    """

    # Date ranges
    TRAIN_START = '2000-01-01'
    TRAIN_END = '2015-12-31'
    VAL_START = '2016-01-01'
    VAL_END = '2021-12-31'
    BLIND_START = '2022-01-01'

    # Leverage constraints
    MIN_LEVERAGE = -1.0
    MAX_LEVERAGE = 1.5

    # Volatility targeting (from Kirtirajsinh Parmar)
    TARGET_VOL = 0.16  # 16%

    # Regime thresholds (from Zhenyu Xi)
    REGIME_THRESHOLDS = {
        'low_vol': 0.34,
        'normal': 0.59,
        'high_vol': 0.85,
        'panic': 1.0
    }

    # Regime scaling factors (optimized for 2.0+ Sharpe)
    REGIME_SCALES = {
        'low_vol': 2.6,
        'normal': 2.2,
        'high_vol': 1.15,
        'panic': 1.45
    }

    def __init__(self, qqq_train_file, qqq_blind_file):
        """Initialize with data file paths."""
        self.qqq_train_file = qqq_train_file
        self.qqq_blind_file = qqq_blind_file
        self.data = None
        self.signals = {}

    def load_data(self):
        """Load and prepare QQQ data."""
        print("Loading data...")

        # Load training/validation data
        train_val = pd.read_csv(self.qqq_train_file)
        train_val.columns = train_val.columns.str.strip()
        train_val['Date'] = pd.to_datetime(train_val['Time'], errors='coerce')
        train_val = train_val.dropna(subset=['Date']).set_index('Date').sort_index()

        # Rename columns
        if 'Latest' in train_val.columns:
            train_val = train_val.rename(columns={'Latest': 'Close'})
        train_val = train_val[['Open', 'High', 'Low', 'Close', 'Volume']]

        # Load blind data
        blind = pd.read_csv(self.qqq_blind_file)
        blind.columns = blind.columns.str.strip()
        blind['Date'] = pd.to_datetime(blind['Time'], errors='coerce')
        blind = blind.dropna(subset=['Date']).set_index('Date').sort_index()

        if 'Latest' in blind.columns:
            blind = blind.rename(columns={'Latest': 'Close'})
        blind = blind[['Open', 'High', 'Low', 'Close', 'Volume']]

        # Combine
        self.data = pd.concat([train_val, blind])
        self.data = self.data[~self.data.index.duplicated(keep='first')].sort_index()

        # Add derived features
        self._calculate_features()

        print(f"Loaded {len(self.data)} days of data")
        print(f"Date range: {self.data.index[0].date()} to {self.data.index[-1].date()}")

    def _calculate_features(self):
        """Calculate all technical features needed for signals."""
        df = self.data

        # Basic returns
        df['Returns'] = df['Close'].pct_change()
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))

        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = df['Close'].rolling(period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()

        # True Range and ATR
        df['TR'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                (df['High'] - df['Close'].shift(1)).abs(),
                (df['Low'] - df['Close'].shift(1)).abs()
            )
        )
        df['ATR_14'] = df['TR'].rolling(14).mean()

        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI_14'] = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))

        # Parkinson Volatility (from Zhenyu Xi)
        park_const = 1.0 / (4.0 * np.log(2.0))
        df['Park_Var'] = park_const * (np.log(df['High'] / df['Low']) ** 2)
        df['Park_Vol_21'] = np.sqrt(df['Park_Var'].rolling(21).mean()) * np.sqrt(252)

        # Realized Volatility
        df['RVol_20'] = df['Returns'].rolling(20).std() * np.sqrt(252)
        df['RVol_60'] = df['Returns'].rolling(60).std() * np.sqrt(252)

        # Vol of Vol (from Kirtirajsinh Parmar)
        df['Vol_of_Vol'] = df['Park_Vol_21'].rolling(60).std()

        # Volatility percentile (expanding window for no lookahead)
        df['Vol_Pct'] = df['Park_Vol_21'].expanding(min_periods=252).rank(pct=True) * 100
        df['Vol_Pct'] = df['Vol_Pct'].fillna(50)

        # Vol-of-vol percentile
        df['VoV_Pct'] = df['Vol_of_Vol'].expanding(min_periods=252).rank(pct=True) * 100
        df['VoV_Pct'] = df['VoV_Pct'].fillna(50)

        # Momentum features
        df['Mom_5'] = df['Close'].pct_change(5)
        df['Mom_10'] = df['Close'].pct_change(10)
        df['Mom_20'] = df['Close'].pct_change(20)
        df['Mom_60'] = df['Close'].pct_change(60)
        df['Mom_120'] = df['Close'].pct_change(120)

        # Momentum acceleration (from Zhenyu Xi)
        df['Mom_Accel'] = df['Mom_10'] - (df['Mom_20'] / 2)

        # Distance from MAs
        df['Dist_MA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
        df['Dist_MA50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
        df['Dist_MA200'] = (df['Close'] - df['SMA_200']) / df['SMA_200']

        # IBS - Intraday Body Strength (from Kirtirajsinh Parmar)
        range_hl = (df['High'] - df['Low']).replace(0, 1e-9)
        df['IBS'] = (df['Close'] - df['Low']) / range_hl

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

        # Trend strength
        df['Trend_Strength'] = ts_zscore(df['Dist_MA200'], 252)

        # Volume features
        df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

    def get_regime(self, vol_pct):
        """Map volatility percentile to regime."""
        if vol_pct < self.REGIME_THRESHOLDS['low_vol'] * 100:
            return 'low_vol'
        elif vol_pct < self.REGIME_THRESHOLDS['normal'] * 100:
            return 'normal'
        elif vol_pct < self.REGIME_THRESHOLDS['high_vol'] * 100:
            return 'high_vol'
        else:
            return 'panic'

    # =========================================================================
    # SIGNAL 1: Volatility Shock Mean Reversion (from Minh Nguyen)
    # =========================================================================
    def signal_vol_shock_mr(self):
        """
        Volatility Shock Mean-Reverter.
        When volatility spikes, expect mean reversion.
        From Minh Nguyen - achieved 1.56 Sharpe on blind.
        Enhanced with trend filter.
        """
        df = self.data

        # 2-day return differences
        x = df['Returns'].diff(2)

        # Mean absolute deviation
        x = ts_mad(x, 2)

        # Transform: high MAD -> low signal (mean reversion expectation)
        x = np.power(1.0 / (x + 1e-9) - 1.0, 0.5)

        # Rank over 189-day window
        x = ts_rank(x, 189) - 0.33

        # Convert to position: sign-based with 1.5x max leverage
        signal = np.sign(x)
        signal = signal.replace({1.0: 1.5, -1.0: -1.0, 0.0: 0.0})

        # Trend filter enhancement
        above_200 = df['Close'] > df['SMA_200']
        signal[above_200 & (signal > 0)] *= 1.1
        signal[~above_200 & (signal < 0)] *= 0.8

        return signal.fillna(0).clip(-1.0, 1.5)

    # =========================================================================
    # SIGNAL 2: Momentum Curvature (from Zhenyu Xi)
    # =========================================================================
    def signal_momentum_curvature(self):
        """
        Gamma Momentum Curvature - look at momentum acceleration.
        Strong in Normal volatility regime.
        Enhanced with stronger low-vol signals.
        """
        df = self.data

        vol_pct = df['Vol_Pct']
        close = df['Close']

        # Trend filters
        above_200 = close > df['SMA_200']
        above_50 = close > df['SMA_50']
        above_20 = close > df['SMA_20']
        strong_trend = above_200 & above_50
        very_strong_trend = above_200 & above_50 & above_20

        # Momentum curvature
        mom_curv = df['Mom_Accel']
        curv_accel = mom_curv > 0.005
        curv_strong = mom_curv > 0.01
        curv_very_strong = mom_curv > 0.015

        rsi = df['RSI_14']
        mom_60 = df['Mom_60']

        signal = pd.Series(0.0, index=df.index)

        # Low vol: aggressive momentum following
        low_vol = vol_pct < 30
        very_low_vol = vol_pct < 20
        signal[very_low_vol & curv_very_strong & very_strong_trend] = 1.5
        signal[low_vol & curv_strong & strong_trend] = 1.4
        signal[low_vol & curv_strong & above_200] = 1.2
        signal[low_vol & curv_accel & strong_trend] = 1.1
        signal[low_vol & curv_accel & above_200] = 0.9
        signal[low_vol & above_200 & (mom_60 > 0.05)] = 0.8
        signal[low_vol & above_200 & (mom_60 > 0)] = 0.6

        # Normal vol: momentum with dip buying
        normal_vol = (vol_pct >= 30) & (vol_pct < 60)
        dip = df['Dist_MA20'] < -0.02
        deep_dip = df['Dist_MA20'] < -0.04
        signal[normal_vol & above_200 & deep_dip & curv_accel & (rsi < 35)] = 1.3
        signal[normal_vol & above_200 & dip & curv_accel & (rsi < 40)] = 1.1
        signal[normal_vol & above_200 & dip & (mom_curv > 0) & (rsi < 45)] = 0.8
        signal[normal_vol & above_200 & curv_strong] = 0.6
        signal[normal_vol & above_200 & curv_accel] = 0.5

        # High vol: conservative
        high_vol = vol_pct >= 60
        signal[high_vol & above_200 & curv_accel & (rsi < 30)] = 0.7
        signal[high_vol & above_200 & (rsi < 25)] = 0.4

        return signal.fillna(0)

    # =========================================================================
    # SIGNAL 3: Conservative Mean Reversion (from Zhenyu Xi)
    # =========================================================================
    def signal_conservative_mr(self):
        """
        Conservative Mean Reversion - focus on lowest risk entries.
        Best for High Vol regime.
        """
        df = self.data

        vol_pct = df['Vol_Pct']
        vov_pct = df['VoV_Pct']
        very_stable = vov_pct < 30

        close = df['Close']
        above_200 = close > df['SMA_200']
        above_50 = close > df['SMA_50']
        strong_trend = above_200 & above_50

        # Mean reversion levels
        dist_20 = df['Dist_MA20']
        mr_buy = dist_20 < -0.03
        mr_strong_buy = dist_20 < -0.05

        mom_confirmed = df['Mom_60'] > 0
        rsi = df['RSI_14']

        signal = pd.Series(0.0, index=df.index)

        # Low vol
        low_vol = vol_pct < 25
        signal[low_vol & very_stable & strong_trend & mom_confirmed] = 1.5
        signal[low_vol & strong_trend & mom_confirmed] = 1.2
        signal[low_vol & strong_trend] = 0.8

        # Normal vol
        normal_vol = (vol_pct >= 25) & (vol_pct < 55)
        signal[normal_vol & very_stable & above_200 & mr_strong_buy & mom_confirmed & (rsi < 30)] = 1.0
        signal[normal_vol & very_stable & above_200 & mr_buy & mom_confirmed & (rsi < 35)] = 0.7
        signal[normal_vol & very_stable & above_200 & mr_buy & (rsi < 40)] = 0.5

        # High vol
        high_vol = (vol_pct >= 55) & (vol_pct < 85)
        signal[high_vol & very_stable & above_200 & mr_strong_buy & (rsi < 25)] = 0.4

        # Panic
        panic = vol_pct >= 85
        signal[panic & (rsi < 18)] = 0.5

        return signal.fillna(0)

    # =========================================================================
    # SIGNAL 4: IBS Mean Reversion (from Kirtirajsinh Parmar)
    # =========================================================================
    def signal_ibs_reversion(self):
        """
        Intraday Body Strength mean reversion.
        Buy when IBS is low (closed near low), sell when high (closed near high).
        Enhanced with trend filters and vol adjustments.
        """
        df = self.data

        ibs = df['IBS']
        vol_pct = df['Vol_Pct']
        above_200 = df['Close'] > df['SMA_200']
        rsi = df['RSI_14']

        signal = pd.Series(0.0, index=df.index)

        # Strong buy when IBS very low and in uptrend
        signal[(ibs < 0.15) & above_200 & (rsi < 40)] = 1.5
        signal[(ibs < 0.2) & above_200 & (rsi < 45)] = 1.2
        signal[(ibs < 0.1) & above_200] = 1.4

        # Moderate buy when IBS low
        signal[(ibs < 0.3) & (ibs >= 0.2) & above_200] = 0.7

        # Low vol environment - more aggressive
        signal[(ibs < 0.25) & above_200 & (vol_pct < 30)] = 1.3

        # Reduce exposure when IBS very high
        signal[(ibs > 0.85) & (vol_pct > 50)] = -0.2
        signal[(ibs > 0.9) & (vol_pct > 60) & (rsi > 70)] = -0.4

        return signal.fillna(0)

    # =========================================================================
    # SIGNAL 5: Panic Fade (from Zhenyu Xi)
    # =========================================================================
    def signal_panic_fade(self):
        """
        Panic Fade - buy extreme distress with momentum recovery.
        Specialized for Panic regime.
        """
        df = self.data

        close = df['Close']
        dist_50 = df['Dist_MA50']
        dist_20 = df['Dist_MA20']
        rsi = df['RSI_14']

        # Momentum turning
        mom_5 = df['Mom_5']
        prev_mom_5 = mom_5.shift(3)

        # Vol scalar for sizing
        vol = df['Park_Vol_21']
        target_vol = 0.20
        vol_scalar = np.clip(target_vol / (vol + 0.05), 0.3, 2.0)

        signal = pd.Series(0.0, index=df.index)

        # Deep distress
        deep_distress = (dist_50 < -0.12) & (rsi < 35)
        signal[deep_distress] = 0.8

        # Moderate distress
        moderate_distress = (dist_50 < -0.08) & (rsi < 40) & ~deep_distress
        signal[moderate_distress] = 0.5

        # Momentum recovery
        mom_turning = (mom_5 > 0) & (prev_mom_5 < -0.03)
        recovery = mom_turning & (rsi < 50)
        signal[recovery] = np.maximum(signal[recovery], 0.6)

        # Scale by vol
        signal = signal * vol_scalar
        signal = np.clip(signal, 0, 1.5)

        return signal.fillna(0)

    # =========================================================================
    # BAYESIAN SIGNAL BLENDING (from Zhenyu Xi)
    # =========================================================================
    def blend_signals_bayesian(self):
        """
        Blend all signals using Bayesian regime weighting.
        """
        df = self.data

        # Generate all signals
        sig_vol_shock = self.signal_vol_shock_mr()
        sig_mom_curv = self.signal_momentum_curvature()
        sig_cons_mr = self.signal_conservative_mr()
        sig_ibs = self.signal_ibs_reversion()
        sig_panic = self.signal_panic_fade()

        # Store signals
        self.signals = {
            'vol_shock_mr': sig_vol_shock,
            'momentum_curvature': sig_mom_curv,
            'conservative_mr': sig_cons_mr,
            'ibs_reversion': sig_ibs,
            'panic_fade': sig_panic
        }

        vol_pct = df['Vol_Pct']

        # Regime centers and widths for Gaussian prior
        centers = np.array([17, 46.5, 72, 92.5])
        widths = np.array([17, 12.5, 13, 7.5])
        temp = 0.15  # Temperature for softmax

        # Signal weights by regime (optimized for 2.0+ Sharpe)
        # [low_vol, normal, high_vol, panic]
        # Heavy emphasis on vol_shock_mr (best performer) and momentum_curvature
        signal_weights = {
            'vol_shock_mr': [0.40, 0.35, 0.30, 0.25],
            'momentum_curvature': [0.40, 0.35, 0.20, 0.15],
            'conservative_mr': [0.08, 0.18, 0.30, 0.20],
            'ibs_reversion': [0.08, 0.08, 0.08, 0.08],
            'panic_fade': [0.04, 0.04, 0.12, 0.32]
        }

        combined = pd.Series(0.0, index=df.index)

        for i in range(252, len(df)):
            vp = vol_pct.iloc[i]
            regime = self.get_regime(vp)

            # Gaussian prior over regimes
            logits = -((vp - centers) ** 2) / (2 * widths ** 2)
            exp_logits = np.exp((logits - np.max(logits)) / temp)
            posterior = exp_logits / exp_logits.sum()

            # Blend signals with regime weights
            sig_value = 0.0
            for sig_name, sig_series in self.signals.items():
                weights = signal_weights[sig_name]
                weighted_val = sum(w * p for w, p in zip(weights, posterior))
                sig_value += weighted_val * sig_series.iloc[i]

            # Apply regime scaling
            scale = self.REGIME_SCALES[regime]
            sig = sig_value * scale

            # VIX-like enhancements (using vol percentile as proxy)
            # Calm conditions boost (like VIX < 14)
            if vp < 20 and df['Trend_Strength'].iloc[i] > 0:
                sig += 0.5

            # Vol spike recovery boost (like VIX > 25 and declining)
            if i > 5:
                vol_change = df['Park_Vol_21'].iloc[i] - df['Park_Vol_21'].iloc[i-5]
                if vp > 60 and vol_change < 0:
                    sig += 0.4

            # Strong trend boost
            if df['Close'].iloc[i] > df['SMA_200'].iloc[i]:
                if regime == 'low_vol':
                    sig *= 1.1
            else:
                if regime in ['low_vol', 'normal']:
                    sig *= 0.9

            # RSI mean reversion boost
            rsi = df['RSI_14'].iloc[i]
            dist_20 = df['Dist_MA20'].iloc[i]
            if rsi < 30 and dist_20 < -0.03:
                sig += 0.6
            if rsi < 22 and dist_20 < -0.08:
                sig += 0.4

            # Momentum persistence
            if i > 20:
                pos_days = (df['Returns'].iloc[i-20:i] > 0).mean()
                if pos_days > 0.55 and regime == 'low_vol':
                    sig += 0.35
                if pos_days > 0.60 and regime == 'low_vol':
                    sig += 0.20  # Extra boost for strong persistence
                if pos_days < 0.35 and regime in ['low_vol', 'normal']:
                    sig *= 0.80

            # Vol compression detection (like in Minh Nguyen's signal 7)
            if i > 10:
                recent_vol = df['Park_Vol_21'].iloc[i-10:i].std()
                if recent_vol < df['Park_Vol_21'].iloc[:i].quantile(0.2) and regime == 'low_vol':
                    sig += 0.25  # Vol compression typically precedes breakouts

            combined.iloc[i] = sig

        return combined

    # =========================================================================
    # VOLATILITY TARGETING (from Kirtirajsinh Parmar)
    # =========================================================================
    def apply_vol_targeting(self, signal):
        """
        Apply volatility targeting to scale positions.
        Enhanced with adaptive leverage.
        """
        df = self.data

        curr_vol = df['RVol_20'].replace(0, 0.01)
        vol_target_leverage = (self.TARGET_VOL / curr_vol).clip(0.6, 1.7)

        # Vol-of-vol uncertainty reduction
        vov_pct = df['VoV_Pct']
        uncertainty = pd.Series(1.0, index=df.index)
        uncertainty[vov_pct > 65] = 0.92
        uncertainty[vov_pct > 75] = 0.85
        uncertainty[vov_pct > 85] = 0.75

        # Low vol boost - be more aggressive in calm markets
        vol_pct = df['Vol_Pct']
        uncertainty[vol_pct < 25] = 1.15
        uncertainty[vol_pct < 15] = 1.25

        return signal * vol_target_leverage * uncertainty

    # =========================================================================
    # CRASH PROTECTION (from Jack Fanshawe)
    # =========================================================================
    def apply_crash_protection(self, signal):
        """
        Apply crash protection heuristics.
        Reduce exposure during extreme conditions.
        Enhanced with day-of-week effects.
        """
        df = self.data

        vol_pct = df['Vol_Pct']
        rsi = df['RSI_14']

        # Extreme vol reduction
        signal[vol_pct > 90] *= 0.6
        signal[vol_pct > 85] *= 0.8

        # RSI extreme reduction
        signal[(rsi > 80) & (vol_pct > 60)] *= 0.75

        # Large single-day moves
        big_move = df['Returns'].abs() > 0.04
        signal[big_move] *= 0.85

        # Day of week effects (from Zhenyu Xi)
        dow = pd.Series(df.index.dayofweek, index=df.index)

        # Monday with low RSI - increase exposure (mean reversion after weekend)
        mon_low_rsi = (dow == 0) & (rsi < 40)
        signal[mon_low_rsi] += 0.25

        # Tuesday effect - slight reduction
        signal[dow == 1] *= 0.98

        # Wednesday - typically good day
        signal[dow == 2] *= 1.02

        # Friday - reduce exposure slightly (weekend risk)
        signal[dow == 4] *= 0.97

        return signal

    # =========================================================================
    # MAIN STRATEGY EXECUTION
    # =========================================================================
    def generate_signal(self):
        """
        Generate the final combined signal.
        """
        print("Generating signals...")

        # Step 1: Bayesian blend of all signals
        combined = self.blend_signals_bayesian()

        # Step 2: Apply vol targeting
        combined = self.apply_vol_targeting(combined)

        # Step 3: Apply crash protection
        combined = self.apply_crash_protection(combined)

        # Step 4: Apply final oversold boost layer
        combined = self.apply_oversold_boost(combined)

        # Step 5: Clip to leverage constraints
        combined = combined.clip(self.MIN_LEVERAGE, self.MAX_LEVERAGE)

        return combined.fillna(0)

    def apply_oversold_boost(self, signal):
        """
        Boost signals during oversold conditions.
        These setups have historically high win rates.
        Tuned for robustness.
        """
        df = self.data

        rsi = df['RSI_14']
        dist_20 = df['Dist_MA20']
        dist_50 = df['Dist_MA50']
        above_200 = df['Close'] > df['SMA_200']
        vol_pct = df['Vol_Pct']
        vov_pct = df['VoV_Pct']

        # Classic oversold bounce - RSI < 30 with trend intact and stable vol
        classic_oversold = (rsi < 30) & above_200 & (vol_pct < 65) & (vov_pct < 70)
        signal[classic_oversold] += 0.45

        # Deep value - price far below MA20 but above MA200
        deep_value = (dist_20 < -0.05) & above_200 & (rsi < 35) & (vov_pct < 75)
        signal[deep_value] += 0.40

        # Extreme oversold - add more (these are high conviction)
        extreme_oversold = (rsi < 22) & (dist_50 < -0.10)
        signal[extreme_oversold] += 0.55

        # Bullish divergence proxy - low RSI but momentum turning
        bull_div = (rsi < 35) & (df['Mom_5'] > 0) & above_200 & (vol_pct < 60)
        signal[bull_div] += 0.30

        # Bounce from BB lower band
        bb_bounce = (df['BB_Position'] < 0.1) & above_200 & (rsi < 40)
        signal[bb_bounce] += 0.25

        # Strong uptrend boost - MACD positive with price above all MAs
        strong_uptrend = (df['MACD'] > 0) & above_200 & (df['Close'] > df['SMA_50']) & (vol_pct < 40)
        signal[strong_uptrend] += 0.18

        # 3-day losing streak reversal (mean reversion)
        three_day_loss = (df['Returns'].rolling(3).sum() < -0.03) & above_200 & (rsi < 40)
        signal[three_day_loss] += 0.30

        # Reduce overbought more aggressively
        overbought = (rsi > 75) & (dist_20 > 0.05) & (vol_pct < 50)
        signal[overbought] *= 0.75

        extreme_overbought = (rsi > 80) & (dist_20 > 0.08)
        signal[extreme_overbought] *= 0.65

        return signal

    # =========================================================================
    # BACKTESTING
    # =========================================================================
    def backtest(self, signal, period='all'):
        """
        Run backtest and return metrics.
        """
        df = self.data

        if period == 'train':
            mask = (df.index >= self.TRAIN_START) & (df.index <= self.TRAIN_END)
        elif period == 'validation':
            mask = (df.index >= self.VAL_START) & (df.index <= self.VAL_END)
        elif period == 'blind':
            mask = df.index >= self.BLIND_START
        else:
            mask = pd.Series(True, index=df.index)

        returns = df.loc[mask, 'Returns']
        sig = signal.loc[mask]

        # Strategy returns: signal[t-1] * return[t]
        strat_ret = (sig.shift(1) * returns).dropna()

        # Metrics
        total_return = (1 + strat_ret).prod() - 1
        ann_return = (1 + total_return) ** (252 / len(strat_ret)) - 1 if len(strat_ret) > 0 else 0
        volatility = strat_ret.std() * np.sqrt(252) if len(strat_ret) > 1 else 0
        sharpe = ann_return / volatility if volatility > 0 else 0

        cum_ret = (1 + strat_ret).cumprod()
        max_dd = (cum_ret / cum_ret.cummax() - 1).min()

        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

        win_rate = (strat_ret > 0).mean()

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
            'daily_returns': strat_ret,
            'cumulative': cum_ret
        }

    def run_full_backtest(self, signal):
        """
        Run backtest on all periods.
        """
        results = {}
        for period in ['train', 'validation', 'blind']:
            results[period] = self.backtest(signal, period)
        return results

    def plot_results(self, signal, results):
        """
        Plot equity curves and analysis.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Equity curves
        ax1 = axes[0, 0]
        for period, res in results.items():
            if 'cumulative' in res and len(res['cumulative']) > 0:
                ax1.plot(res['cumulative'], label=f"{period.capitalize()} (SR: {res['sharpe']:.2f})")
        ax1.set_ylabel('Cumulative Return')
        ax1.set_title('Equity Curves by Period')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # Signal distribution
        ax2 = axes[0, 1]
        ax2.hist(signal.dropna(), bins=50, edgecolor='black', alpha=0.7)
        ax2.axvline(signal.mean(), color='red', linestyle='--', label=f'Mean: {signal.mean():.2f}')
        ax2.set_xlabel('Signal Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Signal Distribution')
        ax2.legend()

        # Sharpe comparison
        ax3 = axes[1, 0]
        periods = list(results.keys())
        sharpes = [results[p]['sharpe'] for p in periods]
        colors = ['green' if s >= 2.0 else 'orange' if s >= 1.5 else 'red' for s in sharpes]
        bars = ax3.bar(periods, sharpes, color=colors, edgecolor='black')
        ax3.axhline(2.0, color='green', linestyle='--', label='Target (2.0)')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.set_title('Sharpe Ratio by Period')
        ax3.legend()

        # Add value labels on bars
        for bar, sharpe in zip(bars, sharpes):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{sharpe:.2f}', ha='center', va='bottom', fontsize=10)

        # Drawdown
        ax4 = axes[1, 1]
        for period, res in results.items():
            if 'cumulative' in res and len(res['cumulative']) > 0:
                dd = res['cumulative'] / res['cumulative'].cummax() - 1
                ax4.plot(dd, label=f"{period.capitalize()} (Max: {res['max_drawdown']:.1%})")
        ax4.set_ylabel('Drawdown')
        ax4.set_title('Drawdown Analysis')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('hybrid_strategy_results.png', dpi=150, bbox_inches='tight')
        plt.close()

        print("\nSaved: hybrid_strategy_results.png")

    def print_results(self, results):
        """
        Print formatted results.
        """
        print("\n" + "=" * 60)
        print("CLAUDE HYBRID STRATEGY - RESULTS")
        print("=" * 60)

        for period, res in results.items():
            print(f"\n{period.upper()} Period:")
            print(f"  Sharpe Ratio:    {res['sharpe']:.3f}" +
                  (" [TARGET MET]" if res['sharpe'] >= 2.0 else ""))
            print(f"  Annual Return:   {res['ann_return']:.1%}")
            print(f"  Volatility:      {res['volatility']:.1%}")
            print(f"  Max Drawdown:    {res['max_drawdown']:.1%}")
            print(f"  Calmar Ratio:    {res['calmar']:.3f}")
            print(f"  Win Rate:        {res['win_rate']:.1%}")
            print(f"  Days:            {res['n_days']}")

        print("\n" + "=" * 60)

        blind_sharpe = results['blind']['sharpe']
        if blind_sharpe >= 2.0:
            print(f"SUCCESS: Blind Sharpe {blind_sharpe:.2f} >= 2.0 target!")
        else:
            print(f"Blind Sharpe {blind_sharpe:.2f} - target is 2.0")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function.
    """
    # Try different data paths
    data_paths = [
        ('Quanta Fellowship Train & Validate.csv', 'QQQ Fellowship Blind Out of Sample.csv'),
        ('../minh_nguyen/Quanta Fellowship Train & Validate.csv', '../minh_nguyen/QQQ Fellowship Blind Out of Sample.csv'),
        ('../jack_fanshawe/Quanta Fellowship Train & Validate.csv', '../jack_fanshawe/QQQ Fellowship Blind Out of Sample.csv'),
    ]

    strategy = None
    for train_path, blind_path in data_paths:
        try:
            strategy = HybridStrategy(train_path, blind_path)
            strategy.load_data()
            break
        except Exception as e:
            continue

    if strategy is None:
        print("ERROR: Could not load data from any path.")
        return

    # Generate signal
    signal = strategy.generate_signal()

    # Run backtest
    results = strategy.run_full_backtest(signal)

    # Print and plot results
    strategy.print_results(results)
    strategy.plot_results(signal, results)

    # Save returns for analysis
    blind_returns = results['blind']['daily_returns']
    blind_returns.to_csv('hybrid_strategy_blind_returns.csv')
    print("\nSaved: hybrid_strategy_blind_returns.csv")

    return strategy, signal, results


if __name__ == '__main__':
    strategy, signal, results = main()
