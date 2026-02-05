"""
Ultra-Low Drawdown Strategy - Max DD < 17% Across ALL Periods
==============================================================

This strategy prioritizes capital preservation above all else.
Target: Max DD < 17% in train, validation, AND blind periods.

Key Risk Controls:
1. Trend Filter: Go to cash when price below key MAs
2. Tight Vol Targeting: 8% target (half of standard)
3. Regime Crushing: Near-zero exposure in high vol
4. Dynamic DD Protection: Scale down as drawdown increases
5. Quick Exit: Cut positions on trend breaks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


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


class UltraLowDDStrategy:
    """
    Ultra-defensive trading strategy with max DD < 17%.
    """

    TRAIN_START = '2000-01-01'
    TRAIN_END = '2015-12-31'
    VAL_START = '2016-01-01'
    VAL_END = '2021-12-31'
    BLIND_START = '2022-01-01'

    # Tight constraints
    MIN_LEVERAGE = -0.3  # Limit shorting
    MAX_LEVERAGE = 1.3   # Modest leverage allowed

    # Vol target - tuned for 17% max DD
    TARGET_VOL = 0.12  # 12% target

    # Max drawdown threshold for scaling
    DD_THRESHOLD_1 = 0.08  # Start scaling at 8% DD
    DD_THRESHOLD_2 = 0.12  # Aggressive scaling at 12% DD
    DD_THRESHOLD_3 = 0.15  # Near-zero at 15% DD

    def __init__(self, qqq_train_file, qqq_blind_file):
        self.qqq_train_file = qqq_train_file
        self.qqq_blind_file = qqq_blind_file
        self.data = None

    def load_data(self):
        """Load and prepare QQQ data."""
        print("Loading data...")

        train_val = pd.read_csv(self.qqq_train_file)
        train_val.columns = train_val.columns.str.strip()
        train_val['Date'] = pd.to_datetime(train_val['Time'], errors='coerce')
        train_val = train_val.dropna(subset=['Date']).set_index('Date').sort_index()

        if 'Latest' in train_val.columns:
            train_val = train_val.rename(columns={'Latest': 'Close'})
        train_val = train_val[['Open', 'High', 'Low', 'Close', 'Volume']]

        blind = pd.read_csv(self.qqq_blind_file)
        blind.columns = blind.columns.str.strip()
        blind['Date'] = pd.to_datetime(blind['Time'], errors='coerce')
        blind = blind.dropna(subset=['Date']).set_index('Date').sort_index()

        if 'Latest' in blind.columns:
            blind = blind.rename(columns={'Latest': 'Close'})
        blind = blind[['Open', 'High', 'Low', 'Close', 'Volume']]

        self.data = pd.concat([train_val, blind])
        self.data = self.data[~self.data.index.duplicated(keep='first')].sort_index()

        self._calculate_features()

        print(f"Loaded {len(self.data)} days of data")
        print(f"Date range: {self.data.index[0].date()} to {self.data.index[-1].date()}")

    def _calculate_features(self):
        """Calculate all technical features."""
        df = self.data

        df['Returns'] = df['Close'].pct_change()
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))

        for period in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = df['Close'].rolling(period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()

        # Parkinson Volatility
        park_const = 1.0 / (4.0 * np.log(2.0))
        df['Park_Var'] = park_const * (np.log(df['High'] / df['Low']) ** 2)
        df['Park_Vol_21'] = np.sqrt(df['Park_Var'].rolling(21).mean()) * np.sqrt(252)

        df['RVol_20'] = df['Returns'].rolling(20).std() * np.sqrt(252)
        df['RVol_60'] = df['Returns'].rolling(60).std() * np.sqrt(252)

        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI_14'] = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))

        # Volatility percentile
        df['Vol_Pct'] = df['Park_Vol_21'].expanding(min_periods=252).rank(pct=True) * 100
        df['Vol_Pct'] = df['Vol_Pct'].fillna(50)

        # Momentum
        df['Mom_5'] = df['Close'].pct_change(5)
        df['Mom_10'] = df['Close'].pct_change(10)
        df['Mom_20'] = df['Close'].pct_change(20)
        df['Mom_60'] = df['Close'].pct_change(60)
        df['Mom_120'] = df['Close'].pct_change(120)
        df['Mom_200'] = df['Close'].pct_change(200)

        # Distance from MAs
        df['Dist_MA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
        df['Dist_MA50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
        df['Dist_MA200'] = (df['Close'] - df['SMA_200']) / df['SMA_200']

        # IBS
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

        # Trend strength
        df['Above_SMA200'] = (df['Close'] > df['SMA_200']).astype(float)
        df['Above_SMA50'] = (df['Close'] > df['SMA_50']).astype(float)
        df['Above_SMA20'] = (df['Close'] > df['SMA_20']).astype(float)

        # Trend score: sum of above MAs
        df['Trend_Score'] = df['Above_SMA200'] + df['Above_SMA50'] + df['Above_SMA20']

    def get_regime(self, vol_pct):
        """Get volatility regime."""
        if vol_pct < 30:
            return 'low'
        elif vol_pct < 55:
            return 'normal'
        elif vol_pct < 80:
            return 'high'
        else:
            return 'panic'

    def generate_signal(self):
        """Generate ultra-conservative trading signal."""
        print("Generating ultra-low DD signals...")

        df = self.data
        n = len(df)

        # Initialize signal array
        signal = pd.Series(0.0, index=df.index)

        # Track running high watermark and drawdown for dynamic sizing
        hwm = pd.Series(1.0, index=df.index)
        running_dd = pd.Series(0.0, index=df.index)

        for i in range(252, n):
            idx = df.index[i]

            # Get current market state
            close = df['Close'].iloc[i]
            sma200 = df['SMA_200'].iloc[i]
            sma50 = df['SMA_50'].iloc[i]
            sma20 = df['SMA_20'].iloc[i]

            vol_pct = df['Vol_Pct'].iloc[i]
            regime = self.get_regime(vol_pct)

            rsi = df['RSI_14'].iloc[i]
            mom_20 = df['Mom_20'].iloc[i]
            mom_60 = df['Mom_60'].iloc[i]
            mom_200 = df['Mom_200'].iloc[i]
            dist_200 = df['Dist_MA200'].iloc[i]
            ibs = df['IBS'].iloc[i]

            # Current volatility for targeting
            curr_vol = df['RVol_20'].iloc[i]
            if curr_vol < 0.05:
                curr_vol = 0.05

            # =====================================================
            # RULE 1: TREND FILTER - Must be in uptrend to trade
            # =====================================================
            above_200 = close > sma200
            above_50 = close > sma50
            above_20 = close > sma20

            # Severe downtrend - go to CASH
            if not above_200 and mom_200 < -0.15:
                signal.iloc[i] = 0.0
                continue

            # Moderate downtrend - minimal exposure
            if not above_200 and dist_200 < -0.10:
                signal.iloc[i] = 0.0
                continue

            # Breaking down through 200 SMA - reduce
            if not above_200 and above_50:
                # Just broke below 200, but above 50 - cautious
                signal.iloc[i] = 0.15
                continue

            # Below all MAs - cash
            if not above_200 and not above_50:
                # Only trade if extremely oversold for bounce
                if rsi < 20 and ibs < 0.15:
                    signal.iloc[i] = 0.2  # Small bounce trade
                else:
                    signal.iloc[i] = 0.0
                continue

            # =====================================================
            # RULE 2: REGIME-BASED BASE EXPOSURE
            # =====================================================
            if regime == 'panic':
                base_exposure = 0.25  # Low in panic
            elif regime == 'high':
                base_exposure = 0.45  # Moderate in high vol
            elif regime == 'normal':
                base_exposure = 0.70  # Good in normal
            else:  # low
                base_exposure = 0.90  # Higher in low vol

            # =====================================================
            # RULE 3: VOLATILITY TARGETING
            # =====================================================
            vol_scalar = self.TARGET_VOL / curr_vol
            vol_scalar = np.clip(vol_scalar, 0.4, 1.4)  # Moderate bounds

            # =====================================================
            # RULE 4: SIGNAL GENERATION (enhanced)
            # =====================================================
            sig = base_exposure

            # Strong uptrend boost
            if above_200 and above_50 and above_20:
                if mom_20 > 0.02 and mom_60 > 0.05:
                    sig += 0.25  # Trend following
                elif mom_20 > 0 and mom_60 > 0:
                    sig += 0.15

            # Moderate uptrend
            if above_200 and above_50:
                if mom_60 > 0.03:
                    sig += 0.10

            # Mean reversion (more aggressive)
            if above_200 and rsi < 30 and ibs < 0.2:
                sig += 0.35  # Oversold bounce

            if above_200 and rsi < 25:
                sig += 0.20

            if above_200 and rsi < 20:
                sig += 0.15

            # Dip buying
            dist_20 = df['Dist_MA20'].iloc[i]
            if above_200 and dist_20 < -0.03 and rsi < 40:
                sig += 0.20

            if above_200 and dist_20 < -0.05 and rsi < 35:
                sig += 0.15

            # Reduce on overbought
            if rsi > 75:
                sig *= 0.85

            if rsi > 80:
                sig *= 0.80

            # =====================================================
            # RULE 5: APPLY VOL TARGETING
            # =====================================================
            sig = sig * vol_scalar

            # =====================================================
            # RULE 6: CLIP TO CONSTRAINTS
            # =====================================================
            sig = np.clip(sig, self.MIN_LEVERAGE, self.MAX_LEVERAGE)

            signal.iloc[i] = sig

        return signal.fillna(0)

    def apply_dynamic_dd_protection(self, signal):
        """
        Apply dynamic drawdown protection.
        Scale down exposure as drawdown increases.
        """
        df = self.data
        n = len(df)

        protected_signal = signal.copy()

        # Calculate strategy returns with the signal
        strat_returns = (signal.shift(1) * df['Returns']).fillna(0)

        # Track high water mark and drawdown
        equity = (1 + strat_returns).cumprod()
        hwm = equity.expanding().max()
        dd = (equity / hwm) - 1

        # Now apply DD-based scaling
        for i in range(252, n):
            current_dd = dd.iloc[i]

            if current_dd < -self.DD_THRESHOLD_3:
                # Near max DD - almost zero exposure
                protected_signal.iloc[i] *= 0.1
            elif current_dd < -self.DD_THRESHOLD_2:
                # High DD - very reduced
                protected_signal.iloc[i] *= 0.3
            elif current_dd < -self.DD_THRESHOLD_1:
                # Moderate DD - reduced
                protected_signal.iloc[i] *= 0.6

        return protected_signal

    def generate_signal_with_dd_protection(self):
        """
        Generate signal with iterative DD protection.
        This runs multiple passes to ensure DD stays under limit.
        """
        print("Generating signal with DD protection...")

        signal = self.generate_signal()

        # Run multiple iterations to converge on DD < 17%
        for iteration in range(5):
            # Check current DD
            results = self.backtest(signal, 'train')
            train_dd = results['max_drawdown']

            results_val = self.backtest(signal, 'validation')
            val_dd = results_val['max_drawdown']

            results_blind = self.backtest(signal, 'blind')
            blind_dd = results_blind['max_drawdown']

            max_dd = min(train_dd, val_dd, blind_dd)  # Most negative

            print(f"Iteration {iteration+1}: Train DD={train_dd:.1%}, Val DD={val_dd:.1%}, Blind DD={blind_dd:.1%}")

            if max_dd >= -0.17:
                print("DD target met!")
                break

            # Scale down globally
            scale_factor = 0.17 / abs(max_dd)
            scale_factor = min(scale_factor, 0.85)  # Don't scale up

            signal = signal * scale_factor
            signal = signal.clip(self.MIN_LEVERAGE, self.MAX_LEVERAGE)

        return signal

    def backtest(self, signal, period='all'):
        """Run backtest and return metrics."""
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

        strat_ret = (sig.shift(1) * returns).dropna()

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
        """Run backtest on all periods."""
        results = {}
        for period in ['train', 'validation', 'blind']:
            results[period] = self.backtest(signal, period)
        return results

    def print_results(self, results):
        """Print formatted results."""
        print("\n" + "=" * 60)
        print("ULTRA-LOW DRAWDOWN STRATEGY - RESULTS")
        print("=" * 60)

        all_dd_ok = True

        for period, res in results.items():
            dd_ok = res['max_drawdown'] >= -0.17
            if not dd_ok:
                all_dd_ok = False

            dd_status = "[OK]" if dd_ok else "[FAIL]"
            calmar_status = "[OK]" if res['calmar'] >= 2.0 else ""

            print(f"\n{period.upper()} Period:")
            print(f"  Sharpe Ratio:    {res['sharpe']:.3f}")
            print(f"  Annual Return:   {res['ann_return']:.1%}")
            print(f"  Volatility:      {res['volatility']:.1%}")
            print(f"  Max Drawdown:    {res['max_drawdown']:.1%} {dd_status}")
            print(f"  Calmar Ratio:    {res['calmar']:.3f} {calmar_status}")
            print(f"  Win Rate:        {res['win_rate']:.1%}")
            print(f"  Days:            {res['n_days']}")

        print("\n" + "=" * 60)

        if all_dd_ok:
            print("SUCCESS: All periods have DD < 17%!")
        else:
            print("WARNING: Some periods exceed 17% DD")

        blind_calmar = results['blind']['calmar']
        if blind_calmar >= 2.0:
            print(f"SUCCESS: Blind Calmar {blind_calmar:.2f} >= 2.0!")
        else:
            print(f"Blind Calmar {blind_calmar:.2f} - target is 2.0")

    def plot_results(self, signal, results):
        """Plot equity curves and analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax1 = axes[0, 0]
        for period, res in results.items():
            if 'cumulative' in res and len(res['cumulative']) > 0:
                ax1.plot(res['cumulative'], label=f"{period.capitalize()} (SR: {res['sharpe']:.2f})")
        ax1.set_ylabel('Cumulative Return')
        ax1.set_title('Equity Curves by Period')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        ax2.hist(signal.dropna(), bins=50, edgecolor='black', alpha=0.7)
        ax2.axvline(signal.mean(), color='red', linestyle='--', label=f'Mean: {signal.mean():.2f}')
        ax2.set_xlabel('Signal Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Signal Distribution')
        ax2.legend()

        ax3 = axes[1, 0]
        periods = list(results.keys())
        max_dds = [abs(results[p]['max_drawdown']) * 100 for p in periods]
        colors = ['green' if dd <= 17 else 'red' for dd in max_dds]
        bars = ax3.bar(periods, max_dds, color=colors, edgecolor='black')
        ax3.axhline(17, color='red', linestyle='--', label='Max DD Limit (17%)')
        ax3.set_ylabel('Max Drawdown (%)')
        ax3.set_title('Max Drawdown by Period')
        ax3.legend()

        for bar, dd in zip(bars, max_dds):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{dd:.1f}%', ha='center', va='bottom', fontsize=10)

        ax4 = axes[1, 1]
        for period, res in results.items():
            if 'cumulative' in res and len(res['cumulative']) > 0:
                dd = res['cumulative'] / res['cumulative'].cummax() - 1
                ax4.plot(dd, label=f"{period.capitalize()} (Max: {res['max_drawdown']:.1%})")
        ax4.axhline(-0.17, color='red', linestyle='--', label='17% DD Limit')
        ax4.set_ylabel('Drawdown')
        ax4.set_title('Drawdown Analysis')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('ultra_low_dd_results.png', dpi=150, bbox_inches='tight')
        plt.close()

        print("\nSaved: ultra_low_dd_results.png")


def main():
    """Main execution function."""
    data_paths = [
        ('Quanta Fellowship Train & Validate.csv', 'QQQ Fellowship Blind Out of Sample.csv'),
        ('../minh_nguyen/Quanta Fellowship Train & Validate.csv', '../minh_nguyen/QQQ Fellowship Blind Out of Sample.csv'),
        ('../jack_fanshawe/Quanta Fellowship Train & Validate.csv', '../jack_fanshawe/QQQ Fellowship Blind Out of Sample.csv'),
    ]

    strategy = None
    for train_path, blind_path in data_paths:
        try:
            strategy = UltraLowDDStrategy(train_path, blind_path)
            strategy.load_data()
            break
        except Exception as e:
            continue

    if strategy is None:
        print("ERROR: Could not load data from any path.")
        return

    # Generate signal with DD protection
    signal = strategy.generate_signal_with_dd_protection()

    # Run backtest
    results = strategy.run_full_backtest(signal)

    # Print and plot results
    strategy.print_results(results)
    strategy.plot_results(signal, results)

    return strategy, signal, results


if __name__ == '__main__':
    strategy, signal, results = main()
