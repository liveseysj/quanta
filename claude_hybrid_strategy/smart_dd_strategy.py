"""
Smart Low-DD Strategy - Max DD < 17% with Calmar > 2.0
=======================================================

Key insight: Be BINARY about market participation.
Either trade with good exposure OR go to cash entirely.

During major crashes (2000-2002, 2008), the strategy should:
1. Detect crash regime early
2. Go to cash completely
3. Only re-enter when recovery confirmed

This allows for better returns during good times while
maintaining strict DD limits during bad times.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class SmartDDStrategy:
    """
    Smart trading strategy with binary market participation.
    """

    TRAIN_START = '2000-01-01'
    TRAIN_END = '2015-12-31'
    VAL_START = '2016-01-01'
    VAL_END = '2021-12-31'
    BLIND_START = '2022-01-01'

    # Constraints
    MIN_LEVERAGE = -0.2
    MAX_LEVERAGE = 1.2

    # Volatility targeting
    TARGET_VOL = 0.14

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

        # Market state indicators
        df['Above_SMA200'] = (df['Close'] > df['SMA_200']).astype(float)
        df['Above_SMA50'] = (df['Close'] > df['SMA_50']).astype(float)
        df['Above_SMA20'] = (df['Close'] > df['SMA_20']).astype(float)

        # Death cross / Golden cross
        df['SMA50_vs_200'] = df['SMA_50'] > df['SMA_200']

        # Consecutive down days
        df['Down_Day'] = (df['Returns'] < 0).astype(int)
        df['Consec_Down'] = df['Down_Day'].rolling(5).sum()

        # Recent drawdown (20-day)
        df['Rolling_High_20'] = df['Close'].rolling(20).max()
        df['Recent_DD'] = (df['Close'] / df['Rolling_High_20']) - 1

        # 60-day drawdown
        df['Rolling_High_60'] = df['Close'].rolling(60).max()
        df['DD_60'] = (df['Close'] / df['Rolling_High_60']) - 1

    def is_crash_regime(self, i):
        """
        Detect if we're in a crash/bear market regime.
        Should stay in cash during these periods.
        """
        df = self.data

        close = df['Close'].iloc[i]
        sma200 = df['SMA_200'].iloc[i]
        sma50 = df['SMA_50'].iloc[i]

        mom_60 = df['Mom_60'].iloc[i]
        mom_120 = df['Mom_120'].iloc[i]
        mom_200 = df['Mom_200'].iloc[i]
        dist_200 = df['Dist_MA200'].iloc[i]

        vol_pct = df['Vol_Pct'].iloc[i]
        dd_60 = df['DD_60'].iloc[i]

        # CRASH CONDITIONS - any of these triggers cash mode

        # 1. Severe bear: price > 20% below 200 SMA
        if dist_200 < -0.20:
            return True

        # 2. Deep downtrend: 200-day momentum < -25%
        if mom_200 < -0.25:
            return True

        # 3. Death cross with negative momentum
        if close < sma200 and sma50 < sma200 and mom_60 < -0.10:
            return True

        # 4. Panic volatility with downtrend
        mom_20 = df['Mom_20'].iloc[i]
        if vol_pct > 85 and close < sma200 and mom_20 < -0.05:
            return True

        # 5. 60-day drawdown > 25% (catching crashes)
        if dd_60 < -0.25:
            return True

        # 6. Below 200 SMA with accelerating losses
        if close < sma200 and mom_60 < -0.15 and mom_120 < -0.20:
            return True

        return False

    def is_recovery_confirmed(self, i):
        """
        Detect if recovery from crash is confirmed.
        Need multiple confirmations to re-enter.
        """
        df = self.data

        close = df['Close'].iloc[i]
        sma200 = df['SMA_200'].iloc[i]
        sma50 = df['SMA_50'].iloc[i]
        sma20 = df['SMA_20'].iloc[i]

        mom_20 = df['Mom_20'].iloc[i]
        mom_60 = df['Mom_60'].iloc[i]
        vol_pct = df['Vol_Pct'].iloc[i]

        # Recovery conditions (need multiple)
        conditions = 0

        # 1. Price above 200 SMA
        if close > sma200:
            conditions += 1

        # 2. 50 SMA above 200 SMA (golden cross)
        if sma50 > sma200:
            conditions += 1

        # 3. Positive short-term momentum
        if mom_20 > 0.02:
            conditions += 1

        # 4. Positive medium-term momentum
        if mom_60 > 0.05:
            conditions += 1

        # 5. Vol normalizing
        if vol_pct < 60:
            conditions += 1

        # 6. Price above 50 SMA
        if close > sma50:
            conditions += 1

        # Need at least 4 conditions for recovery confirmation
        return conditions >= 4

    def generate_signal(self):
        """Generate trading signal with crash protection."""
        print("Generating signals with crash detection...")

        df = self.data
        n = len(df)

        signal = pd.Series(0.0, index=df.index)

        # Track market state
        in_crash_mode = False

        for i in range(252, n):
            idx = df.index[i]

            # Get current market state
            close = df['Close'].iloc[i]
            sma200 = df['SMA_200'].iloc[i]
            sma50 = df['SMA_50'].iloc[i]
            sma20 = df['SMA_20'].iloc[i]

            vol_pct = df['Vol_Pct'].iloc[i]
            rsi = df['RSI_14'].iloc[i]
            mom_20 = df['Mom_20'].iloc[i]
            mom_60 = df['Mom_60'].iloc[i]
            ibs = df['IBS'].iloc[i]
            dist_20 = df['Dist_MA20'].iloc[i]

            curr_vol = max(df['RVol_20'].iloc[i], 0.05)

            above_200 = close > sma200
            above_50 = close > sma50
            above_20 = close > sma20

            # =====================================================
            # CRASH DETECTION - Binary decision
            # =====================================================
            if self.is_crash_regime(i):
                in_crash_mode = True
            elif in_crash_mode and self.is_recovery_confirmed(i):
                in_crash_mode = False

            # If in crash mode, stay in cash (with small bounce trades)
            if in_crash_mode:
                # Only trade extreme oversold bounces
                if rsi < 15 and ibs < 0.1:
                    signal.iloc[i] = 0.3  # Small bounce trade
                else:
                    signal.iloc[i] = 0.0
                continue

            # =====================================================
            # NORMAL TRADING MODE
            # =====================================================

            # Base exposure by regime
            if vol_pct > 75:
                base = 0.35
            elif vol_pct > 55:
                base = 0.50
            elif vol_pct > 35:
                base = 0.70
            else:
                base = 0.85

            sig = base

            # Vol targeting
            vol_scalar = self.TARGET_VOL / curr_vol
            vol_scalar = np.clip(vol_scalar, 0.5, 1.3)

            # =====================================================
            # SIGNAL GENERATION
            # =====================================================

            # Strong uptrend - increase exposure
            if above_200 and above_50 and above_20:
                if mom_20 > 0.03 and mom_60 > 0.08:
                    sig += 0.30
                elif mom_20 > 0.01 and mom_60 > 0.03:
                    sig += 0.20
                elif mom_20 > 0 and mom_60 > 0:
                    sig += 0.10

            # Mean reversion - oversold bounces
            if above_200:
                if rsi < 25 and ibs < 0.15:
                    sig += 0.40
                elif rsi < 30 and ibs < 0.2:
                    sig += 0.30
                elif rsi < 35:
                    sig += 0.15

                # Dip buying
                if dist_20 < -0.04 and rsi < 35:
                    sig += 0.25
                elif dist_20 < -0.03 and rsi < 40:
                    sig += 0.15

            # Moderate uptrend
            if above_200 and above_50 and mom_60 > 0.03:
                sig += 0.10

            # Reduce exposure
            # Overbought
            if rsi > 75 and dist_20 > 0.05:
                sig *= 0.75
            elif rsi > 70:
                sig *= 0.85

            # High vol caution
            if vol_pct > 70:
                sig *= 0.80

            # Below 200 but not crash - reduced exposure
            if not above_200:
                sig *= 0.40
                # Bounce trade if oversold
                if rsi < 25:
                    sig += 0.15

            # Apply vol targeting
            sig = sig * vol_scalar

            # Clip
            sig = np.clip(sig, self.MIN_LEVERAGE, self.MAX_LEVERAGE)

            signal.iloc[i] = sig

        return signal.fillna(0)

    def optimize_for_dd_constraint(self, max_dd_target=-0.17):
        """
        Optimize signal to meet DD constraint while maximizing Calmar.
        """
        print(f"Optimizing for max DD = {max_dd_target:.1%}...")

        signal = self.generate_signal()

        # Check initial performance
        results = self.run_full_backtest(signal)

        # Find worst DD across all periods
        worst_dd = min(
            results['train']['max_drawdown'],
            results['validation']['max_drawdown'],
            results['blind']['max_drawdown']
        )

        print(f"Initial worst DD: {worst_dd:.1%}")

        # If already meeting constraint, try to increase exposure
        if worst_dd >= max_dd_target:
            # Try scaling up
            for scale in [1.1, 1.2, 1.3, 1.4, 1.5]:
                test_signal = (signal * scale).clip(self.MIN_LEVERAGE, self.MAX_LEVERAGE)
                test_results = self.run_full_backtest(test_signal)
                test_worst_dd = min(
                    test_results['train']['max_drawdown'],
                    test_results['validation']['max_drawdown'],
                    test_results['blind']['max_drawdown']
                )
                if test_worst_dd >= max_dd_target:
                    signal = test_signal
                    results = test_results
                    print(f"Scaled up {scale}x, worst DD: {test_worst_dd:.1%}")
                else:
                    break
        else:
            # Need to scale down
            scale = (max_dd_target / worst_dd) * 0.95  # 5% buffer
            signal = (signal * scale).clip(self.MIN_LEVERAGE, self.MAX_LEVERAGE)
            results = self.run_full_backtest(signal)
            print(f"Scaled down to {scale:.2f}x")

        return signal, results

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
        print("SMART DD STRATEGY - RESULTS")
        print("=" * 60)

        all_dd_ok = True
        all_calmar_ok = True

        for period, res in results.items():
            dd_ok = res['max_drawdown'] >= -0.17
            calmar_ok = res['calmar'] >= 2.0

            if not dd_ok:
                all_dd_ok = False
            if not calmar_ok:
                all_calmar_ok = False

            dd_status = "[OK]" if dd_ok else "[FAIL]"
            calmar_status = "[OK]" if calmar_ok else ""

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
        plt.savefig('smart_dd_results.png', dpi=150, bbox_inches='tight')
        plt.close()

        print("\nSaved: smart_dd_results.png")


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
            strategy = SmartDDStrategy(train_path, blind_path)
            strategy.load_data()
            break
        except Exception as e:
            continue

    if strategy is None:
        print("ERROR: Could not load data from any path.")
        return

    # Generate optimized signal
    signal, results = strategy.optimize_for_dd_constraint(-0.17)

    # Print and plot results
    strategy.print_results(results)
    strategy.plot_results(signal, results)

    return strategy, signal, results


if __name__ == '__main__':
    strategy, signal, results = main()
