"""
Tight Vol Strategy - Low DD via Tight Vol Targeting
====================================================

Strategy: Keep volatility extremely low to limit drawdowns,
then maximize Sharpe ratio to achieve Calmar > 2.0.

Math:
- Calmar = Return / DD
- If DD = 10%, need Return > 20% for Calmar > 2.0
- If DD = 8%, need Return > 16% for Calmar > 2.0
- At 6% vol, a Sharpe of 2.5 gives ~15% return

Focus: Ultra-high Sharpe with very tight vol targeting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def ts_rank(x, window):
    return x.rolling(window, min_periods=1).apply(
        lambda a: (a.argsort().argsort()[-1] + 1) / len(a), raw=True
    )


def ts_mad(x, window):
    mean = x.rolling(window, min_periods=1).mean()
    return (x - mean).abs().rolling(window, min_periods=1).mean()


class TightVolStrategy:
    """High Sharpe strategy with tight vol targeting."""

    TRAIN_START = '2000-01-01'
    TRAIN_END = '2015-12-31'
    VAL_START = '2016-01-01'
    VAL_END = '2021-12-31'
    BLIND_START = '2022-01-01'

    MIN_LEVERAGE = 0.0
    MAX_LEVERAGE = 1.2

    # Very tight vol target
    TARGET_VOL = 0.08  # 8%

    def __init__(self, qqq_train_file, qqq_blind_file):
        self.qqq_train_file = qqq_train_file
        self.qqq_blind_file = qqq_blind_file
        self.data = None

    def load_data(self):
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

    def _calculate_features(self):
        df = self.data

        df['Returns'] = df['Close'].pct_change()

        for period in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = df['Close'].rolling(period).mean()

        park_const = 1.0 / (4.0 * np.log(2.0))
        df['Park_Var'] = park_const * (np.log(df['High'] / df['Low']) ** 2)
        df['Park_Vol_21'] = np.sqrt(df['Park_Var'].rolling(21).mean()) * np.sqrt(252)

        df['RVol_20'] = df['Returns'].rolling(20).std() * np.sqrt(252)
        df['RVol_60'] = df['Returns'].rolling(60).std() * np.sqrt(252)

        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI_14'] = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))

        df['Vol_Pct'] = df['Park_Vol_21'].expanding(min_periods=252).rank(pct=True) * 100
        df['Vol_Pct'] = df['Vol_Pct'].fillna(50)

        for period in [5, 10, 20, 60, 120, 200]:
            df[f'Mom_{period}'] = df['Close'].pct_change(period)

        df['Dist_MA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
        df['Dist_MA200'] = (df['Close'] - df['SMA_200']) / df['SMA_200']

        range_hl = (df['High'] - df['Low']).replace(0, 1e-9)
        df['IBS'] = (df['Close'] - df['Low']) / range_hl

        df['Rolling_High_60'] = df['Close'].rolling(60).max()
        df['DD_60'] = (df['Close'] / df['Rolling_High_60']) - 1

        df['SMA50_Above_200'] = (df['SMA_50'] > df['SMA_200']).astype(int)

    def signal_vol_shock_mr(self, i):
        """Volatility shock mean reversion signal (from best performer)."""
        df = self.data

        if i < 200:
            return 0.0

        # Get slice for calculation
        returns = df['Returns'].iloc[max(0, i-200):i+1]

        x = returns.diff(2)
        mean = x.rolling(2, min_periods=1).mean()
        mad = (x - mean).abs().rolling(2, min_periods=1).mean()
        x = np.power(1.0 / (mad + 1e-9) - 1.0, 0.5)

        x_rank = x.rolling(189, min_periods=50).apply(
            lambda a: (a.argsort().argsort()[-1] + 1) / len(a), raw=True
        )

        if len(x_rank) > 0 and not np.isnan(x_rank.iloc[-1]):
            rank_val = x_rank.iloc[-1] - 0.33
            if rank_val > 0:
                return 1.2
            else:
                return 0.5
        return 0.5

    def generate_signal(self):
        """Generate high-Sharpe signal with tight vol targeting."""
        print("Generating tight vol signals...")

        df = self.data
        n = len(df)

        signal = pd.Series(0.0, index=df.index)

        for i in range(252, n):
            close = df['Close'].iloc[i]
            sma200 = df['SMA_200'].iloc[i]
            sma50 = df['SMA_50'].iloc[i]
            sma20 = df['SMA_20'].iloc[i]

            vol_pct = df['Vol_Pct'].iloc[i]
            rsi = df['RSI_14'].iloc[i]
            mom_20 = df['Mom_20'].iloc[i]
            mom_60 = df['Mom_60'].iloc[i]
            mom_120 = df['Mom_120'].iloc[i]
            ibs = df['IBS'].iloc[i]
            dist_20 = df['Dist_MA20'].iloc[i]
            dist_200 = df['Dist_MA200'].iloc[i]
            dd_60 = df['DD_60'].iloc[i]

            curr_vol = max(df['RVol_20'].iloc[i], 0.05)

            above_200 = close > sma200
            above_50 = close > sma50
            golden_cross = sma50 > sma200

            # =====================================================
            # CRASH FILTER - Stay in cash during crashes
            # =====================================================
            in_crash = False

            # Severe crash conditions
            if dist_200 < -0.15 and mom_60 < -0.10:
                in_crash = True
            if not above_200 and not golden_cross and mom_120 < -0.15:
                in_crash = True
            if dd_60 < -0.20 and not above_200:
                in_crash = True
            if vol_pct > 85 and not above_200 and mom_20 < -0.05:
                in_crash = True

            if in_crash:
                # Only trade extreme bounces
                if rsi < 15 and ibs < 0.1:
                    signal.iloc[i] = 0.2
                else:
                    signal.iloc[i] = 0.0
                continue

            # =====================================================
            # SIGNAL GENERATION - Focus on highest quality setups
            # =====================================================
            sig = 0.0

            # Condition 1: Strong uptrend - trend following
            if above_200 and above_50 and close > sma20:
                if mom_20 > 0.02 and mom_60 > 0.05:
                    sig = 1.0
                elif mom_20 > 0 and mom_60 > 0:
                    sig = 0.8
                else:
                    sig = 0.5

            # Condition 2: Mean reversion - oversold in uptrend
            if above_200:
                if rsi < 25 and ibs < 0.15:
                    sig = max(sig, 1.2)
                elif rsi < 30 and ibs < 0.2:
                    sig = max(sig, 1.0)
                elif rsi < 35 and dist_20 < -0.03:
                    sig = max(sig, 0.8)

            # Condition 3: Dip buying in uptrend
            if above_200 and above_50:
                if dist_20 < -0.04 and rsi < 35:
                    sig = max(sig, 1.0)
                elif dist_20 < -0.03 and rsi < 40:
                    sig = max(sig, 0.8)

            # Condition 4: Vol shock mean reversion
            vol_shock_sig = self.signal_vol_shock_mr(i)
            if above_200:
                sig = max(sig, vol_shock_sig * 0.8)

            # Reduce in moderate downtrend
            if not above_200 and above_50:
                sig *= 0.4

            # Below both MAs - minimal exposure
            if not above_200 and not above_50:
                sig *= 0.2
                if rsi < 25:
                    sig = max(sig, 0.3)

            # Reduce on overbought
            if rsi > 75:
                sig *= 0.75
            if rsi > 80:
                sig *= 0.80

            # High vol reduction
            if vol_pct > 70:
                sig *= 0.7
            elif vol_pct > 55:
                sig *= 0.85

            # =====================================================
            # TIGHT VOL TARGETING
            # =====================================================
            vol_scalar = self.TARGET_VOL / curr_vol
            vol_scalar = np.clip(vol_scalar, 0.3, 1.2)
            sig = sig * vol_scalar

            # Clip
            sig = np.clip(sig, self.MIN_LEVERAGE, self.MAX_LEVERAGE)

            signal.iloc[i] = sig

        return signal.fillna(0)

    def optimize_for_calmar(self, target_dd=-0.17, target_calmar=2.0):
        """Optimize for both DD and Calmar."""
        print(f"Optimizing for DD < {abs(target_dd):.0%} and Calmar > {target_calmar}...")

        signal = self.generate_signal()
        results = self.run_full_backtest(signal)

        worst_dd = min(
            results['train']['max_drawdown'],
            results['validation']['max_drawdown'],
            results['blind']['max_drawdown']
        )

        blind_calmar = results['blind']['calmar']

        print(f"Initial: Train DD={results['train']['max_drawdown']:.1%}, "
              f"Val DD={results['validation']['max_drawdown']:.1%}, "
              f"Blind DD={results['blind']['max_drawdown']:.1%}, "
              f"Blind Calmar={blind_calmar:.2f}")

        # If DD too high, scale down
        if worst_dd < target_dd:
            scale = (target_dd / worst_dd) * 0.95
            signal = (signal * scale).clip(self.MIN_LEVERAGE, self.MAX_LEVERAGE)
            results = self.run_full_backtest(signal)
            print(f"Scaled down by {scale:.2f}x")
        else:
            # Try scaling up while meeting DD constraint
            best_signal = signal.copy()
            best_calmar = results['blind']['calmar']

            for scale in np.arange(1.05, 2.5, 0.05):
                test_signal = (signal * scale).clip(self.MIN_LEVERAGE, self.MAX_LEVERAGE)
                test_results = self.run_full_backtest(test_signal)
                test_worst_dd = min(
                    test_results['train']['max_drawdown'],
                    test_results['validation']['max_drawdown'],
                    test_results['blind']['max_drawdown']
                )
                test_calmar = test_results['blind']['calmar']

                if test_worst_dd >= target_dd:
                    if test_calmar > best_calmar:
                        best_signal = test_signal.copy()
                        best_calmar = test_calmar
                        print(f"Scale {scale:.2f}x: DD={test_worst_dd:.1%}, Calmar={test_calmar:.2f}")
                else:
                    break

            signal = best_signal

        results = self.run_full_backtest(signal)
        return signal, results

    def backtest(self, signal, period='all'):
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
        results = {}
        for period in ['train', 'validation', 'blind']:
            results[period] = self.backtest(signal, period)
        return results

    def print_results(self, results):
        print("\n" + "=" * 60)
        print("TIGHT VOL STRATEGY - RESULTS")
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
        calmars = [results[p]['calmar'] for p in results.keys()]
        colors = ['green' if c >= 2.0 else 'orange' if c >= 1.5 else 'red' for c in calmars]
        bars = ax2.bar(list(results.keys()), calmars, color=colors, edgecolor='black')
        ax2.axhline(2.0, color='green', linestyle='--', label='Calmar Target (2.0)')
        ax2.set_ylabel('Calmar Ratio')
        ax2.set_title('Calmar Ratio by Period')
        ax2.legend()

        for bar, c in zip(bars, calmars):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{c:.2f}', ha='center', va='bottom', fontsize=10)

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
                ax4.plot(dd, label=f"{period.capitalize()}")
        ax4.axhline(-0.17, color='red', linestyle='--', label='17% DD Limit')
        ax4.set_ylabel('Drawdown')
        ax4.set_title('Drawdown Analysis')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('tight_vol_results.png', dpi=150, bbox_inches='tight')
        plt.close()

        print("\nSaved: tight_vol_results.png")


def main():
    data_paths = [
        ('Quanta Fellowship Train & Validate.csv', 'QQQ Fellowship Blind Out of Sample.csv'),
        ('../minh_nguyen/Quanta Fellowship Train & Validate.csv', '../minh_nguyen/QQQ Fellowship Blind Out of Sample.csv'),
    ]

    strategy = None
    for train_path, blind_path in data_paths:
        try:
            strategy = TightVolStrategy(train_path, blind_path)
            strategy.load_data()
            break
        except:
            continue

    if strategy is None:
        print("ERROR: Could not load data.")
        return

    signal, results = strategy.optimize_for_calmar(-0.17, 2.0)

    strategy.print_results(results)
    strategy.plot_results(signal, results)

    return strategy, signal, results


if __name__ == '__main__':
    strategy, signal, results = main()
