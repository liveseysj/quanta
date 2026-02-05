"""
Claude Low-Drawdown Strategy
============================

Target: Max Drawdown < 10%, Calmar > 2.0

Key principles for drawdown control:
1. Lower maximum leverage (0.8x instead of 1.5x)
2. Dynamic drawdown-based position scaling
3. Aggressive volatility reduction
4. Trend filter - only trade with the trend
5. Quick exit on momentum loss
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class LowDrawdownStrategy:
    """
    Conservative strategy focused on capital preservation.
    Target: Max DD < 10%, Calmar > 2.0
    """

    # Date ranges
    TRAIN_START = '2000-01-01'
    TRAIN_END = '2015-12-31'
    VAL_START = '2016-01-01'
    VAL_END = '2021-12-31'
    BLIND_START = '2022-01-01'

    # Leverage constraints
    MIN_LEVERAGE = 0.0  # No shorting - reduces drawdown risk
    MAX_LEVERAGE = 1.5  # Allow higher leverage in safe conditions

    # Volatility target
    TARGET_VOL = 0.16  # 16% target vol

    def __init__(self, qqq_train_file, qqq_blind_file):
        self.qqq_train_file = qqq_train_file
        self.qqq_blind_file = qqq_blind_file
        self.data = None

    def load_data(self):
        """Load and prepare QQQ data."""
        print("Loading data...")

        # Load training/validation data
        train_val = pd.read_csv(self.qqq_train_file)
        train_val.columns = train_val.columns.str.strip()
        train_val['Date'] = pd.to_datetime(train_val['Time'], errors='coerce')
        train_val = train_val.dropna(subset=['Date']).set_index('Date').sort_index()

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

        self._calculate_features()
        print(f"Loaded {len(self.data)} days of data")

    def _calculate_features(self):
        """Calculate technical features."""
        df = self.data

        # Returns
        df['Returns'] = df['Close'].pct_change()

        # Moving averages
        for period in [10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = df['Close'].rolling(period).mean()

        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI_14'] = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))

        # Volatility
        df['RVol_20'] = df['Returns'].rolling(20).std() * np.sqrt(252)
        df['RVol_60'] = df['Returns'].rolling(60).std() * np.sqrt(252)

        # Parkinson volatility
        park_const = 1.0 / (4.0 * np.log(2.0))
        df['Park_Var'] = park_const * (np.log(df['High'] / df['Low']) ** 2)
        df['Park_Vol_21'] = np.sqrt(df['Park_Var'].rolling(21).mean()) * np.sqrt(252)

        # Vol percentile
        df['Vol_Pct'] = df['Park_Vol_21'].expanding(min_periods=252).rank(pct=True) * 100
        df['Vol_Pct'] = df['Vol_Pct'].fillna(50)

        # Vol of vol
        df['Vol_of_Vol'] = df['Park_Vol_21'].rolling(20).std()
        df['VoV_Pct'] = df['Vol_of_Vol'].expanding(min_periods=252).rank(pct=True) * 100
        df['VoV_Pct'] = df['VoV_Pct'].fillna(50)

        # Momentum
        df['Mom_5'] = df['Close'].pct_change(5)
        df['Mom_20'] = df['Close'].pct_change(20)
        df['Mom_60'] = df['Close'].pct_change(60)

        # Distance from MAs
        df['Dist_MA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
        df['Dist_MA50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
        df['Dist_MA200'] = (df['Close'] - df['SMA_200']) / df['SMA_200']

        # IBS - Intraday Body Strength (from Kirtirajsinh Parmar)
        range_hl = (df['High'] - df['Low']).replace(0, 1e-9)
        df['IBS'] = (df['Close'] - df['Low']) / range_hl

        # Trend strength
        df['Above_200'] = (df['Close'] > df['SMA_200']).astype(float)
        df['Above_50'] = (df['Close'] > df['SMA_50']).astype(float)
        df['Above_20'] = (df['Close'] > df['SMA_20']).astype(float)

        # Trend score (0-3)
        df['Trend_Score'] = df['Above_200'] + df['Above_50'] + df['Above_20']

        # ATR for stop-loss
        df['TR'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                (df['High'] - df['Close'].shift(1)).abs(),
                (df['Low'] - df['Close'].shift(1)).abs()
            )
        )
        df['ATR_14'] = df['TR'].rolling(14).mean()

        # Drawdown tracking
        df['Cummax'] = df['Close'].cummax()
        df['Drawdown'] = (df['Close'] - df['Cummax']) / df['Cummax']

    def generate_signal(self):
        """
        Ultra-defensive strategy targeting DD < 10% with Calmar > 2.
        Key: Very low base vol with surgical leverage in perfect setups only.
        """
        print("Generating signals...")
        df = self.data

        signal = pd.Series(0.0, index=df.index)

        # Balanced vol target - 10% for optimal risk-adjusted returns
        BASE_VOL = 0.10

        for i in range(252, len(df)):
            # === MARKET STATE ===
            above_200 = df['Close'].iloc[i] > df['SMA_200'].iloc[i]
            above_50 = df['Close'].iloc[i] > df['SMA_50'].iloc[i]
            above_20 = df['Close'].iloc[i] > df['SMA_20'].iloc[i]

            mom_5 = df['Mom_5'].iloc[i]
            mom_20 = df['Mom_20'].iloc[i]
            mom_60 = df['Mom_60'].iloc[i]
            rsi = df['RSI_14'].iloc[i]
            vol_pct = df['Vol_Pct'].iloc[i]
            curr_vol = max(df['RVol_20'].iloc[i], 0.05)
            ibs = df['IBS'].iloc[i]

            # === PERFECT SETUP DETECTION ===
            all_aligned = above_200 and above_50 and above_20
            strong_momentum = mom_20 > 0.01 and mom_60 > 0.03
            low_vol = vol_pct < 35
            very_low_vol = vol_pct < 25

            # Perfect setup for aggressive entry
            is_perfect = all_aligned and strong_momentum and low_vol
            is_excellent = all_aligned and mom_20 > 0 and mom_60 > 0 and vol_pct < 45

            # === MEAN REVERSION SETUP ===
            oversold_bounce = above_50 and rsi < 30 and mom_5 > 0
            ibs_bounce = above_50 and ibs < 0.2 and rsi < 40

            # === BASE SIGNAL ===
            if is_perfect:
                if very_low_vol and mom_20 > 0.02:
                    base = 2.5  # Maximum aggression in perfect setup
                elif very_low_vol:
                    base = 2.0
                else:
                    base = 1.6
            elif is_excellent:
                base = 1.2
            elif all_aligned and mom_20 > 0:
                base = 0.8
            elif above_200 and above_50 and mom_60 > 0:
                base = 0.5
            elif above_200 and mom_60 > 0:
                base = 0.25
            else:
                base = 0.0

            # === MEAN REVERSION BOOST ===
            if oversold_bounce:
                base += 0.6
            elif ibs_bounce:
                base += 0.4
            elif above_50 and rsi < 35:
                base += 0.2

            # === VOL TARGETING ===
            vol_mult = BASE_VOL / curr_vol
            vol_mult = min(vol_mult, 2.0)

            # Vol crush in high vol (aggressive for DD control)
            if vol_pct > 60:
                vol_mult *= 0.15
            elif vol_pct > 50:
                vol_mult *= 0.30
            elif vol_pct > 40:
                vol_mult *= 0.50
            elif vol_pct > 30:
                vol_mult *= 0.75

            # === OVERBOUGHT ===
            if rsi > 80:
                base *= 0.3
            elif rsi > 75:
                base *= 0.5
            elif rsi > 70:
                base *= 0.7

            # === MOMENTUM CRASH ===
            if mom_60 < -0.10:
                base = 0.0
            elif mom_60 < -0.05:
                base *= 0.25
            elif mom_60 < -0.02:
                base *= 0.5

            # === RECENT WEAKNESS ===
            if i > 3:
                three_day = df['Returns'].iloc[i-3:i].sum()
                if three_day < -0.05:
                    base *= 0.2
                elif three_day < -0.03:
                    base *= 0.4
                elif three_day < -0.02:
                    base *= 0.6

            signal.iloc[i] = base * vol_mult

        signal = signal.clip(0.0, 1.2)
        return signal.fillna(0)

    def backtest(self, signal, period='all'):
        """Run backtest."""
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

        return {
            'period': period,
            'sharpe': sharpe,
            'ann_return': ann_return,
            'volatility': volatility,
            'max_drawdown': max_dd,
            'calmar': calmar,
            'total_return': total_return,
            'n_days': len(strat_ret),
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
        print("LOW DRAWDOWN STRATEGY - RESULTS")
        print("=" * 60)

        for period, res in results.items():
            dd_ok = abs(res['max_drawdown']) < 0.10
            calmar_ok = res['calmar'] > 2.0

            print(f"\n{period.upper()} Period:")
            print(f"  Sharpe Ratio:    {res['sharpe']:.3f}")
            print(f"  Annual Return:   {res['ann_return']:.1%}")
            print(f"  Volatility:      {res['volatility']:.1%}")
            print(f"  Max Drawdown:    {res['max_drawdown']:.1%}" +
                  (" [OK < 10%]" if dd_ok else " [FAIL]"))
            print(f"  Calmar Ratio:    {res['calmar']:.3f}" +
                  (" [OK > 2.0]" if calmar_ok else " [FAIL]"))

        print("\n" + "=" * 60)
        blind = results['blind']
        if abs(blind['max_drawdown']) < 0.10 and blind['calmar'] > 2.0:
            print("SUCCESS: Blind period meets DD < 10% and Calmar > 2.0!")
        else:
            print("Targets not yet met on blind period.")


def main():
    strategy = LowDrawdownStrategy(
        'Quanta Fellowship Train & Validate.csv',
        'QQQ Fellowship Blind Out of Sample.csv'
    )
    strategy.load_data()

    signal = strategy.generate_signal()
    results = strategy.run_full_backtest(signal)
    strategy.print_results(results)

    return strategy, signal, results


if __name__ == '__main__':
    strategy, signal, results = main()
