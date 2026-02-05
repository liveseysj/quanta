"""
Quanta Fellowship LLM-Based Strategy Builder
===========================================

A comprehensive framework for developing 50+ diverse trading signals for QQQ
using LLM-assisted hypothesis generation to achieve 2.0+ Sharpe ratio.

Key Requirements:
- Train: 2000-2015
- Validate: 2016-2021  
- Blind Holdout: 2022-2025 (never iterate on this)
- Leverage: -1.0 to +1.5x
- No lookahead bias
- Robustness testing (±10% parameter changes)
- 50+ diverse, orthogonal signals
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class QuantaFellowshipProject:
    """Main project class for the Quanta Fellowship challenge."""
    
    def __init__(self):
        self.train_data = None
        self.validation_data = None
        self.blind_data = None
        self.signals = {}
        self.portfolio_signals = {}
        self.performance_metrics = {}
        
        # Date ranges as specified in requirements
        self.TRAIN_START = '2000-01-01'
        self.TRAIN_END = '2015-12-31'
        self.VALIDATION_START = '2016-01-01'
        self.VALIDATION_END = '2021-12-31'
        self.BLIND_START = '2022-01-01'
        self.BLIND_END = '2025-06-30'
        
        # Leverage constraints
        self.MIN_LEVERAGE = -1.0  # 100% short
        self.MAX_LEVERAGE = 1.5   # 150% long
        
    def load_data(self):
        """Load and prepare the QQQ data with proper date parsing."""
        print("Loading QQQ data...")
        
        # Load training and validation data (2000-2021)
        train_val_data = pd.read_csv('Quanta Fellowship Train & Validate.csv')
        # Parse Time robustly: coerce unparsable strings to NaT and drop them
        train_val_data['Time'] = pd.to_datetime(train_val_data['Time'], errors='coerce')
        train_val_data = train_val_data.dropna(subset=['Time']).copy()
        train_val_data.set_index('Time', inplace=True)
        train_val_data.sort_index(inplace=True)
        
        # Load blind out-of-sample data (2022-2025)
        blind_data = pd.read_csv('QQQ Fellowship Blind Out of Sample.csv')
        # Parse Time robustly: coerce unparsable strings to NaT and drop them
        blind_data['Time'] = pd.to_datetime(blind_data['Time'], errors='coerce')
        blind_data = blind_data.dropna(subset=['Time']).copy()
        blind_data.set_index('Time', inplace=True)
        blind_data.sort_index(inplace=True)
        
        # Clean column names
        for df in [train_val_data, blind_data]:
            df.columns = df.columns.str.replace('%Change', 'PctChange')
            # Convert percentage strings to floats
            if 'PctChange' in df.columns:
                df['PctChange'] = df['PctChange'].str.rstrip('%').astype(float) / 100
        
        # Split train/validation data
        self.train_data = train_val_data.loc[self.TRAIN_START:self.TRAIN_END].copy()
        self.validation_data = train_val_data.loc[self.VALIDATION_START:self.VALIDATION_END].copy()
        # Filter blind data to start from BLIND_START (2022-01-01)
        self.blind_data = blind_data.loc[self.BLIND_START:].copy()
        
        # Rename 'Latest' to 'Close' for consistency
        for df in [self.train_data, self.validation_data, self.blind_data]:
            if 'Latest' in df.columns:
                df.rename(columns={'Latest': 'Close'}, inplace=True)
        
        # Add derived features
        for df in [self.train_data, self.validation_data, self.blind_data]:
            self._add_derived_features(df)
        
        print(f"Train data: {len(self.train_data)} days ({self.train_data.index[0].date()} to {self.train_data.index[-1].date()})")
        print(f"Validation data: {len(self.validation_data)} days ({self.validation_data.index[0].date()} to {self.validation_data.index[-1].date()})")
        print(f"Blind data: {len(self.blind_data)} days ({self.blind_data.index[0].date()} to {self.blind_data.index[-1].date()})")
        
    def _add_derived_features(self, df):
        """Add commonly used technical indicators and features."""
        # Returns
        df['Returns'] = df['Close'].pct_change()
        df['LogReturns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # True Range and ATR
        df['TrueRange'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            )
        )
        df['ATR_14'] = df['TrueRange'].rolling(14).mean()
        
        # Simple moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = df['Close'].rolling(period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # MACD
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Volume indicators
        df['Volume_SMA_20'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
        
        # Price position indicators
        df['HighLow_Ratio'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # Volatility measures
        df['Volatility_20'] = df['Returns'].rolling(20).std() * np.sqrt(252)
        df['Volatility_5'] = df['Returns'].rolling(5).std() * np.sqrt(252)
        
    def calculate_performance_metrics(self, returns, period_name=""):
        """Calculate comprehensive performance metrics for a return series."""
        if len(returns) == 0 or returns.isna().all():
            return {}
        
        # Remove NaN values
        returns_clean = returns.dropna()
        if len(returns_clean) == 0:
            return {}
        
        # Basic metrics
        total_return = (1 + returns_clean).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns_clean)) - 1
        volatility = returns_clean.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown calculation
        cumulative = (1 + returns_clean).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        win_rate = (returns_clean > 0).mean()
        
        # Sortino ratio (downside deviation)
        downside_returns = returns_clean[returns_clean < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
        
        return {
            f'{period_name}_Total_Return': total_return,
            f'{period_name}_Annualized_Return': annualized_return,
            f'{period_name}_Volatility': volatility,
            f'{period_name}_Sharpe_Ratio': sharpe_ratio,
            f'{period_name}_Max_Drawdown': max_drawdown,
            f'{period_name}_Calmar_Ratio': calmar_ratio,
            f'{period_name}_Win_Rate': win_rate,
            f'{period_name}_Sortino_Ratio': sortino_ratio,
            f'{period_name}_Days': len(returns_clean)
        }
    
    def backtest_signal(self, signal_series, data_period='train'):
        """Backtest a single signal with STRICT NO-LOOKAHEAD BIAS enforcement."""
        if data_period == 'train':
            data = self.train_data
        elif data_period == 'validation':
            data = self.validation_data
        elif data_period == 'blind':
            data = self.blind_data
        else:
            raise ValueError("data_period must be 'train', 'validation', or 'blind'")
        
        # Align signal with data (forward fill only, no future data)
        aligned_signal = signal_series.reindex(data.index, method='ffill').fillna(0)
        
        # Ensure signal is within leverage constraints
        aligned_signal = np.clip(aligned_signal, self.MIN_LEVERAGE, self.MAX_LEVERAGE)
        
        # CRITICAL: Use signal[t-1] to trade return[t] (NO LOOKAHEAD)
        # This ensures we can only use information available BEFORE the trading decision
        signal_shifted = aligned_signal.shift(1).dropna()  # Use previous day's signal
        
        # Align returns with shifted signal
        returns_aligned = data['Returns'].reindex(signal_shifted.index).dropna()
        
        # Calculate strategy returns: position[t-1] * return[t]
        strategy_returns = signal_shifted * returns_aligned
        
        # Remove any remaining NaN values
        strategy_returns = strategy_returns.dropna()
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(strategy_returns, data_period.capitalize())
        
        return metrics, strategy_returns
    
    def test_signal_robustness(self, signal_func, base_params, data):
        """Test signal robustness by varying parameters ±10%."""
        base_signal = signal_func(data, **base_params)
        base_metrics, _ = self.backtest_signal(base_signal, 'train')
        base_sharpe = base_metrics.get('Train_Sharpe_Ratio', 0)
        
        robustness_results = {'base_sharpe': base_sharpe, 'variations': []}
        
        for param_name, param_value in base_params.items():
            if isinstance(param_value, (int, float)):
                # Test ±10% variations
                for variation in [-0.1, 0.1]:
                    new_params = base_params.copy()
                    new_value = param_value * (1 + variation)
                    
                    # Ensure integer parameters remain integers
                    if isinstance(param_value, int):
                        new_value = int(round(new_value))
                        if new_value == param_value:  # Skip if no change
                            continue
                    
                    new_params[param_name] = new_value
                    
                    try:
                        varied_signal = signal_func(data, **new_params)
                        varied_metrics, _ = self.backtest_signal(varied_signal, 'train')
                        varied_sharpe = varied_metrics.get('Train_Sharpe_Ratio', 0)
                        
                        robustness_results['variations'].append({
                            'param': param_name,
                            'variation': variation,
                            'new_value': new_value,
                            'sharpe': varied_sharpe,
                            'sharpe_change': varied_sharpe - base_sharpe
                        })
                    except Exception as e:
                        print(f"Error testing {param_name}={new_value}: {e}")
        
        # Calculate robustness score (smaller changes = more robust)
        if robustness_results['variations']:
            sharpe_changes = [abs(v['sharpe_change']) for v in robustness_results['variations']]
            avg_sharpe_change = np.mean(sharpe_changes)
            max_sharpe_change = max(sharpe_changes)
            robustness_results['avg_sharpe_change'] = avg_sharpe_change
            robustness_results['max_sharpe_change'] = max_sharpe_change
            robustness_results['robustness_score'] = 1 / (1 + avg_sharpe_change)  # Higher is more robust
        
        return robustness_results

    def create_project_directory(self):
        """Create organized directory structure for the project."""
        import os
        
        directories = [
            'signals',
            'backtests', 
            'portfolios',
            'results',
            'documentation'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        print("Project directory structure created successfully!")


if __name__ == "__main__":
    # Initialize and test the framework
    project = QuantaFellowshipProject()
    project.create_project_directory()
    project.load_data()
    
    print("\n" + "="*60)
    print("QUANTA FELLOWSHIP PROJECT INITIALIZED")
    print("="*60)
    print(f"Framework ready for signal generation!")
    print(f"Target: 50+ signals with 2.0+ Sharpe ratio")
    print(f"Next steps: Implement persona-based signal generators")
