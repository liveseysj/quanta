"""
Generate exposure and equity curve plots for the Claude Hybrid Strategy.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from hybrid_strategy import HybridStrategy

def main():
    # Initialize and load data
    strategy = HybridStrategy(
        'Quanta Fellowship Train & Validate.csv',
        'QQQ Fellowship Blind Out of Sample.csv'
    )
    strategy.load_data()

    # Generate signal
    signal = strategy.generate_signal()

    # Get full backtest results
    df = strategy.data
    returns = df['Returns']
    strat_ret = (signal.shift(1) * returns).fillna(0)
    cum_ret = (1 + strat_ret).cumprod()

    # Buy and hold for comparison
    bh_ret = returns.fillna(0)
    bh_cum = (1 + bh_ret).cumprod()

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(16, 14),
                             gridspec_kw={'height_ratios': [2, 1, 1]})

    # Define period colors
    train_color = '#E8F5E9'  # Light green
    val_color = '#E3F2FD'    # Light blue
    blind_color = '#FFF3E0'  # Light orange

    # Period boundaries
    val_start = pd.Timestamp('2016-01-01')
    blind_start = pd.Timestamp('2022-01-01')

    # =========================================================================
    # PLOT 1: Equity Curve
    # =========================================================================
    ax1 = axes[0]

    # Add period backgrounds
    ax1.axvspan(df.index[0], val_start, alpha=0.3, color=train_color, label='Train (2000-2015)')
    ax1.axvspan(val_start, blind_start, alpha=0.3, color=val_color, label='Validation (2016-2021)')
    ax1.axvspan(blind_start, df.index[-1], alpha=0.3, color=blind_color, label='Blind (2022-2025)')

    # Plot equity curves
    ax1.plot(cum_ret.index, cum_ret.values, 'b-', linewidth=1.5, label='Claude Hybrid Strategy')
    ax1.plot(bh_cum.index, bh_cum.values, 'gray', linewidth=1, alpha=0.7, label='Buy & Hold QQQ')

    # Add vertical lines at period boundaries
    ax1.axvline(val_start, color='green', linestyle='--', alpha=0.7, linewidth=1)
    ax1.axvline(blind_start, color='red', linestyle='--', alpha=0.7, linewidth=1)

    ax1.set_ylabel('Cumulative Return (log scale)', fontsize=12)
    ax1.set_title('Claude Hybrid Strategy - Equity Curve', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(df.index[0], df.index[-1])

    # Add annotations for Sharpe ratios
    train_mask = df.index < val_start
    val_mask = (df.index >= val_start) & (df.index < blind_start)
    blind_mask = df.index >= blind_start

    def calc_sharpe(ret):
        if len(ret) < 20 or ret.std() < 1e-8:
            return 0
        return ret.mean() / ret.std() * np.sqrt(252)

    train_sharpe = calc_sharpe(strat_ret[train_mask])
    val_sharpe = calc_sharpe(strat_ret[val_mask])
    blind_sharpe = calc_sharpe(strat_ret[blind_mask])

    # Add text annotations
    ax1.text(0.02, 0.95, f'Train Sharpe: {train_sharpe:.2f}', transform=ax1.transAxes,
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax1.text(0.35, 0.95, f'Validation Sharpe: {val_sharpe:.2f}', transform=ax1.transAxes,
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax1.text(0.68, 0.95, f'Blind Sharpe: {blind_sharpe:.2f}', transform=ax1.transAxes,
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # =========================================================================
    # PLOT 2: Exposure / Leverage Over Time
    # =========================================================================
    ax2 = axes[1]

    # Add period backgrounds
    ax2.axvspan(df.index[0], val_start, alpha=0.3, color=train_color)
    ax2.axvspan(val_start, blind_start, alpha=0.3, color=val_color)
    ax2.axvspan(blind_start, df.index[-1], alpha=0.3, color=blind_color)

    # Plot exposure
    ax2.fill_between(signal.index, 0, signal.values, where=signal >= 0,
                     color='green', alpha=0.5, label='Long Exposure')
    ax2.fill_between(signal.index, 0, signal.values, where=signal < 0,
                     color='red', alpha=0.5, label='Short Exposure')

    # Add rolling average of exposure
    rolling_exp = signal.rolling(20).mean()
    ax2.plot(rolling_exp.index, rolling_exp.values, 'k-', linewidth=1.5,
             label='20-day Avg Exposure', alpha=0.8)

    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.axhline(1.5, color='gray', linestyle='--', alpha=0.5, label='Max Leverage (1.5x)')
    ax2.axhline(-1.0, color='gray', linestyle='--', alpha=0.5, label='Min Leverage (-1.0x)')
    ax2.axvline(val_start, color='green', linestyle='--', alpha=0.7, linewidth=1)
    ax2.axvline(blind_start, color='red', linestyle='--', alpha=0.7, linewidth=1)

    ax2.set_ylabel('Exposure / Leverage', fontsize=12)
    ax2.set_title('Strategy Exposure Over Time', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(df.index[0], df.index[-1])
    ax2.set_ylim(-1.5, 2.0)

    # Add exposure statistics
    train_exp = signal[train_mask].mean()
    val_exp = signal[val_mask].mean()
    blind_exp = signal[blind_mask].mean()
    ax2.text(0.02, 0.92, f'Avg Exp: {train_exp:.2f}', transform=ax2.transAxes, fontsize=9)
    ax2.text(0.35, 0.92, f'Avg Exp: {val_exp:.2f}', transform=ax2.transAxes, fontsize=9)
    ax2.text(0.68, 0.92, f'Avg Exp: {blind_exp:.2f}', transform=ax2.transAxes, fontsize=9)

    # =========================================================================
    # PLOT 3: Drawdown
    # =========================================================================
    ax3 = axes[2]

    # Add period backgrounds
    ax3.axvspan(df.index[0], val_start, alpha=0.3, color=train_color)
    ax3.axvspan(val_start, blind_start, alpha=0.3, color=val_color)
    ax3.axvspan(blind_start, df.index[-1], alpha=0.3, color=blind_color)

    # Calculate drawdown
    drawdown = cum_ret / cum_ret.cummax() - 1
    bh_drawdown = bh_cum / bh_cum.cummax() - 1

    # Plot drawdowns
    ax3.fill_between(drawdown.index, 0, drawdown.values * 100,
                     color='red', alpha=0.5, label='Strategy Drawdown')
    ax3.plot(bh_drawdown.index, bh_drawdown.values * 100,
             'gray', linewidth=1, alpha=0.7, label='Buy & Hold Drawdown')

    ax3.axvline(val_start, color='green', linestyle='--', alpha=0.7, linewidth=1)
    ax3.axvline(blind_start, color='red', linestyle='--', alpha=0.7, linewidth=1)

    ax3.set_ylabel('Drawdown (%)', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_title('Drawdown Analysis', fontsize=14, fontweight='bold')
    ax3.legend(loc='lower left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(df.index[0], df.index[-1])

    # Add max drawdown annotations
    train_dd = drawdown[train_mask].min() * 100
    val_dd = drawdown[val_mask].min() * 100
    blind_dd = drawdown[blind_mask].min() * 100
    ax3.text(0.02, 0.08, f'Max DD: {train_dd:.1f}%', transform=ax3.transAxes, fontsize=9)
    ax3.text(0.35, 0.08, f'Max DD: {val_dd:.1f}%', transform=ax3.transAxes, fontsize=9)
    ax3.text(0.68, 0.08, f'Max DD: {blind_dd:.1f}%', transform=ax3.transAxes, fontsize=9)

    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.savefig('exposure_equity_curve.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Saved: exposure_equity_curve.png")

    # =========================================================================
    # Print summary statistics
    # =========================================================================
    print("\n" + "="*60)
    print("EXPOSURE STATISTICS")
    print("="*60)

    for name, mask in [('Train', train_mask), ('Validation', val_mask), ('Blind', blind_mask)]:
        exp = signal[mask]
        print(f"\n{name} Period:")
        print(f"  Mean Exposure:     {exp.mean():.3f}")
        print(f"  Std Exposure:      {exp.std():.3f}")
        print(f"  Min Exposure:      {exp.min():.3f}")
        print(f"  Max Exposure:      {exp.max():.3f}")
        print(f"  % Long (>0):       {(exp > 0).mean()*100:.1f}%")
        print(f"  % Short (<0):      {(exp < 0).mean()*100:.1f}%")
        print(f"  % Max Long (>1):   {(exp > 1).mean()*100:.1f}%")


if __name__ == '__main__':
    main()
