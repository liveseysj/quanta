# QQQ Alpha Strategy

## Performance

| Period | Sharpe | Return |
|--------|--------|--------|
| Validation (2016-2021) | 2.01 | 471% |
| Test (2022-2025) | 1.76 | 225% |

## Quick Start

```bash
pip install pandas numpy yfinance matplotlib
python qqq_alpha_strategy.py
```

**Output:**
- Console: Sharpe ratios and returns
- `return_curve.png`: Strategy vs Buy & Hold chart

*Note: Signal generation takes ~1 minute due to Bayesian blending loop.*

## Philosophy

**Core Idea**: Optimize signals for each volatility regime separately, then combine using Bayesian weighting.

### Why Regime-Based?

Markets behave differently in different volatility environments:
- **Low Vol**: Trends persist, leverage up, ride momentum
- **Normal**: Mixed signals, balanced approach
- **High Vol**: Reversals dominate, reduce exposure
- **Panic**: Extreme fear, contrarian mean reversion

### Signal Flow

```
1. Calculate Parkinson Volatility (21-day)
         |
2. Build 5 Specialized Strategy Signals
   - Low Vol: HV Smooth Diverse (trend + momentum)
   - Normal: Gamma Momentum + Mean Revert blend
   - High Vol: Conservative MR (low risk entries)
   - Panic: Panic Fade (distress + recovery)
         |
3. Bayesian Regime Weighting
   - Gaussian prior from vol percentile
   - Likelihood from momentum, RSI, vol direction
   - Posterior = prior x likelihood
         |
4. Apply Regime Scaling
   - Low Vol:  2.06x
   - Normal:   1.76x
   - High Vol: 0.85x
   - Panic:    1.15x
         |
5. Add Enhancements
   - VIX calm/spike detection
   - RSI mean reversion boost
   - Momentum persistence filter
   - Day-of-week effects
         |
6. Clip to [-1.0, 1.5] -> Final Signal
```

## Key Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| LV Scale | 2.06 | Leverage in calm markets |
| MR Boost | 0.94 | Buy oversold (RSI<30) |
| VIX Calm | 0.47 | Boost when VIX<14 |
| Mon Boost | 0.30 | Buy Monday weakness |
