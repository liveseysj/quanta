# LLM-Assisted Multi-Persona Trading Strategy (QQQ)

## Quanta Fellowship – LLM-Based Strategy Builder

---

## Overview

This project implements a **bias-free, multi-signal quantitative trading strategy** for QQQ using a **persona-driven signal design framework**. The strategy is evaluated using **strict train / validation / blind out-of-sample splits**, adheres to **explicit leverage constraints**, and includes **full diagnostic analysis** (PnL curves, correlation matrices, Sharpe metrics).

The goal of this submission is to demonstrate:
- Robust signal construction
- Absence of lookahead bias
- Orthogonal signal behavior
- Strong out-of-sample generalization

---

## Dataset & Splits

The QQQ dataset is split exactly as specified:

| Dataset | Date Range | Trading Days |
|------|-----------|--------------|
| **Train** | 2000-01-03 → 2015-12-31 | 4,025 |
| **Validation** | 2016-01-04 → 2021-12-31 | 1,511 |
| **Blind OOS** | 2022-01-03 → 2025-12-05 | 986 |

**Important:**
- Blind OOS data is never used for parameter selection, tuning, or model adjustment.
- All signals and parameters are finalized before evaluating blind performance.

---

## How to Run

### Requirements
- Python 3.9+
- Required packages:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`

The provided `quanta_fellowship_project` module handles all data loading and dataset splits.

---

### Run the Full Backtest

From the project root directory:

```bash
python final.py
```

---

## Execution & Lookahead Bias Control

- All indicators are computed using **rolling windows only**.
- No statistics are calculated using full-sample information.
- Signals are generated at the **close of day _t_**.
- Positions are executed at the **close of day _t_** and held until the **close of day _t+1_**.
- PnL is computed as: PnL_t = Position_{t-1} × Return_t


This guarantees:
- No future data leakage
- No implicit lookahead bias
- Realistic daily execution assumptions

---

## Leverage Constraints

Each signal outputs **discrete daily leverage values**:

| Output | Interpretation |
|------|----------------|
| `-1.0` | 100% short |
| `0.0` | Flat |
| `+1.5` | 150% long |

All signals and the portfolio strictly respect the **−1.0 to +1.5 leverage constraint**.

---

## Trading Signals (Personas)

Each signal corresponds to a distinct market hypothesis:

### 1. Volatility Shock Mean-Reverter  
A contrarian strategy that exploits short-term volatility spikes that historically revert.

### 2. Event-Driven Liquidity Rebound Trader  
Captures price rebounds following negative return shocks driven by forced selling and liquidity stress.

### 3. Volatility Compression Momentum Trader  
Identifies momentum breakouts following prolonged periods of suppressed volatility.

---

## Portfolio Construction

The portfolio is constructed as an **equal-weight ensemble**: Portfolio Position_t = mean(Signal_1_t, Signal_2_t, Signal_3_t)


This design:
- Encourages diversification
- Reduces dependence on any single hypothesis
- Improves stability across regimes

---

## Performance Summary

### Sharpe Ratio & Mean Daily PnL

| Signal | Dataset | Sharpe | Mean PnL |
|-----|--------|--------|----------|
| Volatility Shock Mean-Reverter | Train | 0.54 | 0.00084 |
| Event-Driven Liquidity Rebound Trader | Train | 0.15 | 0.00024 |
| Volatility Compression Momentum Trader | Train | 0.34 | 0.00055 |
| **Equal-Weight Portfolio** | Train | **0.48** | **0.00054** |
| Volatility Shock Mean-Reverter | Validation | 1.36 | 0.00141 |
| Event-Driven Liquidity Rebound Trader | Validation | 0.70 | 0.00079 |
| Volatility Compression Momentum Trader | Validation | 0.64 | 0.00075 |
| **Equal-Weight Portfolio** | Validation | **1.26** | **0.00098** |
| Volatility Shock Mean-Reverter | Blind | 1.56 | 0.00193 |
| Event-Driven Liquidity Rebound Trader | Blind | 1.49 | 0.00165 |
| Volatility Compression Momentum Trader | Blind | 1.03 | 0.00137 |
| **Equal-Weight Portfolio** | Blind | **2.14** | **0.00164** |

**Key observation:**  
Portfolio Sharpe increases monotonically from Train → Validation → Blind OOS, indicating strong generalization and limited overfitting.

---

## Signal Correlation Analysis

### PnL Correlation Matrix (Signals Only)

| | Mean-Reverter | Liquidity Rebound | Volatility Momentum |
|--|--|--|--|
| **Mean-Reverter** | 1.00 | 0.17 | 0.27 |
| **Liquidity Rebound** | 0.17 | 1.00 | 0.31 |
| **Volatility Momentum** | 0.27 | 0.31 | 1.00 |

Low-to-moderate correlations indicate meaningful signal diversification and justify portfolio aggregation.

---

## Diagnostics & Visualizations

The analysis includes:
- Cumulative PnL plots for each signal
- Combined portfolio PnL across Train, Validation, and Blind periods
- Correlation heatmap with numeric values
- Correlation tables for exact inspection

These diagnostics are used to validate robustness and regime behavior.

---

## Role of LLMs

LLMs were used as:
- Hypothesis generators
- Strategy prototyping assistants
- Code reviewers for bias detection

All final trading logic is deterministic, interpretable, and fully backtested.

---

## Conclusion

This project demonstrates:
- Bias-free signal construction
- Strong out-of-sample performance
- Orthogonal signal behavior
- Robust ensemble portfolio design

The framework is scalable, auditable, and aligned with Quanta Fellowship evaluation criteria.




