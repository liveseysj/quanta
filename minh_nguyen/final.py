import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from quanta_fellowship_project import QuantaFellowshipProject

# =========================================================
# ------------------- CORE OPERATORS ----------------------
# =========================================================

def ts_rank(x, window):
    return x.rolling(window, min_periods=1).apply(
        lambda a: (a.argsort().argsort()[-1] + 1) / len(a),
        raw=True
    )

def ts_argmin(x, window):
    return x.rolling(window, min_periods=1).apply(
        lambda a: np.argmin(a),
        raw=True
    )

def ts_mad(x, window):
    mean = x.rolling(window, min_periods=1).mean()
    return (x - mean).abs().rolling(window, min_periods=1).mean()

def ts_percentile(x, window, p):
    def _pct(arr):
        arr = np.asarray(arr)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return np.nan
        arr.sort()
        idx = p * (len(arr) - 1)
        lo = int(np.floor(idx))
        hi = min(lo + 1, len(arr) - 1)
        w = idx - lo
        return arr[lo] * (1 - w) + arr[hi] * w

    return x.rolling(window, min_periods=1).apply(_pct, raw=False)

def volatility_fast(x, window):
    return x.rolling(window, min_periods=2).std(ddof=1)

def ts_mscore(x, window):
    abs_x = x.abs()
    s = abs_x.rolling(window, min_periods=1).sum()
    n = abs_x.rolling(window, min_periods=1).count()
    return abs_x * n / s

# =========================================================
# ---------------------- EVENT ----------------------------
# =========================================================

def event(signal, trigger, threshold, max_hold, bet_all):
    """
    Signal observed at close t → position applied from t+1 onward
    """
    out = pd.Series(index=signal.index, dtype=float)
    prev = 0.0 if bet_all else np.nan
    counter = max_hold

    for t in range(len(signal)):
        if np.isfinite(trigger.iloc[t]) and trigger.iloc[t] > threshold:
            prev = signal.iloc[t]
            counter = 0
        else:
            counter += 1
            if counter >= max_hold:
                prev = 0.0 if bet_all else np.nan

        out.iloc[t] = prev

    return out

# =========================================================
# ---------------- SIGNAL SCALING -------------------------
# =========================================================

def scale_signal(x):
    """
    Map {-1, 0, +1} → {-1, 0, +1.5}
    """
    return x.replace({1.0: 1.5, -1.0: -1.0, 0.0: 0.0})

# =========================================================
# ---------------------- SIGNALS --------------------------
# =========================================================

def signal_3(df):
    """Volatility Shock Mean-Reverter"""
    x = df['Returns'].diff(2)
    x = ts_mad(x, 2)
    x = np.power(1.0 / x - 1.0, 0.5)
    x = ts_rank(x, 189) - 0.33
    return scale_signal(np.sign(x))

def signal_6(df):
    """Event-Driven Liquidity Rebound Trader"""
    x = ts_percentile(df['Volume'], 126, 0.55)
    x = event(x, df['Returns'], threshold=-0.87, max_hold=2, bet_all=1)
    x = volatility_fast(x, 126)
    x = ts_argmin(x, 252)
    x = ts_rank(np.exp(x), 21) - 0.5
    return scale_signal(-np.sign(x))

def signal_7(df):
    """Volatility Compression Momentum Trader"""
    x = np.power(df['High'], 0.5)
    x = volatility_fast(x, 10)
    x = ts_mscore(1.0 / (x - 1.0), 4)
    x = ts_rank(x, 189) - 0.2
    return scale_signal(np.sign(x))

# =========================================================
# -------------------- BACKTEST ---------------------------
# =========================================================

def compute_pnl(position, returns):
    return position.shift(1) * returns

def sharpe_ratio(pnl):
    pnl = pnl.dropna()
    if len(pnl) == 0:
        return np.nan
    mu = pnl.mean()
    mu2 = (pnl ** 2).mean()
    return mu / np.sqrt(mu2 - mu ** 2) * np.sqrt(252)

def evaluate_all(data):
    s3 = signal_3(data)
    s6 = signal_6(data)
    s7 = signal_7(data)

    portfolio = pd.concat([s3, s6, s7], axis=1).mean(axis=1)

    rows = [
        {"Signal": "Volatility Shock Mean-Reverter",
         "Sharpe": sharpe_ratio(compute_pnl(s3, data["Returns"])),
         "MeanPnL": compute_pnl(s3, data["Returns"]).mean()},
        {"Signal": "Event-Driven Liquidity Rebound Trader",
         "Sharpe": sharpe_ratio(compute_pnl(s6, data["Returns"])),
         "MeanPnL": compute_pnl(s6, data["Returns"]).mean()},
        {"Signal": "Volatility Compression Momentum Trader",
         "Sharpe": sharpe_ratio(compute_pnl(s7, data["Returns"])),
         "MeanPnL": compute_pnl(s7, data["Returns"]).mean()},
        {"Signal": "Equal-Weight Multi-Persona Portfolio",
         "Sharpe": sharpe_ratio(compute_pnl(portfolio, data["Returns"])),
         "MeanPnL": compute_pnl(portfolio, data["Returns"]).mean()},
    ]

    return pd.DataFrame(rows), pd.DataFrame({
        "Volatility Shock Mean-Reverter": compute_pnl(s3, data["Returns"]),
        "Event-Driven Liquidity Rebound Trader": compute_pnl(s6, data["Returns"]),
        "Volatility Compression Momentum Trader": compute_pnl(s7, data["Returns"]),
        "Equal-Weight Multi-Persona Portfolio": compute_pnl(portfolio, data["Returns"]),
    })

# =========================================================
# ------------------------ MAIN ---------------------------
# =========================================================

if __name__ == "__main__":

    project = QuantaFellowshipProject()
    project.load_data()

    train = project.train_data
    val = project.validation_data
    blind = project.blind_data

    metrics_train, pnl_train = evaluate_all(train)
    metrics_val, pnl_val = evaluate_all(val)
    metrics_blind, pnl_blind = evaluate_all(blind)

    final_metrics = pd.concat(
        [
            metrics_train.assign(Dataset="Train"),
            metrics_val.assign(Dataset="Validation"),
            metrics_blind.assign(Dataset="Blind"),
        ],
        ignore_index=True
    )

    print("\n=== Sharpe & Mean PnL Metrics ===")
    print(final_metrics)

    # ------------------ PNL PLOTS ------------------
    pnl_all = pd.concat([pnl_train, pnl_val, pnl_blind])
    cum_pnl = pnl_all.cumsum()

    plt.figure(figsize=(12, 6))
    for col in cum_pnl.columns:
        plt.plot(cum_pnl.index, cum_pnl[col], label=col)

    plt.axvline(pnl_train.index[-1], linestyle="--")
    plt.axvline(pnl_val.index[-1], linestyle="--")

    plt.title("Cumulative PnL — Train / Validation / Blind")
    plt.xlabel("Date")
    plt.ylabel("Cumulative PnL")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ------------------ CORRELATION ------------------
    signal_pnls = pnl_all.drop(columns=["Equal-Weight Multi-Persona Portfolio"])
    corr = signal_pnls.corr()

    print("\n=== PnL Correlation Matrix (Signals Only) ===")
    print(corr.round(3))

    plt.figure(figsize=(6, 5))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")

    for i in range(len(corr)):
        for j in range(len(corr)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}",
                     ha="center", va="center", color="black", fontsize=9)

    plt.xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr)), corr.columns)
    plt.title("Signal PnL Correlation Heatmap")
    plt.tight_layout()
    plt.show()
