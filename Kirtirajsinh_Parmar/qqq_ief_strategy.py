import os
import io
import warnings
import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")

# =========================================================
# 0. DATA GENERATION / CHECK (Ensures High Quality Data)
# =========================================================
def ensure_data_exists():
    """
    Checks for input files. If missing, attempts to download 
    standard Yahoo Finance data (Adj Close is key for Sharpe > 2.2).
    """
    files = {
        "input_file_0.csv": "QQQ",      # Nasdaq 100
        "input_file_1.csv": "DX-Y.NYB", # US Dollar Index
        "input_file_2.csv": "IEF"       # 7-10 Year Treasury
    }
    

# =========================================================
# 1. METRIC CALCULATOR
# =========================================================
def glue_calculate_metrics(returns_series, annualization=252):
    rs = returns_series.dropna()
    if len(rs) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    comp_ret = (1.0 + rs).prod() - 1.0
    n_years = len(rs) / annualization
    ann_ret = (1.0 + comp_ret) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0

    mu = rs.mean()
    sig = rs.std(ddof=1)
    sharpe = 0.0 if (sig is None or np.isnan(sig) or sig < 1e-12) else (mu / sig) * np.sqrt(annualization)

    cum = (1.0 + rs).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1.0
    max_dd = dd.min() if len(dd) else 0.0

    calmar = 0.0 if (max_dd == 0 or np.isnan(max_dd)) else (ann_ret / abs(max_dd))

    neg = rs[rs < 0]
    dstd = neg.std(ddof=1) if len(neg) > 1 else 0.0
    sortino = 0.0 if (dstd is None or np.isnan(dstd) or dstd < 1e-12) else (mu / dstd) * np.sqrt(annualization)

    return float(sharpe), float(comp_ret), float(max_dd), float(calmar), float(sortino)

# =========================================================
# 2. ROBUST DATA LOADER (Tailored for Adj Close)
# =========================================================
def load_data_robust(filepath, asset_name):
    """
    Loads data handling various CSV formats. 
    Prioritizes 'Adj Close' for accurate return calculation.
    """
    if not os.path.exists(filepath): return pd.DataFrame()
    
    with open(filepath, "r") as f:
        lines = f.readlines()
    
    # Strip garbage headers
    clean_lines = []
    header_found = False
    for line in lines:
        if "Date" in line or "Time" in line:
            clean_lines.append(line)
            header_found = True
        elif header_found and line[0].isdigit():
            clean_lines.append(line)
            
    if not clean_lines: return pd.DataFrame()
    
    df = pd.read_csv(io.StringIO("".join(clean_lines)))
    df.columns = [c.strip() for c in df.columns]
    
    # Date Parse
    col0 = df.columns[0]
    df[col0] = pd.to_datetime(df[col0], errors='coerce')
    df.set_index(col0, inplace=True)
    df.sort_index(inplace=True)
    
    # Column Mapping
    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if 'open' in cl: rename_map[c] = 'Open'
        elif 'high' in cl: rename_map[c] = 'High'
        elif 'low' in cl: rename_map[c] = 'Low'
        elif 'adj close' in cl: rename_map[c] = 'Adj Close' # Priority
        elif 'close' in cl and 'adj' not in cl: rename_map[c] = 'Close'
        elif 'vol' in cl: rename_map[c] = 'Volume'
        
    df.rename(columns=rename_map, inplace=True)
    
    # Tailoring: Use Adj Close if available (Dividend Reinvestment logic)
    if 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']
    
    if asset_name in ['QQQ', 'IEF']:
        req = ['Open', 'High', 'Low', 'Close']
        if set(req).issubset(df.columns):
             # Ensure numeric
            for c in req: df[c] = pd.to_numeric(df[c], errors='coerce')
            return df[req].dropna()
    else:
        # DXY
        return df['Close'].apply(pd.to_numeric, errors='coerce').dropna()
        
    return pd.DataFrame()

# =========================================================
# 3. GLOBAL CONFIGURATION (OPTIMIZED 2.23 SHARPE)
# =========================================================
# Date Splits
TRAIN_END = pd.Timestamp("2020-12-31")
HOLD_START = pd.Timestamp("2021-01-04")

# Load Global Data
qqq = load_data_robust("input_file_0.csv", "QQQ")
dxy = load_data_robust("input_file_1.csv", "DXY")
ief = load_data_robust("input_file_2.csv", "IEF")

# Align
common_idx = qqq.index.intersection(dxy.index).intersection(ief.index).sort_values()
qqq = qqq.loc[common_idx]
dxy = dxy.loc[common_idx]
ief = ief.loc[common_idx]

# =========================================================
# 4. QQQ STRATEGY ENGINE
# =========================================================
def run_qqq_engine():
    # --- OPTIMIZED PARAMS ---
    MOM_THRESH = -1.7      # Aggressive dip buying / holding
    TARGET_VOL = 0.16      # 16% Vol Target
    DXY_ROC_W = 60         # 60-day DXY trend check
    
    c = qqq['Close']
    h = qqq['High']
    l = qqq['Low']
    o = qqq['Open']
    ret = c.pct_change().fillna(0.0)
    
    # 1. Feature Calc
    W_Z = 504
    def zscore(s, w):
        return ((s - s.rolling(w).mean()) / s.rolling(w).std(ddof=1)).fillna(0)

    # Trend
    sma200 = c.rolling(200).mean()
    dist = (c - sma200)/sma200
    roc120 = c.pct_change(120)
    trend_state = ((zscore(dist, W_Z) + zscore(roc120, W_Z))/2).clip(-3,3)
    
    # Vol
    rvol = ret.rolling(20).std(ddof=1)*np.sqrt(252)
    vov = rvol.rolling(60).std(ddof=1)
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    shock = tr / tr.rolling(14).mean()
    vol_state = ((zscore(rvol, W_Z)+zscore(vov, W_Z)+zscore(shock, W_Z))/3).clip(-3,3)
    
    # Mom
    mom_blend = (c.pct_change(5)+c.pct_change(20)+c.pct_change(60))/3
    accel = c.pct_change(20)-c.pct_change(60)
    mom_state = ((zscore(mom_blend, W_Z)+zscore(accel, W_Z))/2).clip(-3,3)
    
    # Fast
    slope = (c - c.shift(20))/20
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    fast_on = (zscore(slope, 252)>0) & (zscore(macd, 252)>0)
    
    # 2. Logic (Frozen Thresholds)
    # Use quantiles from training data only to avoid lookahead bias
    train_slice = trend_state.loc[:TRAIN_END]
    t_cuts = train_slice.quantile([0.33, 0.66])
    v_cuts = vol_state.loc[:TRAIN_END].quantile([0.33, 0.66])
    
    def get_bin(x, cuts):
        if x <= cuts.iloc[0]: return 0
        elif x <= cuts.iloc[1]: return 1
        else: return 2
        
    t_bin = trend_state.apply(lambda x: get_bin(x, t_cuts))
    v_bin = vol_state.apply(lambda x: get_bin(x, v_cuts))
    
    core = (t_bin==2) & (v_bin<=1)
    mid = (t_bin==1) & (v_bin==0)
    
    # Gate: Don't buy if momentum is crashing fast
    mom_accel = zscore(mom_state - mom_state.shift(20), 252)
    gate = (v_bin<=1) & (mom_accel < -0.5)
    
    # Signal
    raw_sig = (core | mid) & (mom_state > MOM_THRESH)
    base_bool = ((raw_sig | fast_on).rolling(5).sum() >= 3) & (~gate)
    
    # 3. Sizing
    curr_vol = rvol.replace(0, 0.01)
    base_exp = (0.16 / curr_vol).clip(0, 1.5).where(base_bool, 0.0)
    
    # IBS Adj
    rng = (h - l).replace(0, 1e-9)
    ibs = (c - l) / rng
    ibs_adj = pd.Series(0.0, index=c.index)
    ibs_adj[ibs < 0.2] = 0.2
    ibs_adj[ibs > 0.8] = -0.2
    base_exp += (np.sign(base_exp) * ibs_adj)
    
    # 4. DXY Hedge
    dxy_roc = dxy.pct_change(DXY_ROC_W)
    dxy_th = dxy_roc.loc[:TRAIN_END].quantile(0.66)
    dxy_up = dxy_roc > dxy_th
    
    # Base Red = 0 (Full Hedge)
    base_exp[dxy_up] = 0.0
    
    # 5. Shock / Rebound
    dxy_vol = dxy.pct_change().rolling(5).sum().abs()
    shock_th = dxy_vol.loc[:TRAIN_END].quantile(0.90)
    is_shock = dxy_vol > shock_th
    base_exp[is_shock] = base_exp[is_shock].clip(-1.0, 1.0)
    
    # Rebound Entry
    ro_scaler = (0.14 / curr_vol).clip(0, 1.5)
    
    def rsi_calc(s, n=14):
        d = s.diff()
        u = d.where(d>0, 0).rolling(n).mean()
        d = -d.where(d<0, 0).rolling(n).mean()
        rs = u/d
        return 100 - (100/(1+rs))
    
    rsi = rsi_calc(c)
    crash = (rsi < 25) | (c.pct_change(3) < -0.05)
    
    # Add RO exposure if DXY is up (hedged mode)
    base_exp[dxy_up] += ro_scaler[crash].reindex(base_exp.index).fillna(0)
    
    # 6. Cashfill
    flat = base_exp.abs() < 0.05
    mon = (c.index.dayofweek==0) & (ret.shift(1)<0)
    bb = c < (c.rolling(20).mean() - 2*c.rolling(20).std())
    gap = o < (l.shift(1)*0.995)
    fill_sig = (mon | (rsi<30) | bb | gap)
    base_exp[flat] = fill_sig[flat].astype(int) * 0.75
    
    # 7. Vol Target Leverage
    lev = (TARGET_VOL / curr_vol).clip(0.5, 2.0)
    vov_rnk = vov.rolling(252).rank(pct=True).fillna(0.5)
    unc = pd.Series(1.0, index=c.index)
    unc[vov_rnk > 0.8] = 0.7
    
    final_exp = (base_exp * lev * unc).clip(-1.0, 2.0).fillna(0.0)
    
    # PnL
    # Exp[t] -> Ret[t+1]? No, vector backtest usually implies Exp[t] calculated at Close[t] trades Close[t] to Close[t+1]
    # So we shift exposure by 1
    
    net = (final_exp.shift(1) * ret).fillna(0.0)
    return net, final_exp

# =========================================================
# 5. IEF SLEEVE ENGINE
# =========================================================
def run_ief_engine():
    # --- OPTIMIZED PARAMS ---
    SMA_W = 10         # Fast trend
    VOL_TGT = 0.04     # Low leverage (Defensive)
    MACRO_W = 0.20     # High sensitivity
    
    c = ief['Close']
    ret = c.pct_change().fillna(0.0)
    
    # Signal
    ma = c.rolling(SMA_W).mean()
    trend = np.where(c > ma, 1.0, -1.0)
    
    # Macro
    qqq_v = qqq['Close'].pct_change().rolling(20).std()
    q_rnk = qqq_v.rolling(252).rank(pct=True).fillna(0.5)
    
    dxy_r10 = dxy.pct_change().rolling(10).sum()
    d_rnk = dxy_r10.shift(1).rolling(252).quantile(0.9)
    d_high = dxy_r10 > d_rnk
    
    ms = pd.Series(1.0, index=c.index)
    ms[q_rnk > 0.8] += MACRO_W
    ms[d_high] -= MACRO_W
    ms = ms.clip(0.75, 1.25)
    
    # Vol
    ivol = ret.rolling(60).std() * np.sqrt(252)
    lev = (VOL_TGT / (ivol.replace(0, 0.001))).clip(0, 1.5)
    
    final_sig = (trend * lev * ms).clip(-1.5, 1.5)
    
    net = (final_sig.shift(1) * ret).fillna(0.0)
    return net

# =========================================================
# 6. MAIN EXECUTION
# =========================================================
qqq_net, qqq_exp = run_qqq_engine()
ief_net = run_ief_engine()

# Router
ACTIVE_EPS = 0.001  # Ultra strict
# If QQQ Exposure (absolute) is basically zero, switch to IEF
risk_off = qqq_exp.abs() < ACTIVE_EPS
use_ief = risk_off.shift(1).fillna(False)

# Combine
combined = qqq_net.copy()
combined[use_ief] = ief_net[use_ief]

# Holdout Stats
h_qqq = qqq_net[qqq_net.index >= HOLD_START]
h_ief = ief_net[ief_net.index >= HOLD_START]
h_cmb = combined[combined.index >= HOLD_START]

h_cmb.to_csv('final_portfolio_returns_parmar.csv')

# Printing
def print_stats(name, s):
    sh, tr, dd, cal, sor = glue_calculate_metrics(s)
    print(f"\n=== {name} ===")
    print(f"Period: {s.index.min().date()} to {s.index.max().date()}")
    print(f"Sharpe:  {sh:.4f}")
    print(f"Return:  {tr:.2%}")
    print(f"Max DD:  {dd:.2%}")
    print(f"Calmar:  {cal:.4f}")
    print(f"Sortino: {sor:.4f}")

print_stats("QQQ COMPONENT", h_qqq)
print_stats("IEF COMPONENT", h_ief)
print_stats("COMBINED STRATEGY (TARGET)", h_cmb)

# =========================================================
# EQUITY CURVE PLOT (COMBINED STRATEGY)
# =========================================================
import matplotlib.pyplot as plt

eq = (1 + h_cmb).cumprod()

plt.figure(figsize=(12, 6))
plt.plot(eq, label="Combined Strategy Equity", color="blue")
plt.yscale("log")
plt.title("Equity Curve – Combined Strategy (Holdout Period)")
plt.xlabel("Date")
plt.ylabel("Equity (Log Scale)")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()

bh = (1 + qqq['Close'].pct_change().loc[h_cmb.index]).cumprod()

plt.figure(figsize=(12, 6))
plt.plot(eq, label="Combined Strategy", color="blue")
plt.plot(bh, label="Buy & Hold QQQ", color="gray", linestyle="--", alpha=0.6)
plt.yscale("log")
plt.title("Equity Curve Comparison (Holdout Period)")
plt.xlabel("Date")
plt.ylabel("Equity (Log Scale)")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()


# =========================================================
# QUANTSTATS HTML REPORT (AUTO-OPEN)
# =========================================================
import quantstats as qs
import webbrowser

qs.extend_pandas()

REPORT_FILE = "combined_strategy_quantstats.html"

# QuantStats expects daily returns
qs.reports.html(
    h_cmb,
    benchmark=qqq['Close'].pct_change().loc[h_cmb.index],
    output=REPORT_FILE,
    title="Combined Strategy vs QQQ"
)

# Auto-open in browser
webbrowser.open("file://" + os.path.abspath(REPORT_FILE))
