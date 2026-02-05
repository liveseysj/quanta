import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from scipy.optimize import minimize

# =========================================================
# CONFIG
# =========================================================
QQQ_FILE   = 'Quanta Fellowship Train & Validate.csv'
BLIND_FILE = 'QQQ Fellowship Blind Out of Sample.csv'

TLT_FILE   = 'TLT.csv'
MACRO_FILE = 'master_df_2.csv'

TRAIN_START = '2000-01-01'
TRAIN_END   = '2015-12-31'
VAL_START   = '2016-01-01'
VAL_END     = '2021-12-31'
BLIND_START = '2022-01-01'

# 'Q_Stat_Vol_21' is required for the crash protection logic
META_FEATURES = ['P_Q', 'P_T', 'P_Spread', 'Corr_Q_T_10', 'Ratio_Trend_5', 'Corr_Q_T_30', 
                 'Q_Stat_Vol_21', 'T_Stat_Vol_21', 'Q_New_BS_ATM_Cost', 'T_Sig_Fin_Stress', 
                 'Q_Sig_ATR_Ratio', 'Q_Sig_CMF_Proxy', 'Q_New_Signal_Yield_Vol', 
                 'Q_New_Regime_HighVol', 'T_Sig_VRP', 'T_FEDFUNDS']

# =========================================================
# SHARED FEATURE ENGINEERING
# =========================================================
def calculate_features(df_in):
    df = df_in.copy()
    ohlcv = ['Open', 'High', 'Low', 'Close', 'Volume']
    macro_cols = [c for c in df.columns if c not in ohlcv]
    df[macro_cols] = df[macro_cols].shift(1)

    df['Next_Return'] = df['Close'].shift(-1) / df['Close'] - 1
    df['Target'] = (df['Next_Return'] > 0).astype(int)
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))

    df['Sig_ROC_20'] = df['Close'].pct_change(20)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift(1)).abs(),
        (df['Low'] - df['Close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    df['Sig_ATR_Ratio'] = tr.rolling(5).mean() / tr.rolling(20).mean()
    df['Sig_Price_SMA200'] = df['Close'] / df['Close'].rolling(200).mean() - 1

    # Volatility in Percentage Terms (e.g. 20.0 = 20%)
    realized_vol = df['Log_Ret'].rolling(21).std() * np.sqrt(252) * 100
    df['Sig_VRP'] = (df['VIX_Close'] - realized_vol) if 'VIX_Close' in df.columns else 0

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean().replace(0, 0.001)
    rsi = 100 - (100 / (1 + gain / loss))
    df['Sig_RSI_Vol'] = rsi.rolling(10).std()

    df['Sig_Vol_ROC_5'] = df['Volume'].pct_change(5)

    tp = (df['High'] + df['Low'] + df['Close']) / 3
    mean_dev = 0.015 * tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x)))).replace(0, 0.001)
    df['Sig_CCI'] = (tp - tp.rolling(20).mean()) / mean_dev

    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']).replace(0, 0.01)
    df['Sig_CMF_Proxy'] = clv * df['Volume']

    efficiency_noise = df['Close'].diff().abs().rolling(10).sum().replace(0, 0.001)
    df['Strat_KER_10'] = (df['Close'] - df['Close'].shift(10)).abs() / efficiency_noise

    range_hl = (df['High'] - df['Low']).replace(0, 0.01)
    df['Strat_Intraday_Power'] = (df['Close'] - df['Open']) / range_hl
    df['Strat_Vol_Surprise'] = df['Volume'] / df['Volume'].rolling(20).mean()

    if 'NFCI' in df.columns:
        df['Sig_Fin_Stress'] = df['NFCI']
        df['Sig_NFCI_Delta'] = df['NFCI'].diff(5)
        df['New_NFCI_Level'] = df['NFCI']
        df['New_NFCI_Accel'] = df['NFCI'].diff().diff()
    else:
        for c in ['Sig_Fin_Stress', 'Sig_NFCI_Delta', 'New_NFCI_Level', 'New_NFCI_Accel']:
            df[c] = 0

    if 'BAA10Y' in df.columns:
        df['Sig_Corr_Credit_QQQ'] = df['Close'].rolling(60).corr(df['BAA10Y'])
        df['New_Credit_ROC_20'] = df['BAA10Y'].pct_change(20)
        df['New_Credit_VIX_Mult'] = df['BAA10Y'] * df['VIX_Close'] if 'VIX_Close' in df.columns else 0
    else:
        for c in ['Sig_Corr_Credit_QQQ', 'New_Credit_ROC_20', 'New_Credit_VIX_Mult']:
            df[c] = 0

    if 'VIX_Close' in df.columns:
        df['New_Z_VIX'] = (df['VIX_Close'] - df['VIX_Close'].rolling(60).mean()) / df['VIX_Close'].rolling(60).std()
        df['New_Regime_HighVol'] = (df['VIX_Close'] > 20).astype(int)
        df['New_VIX_Price_Div'] = df['VIX_Close'] / df['Close']
        df['New_VIX_Stock_Corr'] = df['Close'].rolling(30).corr(df['VIX_Close'])

        if 'VVIX' in df.columns:
            df['New_Z_VVIX'] = (df['VVIX'] - df['VVIX'].rolling(60).mean()) / df['VVIX'].rolling(60).std()
            df['New_VVIX_VIX_Ratio'] = df['VVIX'] / df['VIX_Close']
            df['New_Panic_Indicator'] = ((df['VVIX'] > 110) & (df['VIX_Close'] > 30)).astype(int)

        if 'VXN' in df.columns:
            df['New_VXN_ROC_5'] = df['VXN'].pct_change(5)
            df['New_Tech_Stress_Rel'] = df['VXN'] / df['VIX_Close']
            df['New_MeanRev_VXN'] = df['VXN'] - df['VXN'].rolling(50).mean()
            df['New_VXN_MA_Div'] = df['VXN'] / df['VXN'].rolling(50).mean() - 1
            T = 30.0 / 365.0
            df['New_BS_ATM_Cost'] = 0.4 * df['Close'] * (df['VXN'] / 100.0) * np.sqrt(T)

    if 'SKEW_Close' in df.columns:
        df['New_SKEW_Level'] = df['SKEW_Close']
        df['New_SKEW_MA_Div'] = df['SKEW_Close'] / df['SKEW_Close'].rolling(50).mean() - 1
        df['New_Z_SKEW'] = (df['SKEW_Close'] - df['SKEW_Close'].rolling(60).mean()) / df['SKEW_Close'].rolling(60).std()

    if 'T10Y2Y' in df.columns:
        df['New_Yield_Stock_Corr'] = df['Close'].rolling(30).corr(df['T10Y2Y'])
        df['New_Yield_Curve_Slope'] = df['T10Y2Y']

    vwap_5 = (df['Close'] * df['Volume']).rolling(5).sum() / df['Volume'].rolling(5).sum()
    df['Strat_VWAP_Dev_5'] = (df['Close'] - vwap_5) / vwap_5

    range_max_min = (df['High'].rolling(14).max() - df['Low'].rolling(14).min()).replace(0, 0.01)
    df['Strat_Hurst_Proxy'] = df['Close'].rolling(20).std() / range_max_min
    df['Strat_DayOfWeek'] = df.index.dayofweek

    if 'DGS10' in df.columns and 'T10YIE' in df.columns:
        df['New_Signal_Real_Yield'] = df['DGS10'] - df['T10YIE']
        df['New_Signal_Real_Yield_Delta_20'] = df['New_Signal_Real_Yield'].diff(20)
        df['New_Signal_Inflation_Exp'] = df['T10YIE']
        df['New_Signal_Yield_Vol'] = df['DGS10'].rolling(20).std()

    if 'SMH_close' in df.columns:
        df['New_Signal_SMH_Relative'] = df['SMH_close'] / df['Close']
        df['New_Signal_SMH_MOM'] = df['SMH_close'].pct_change(20)

    if 'XLY_close' in df.columns:
        df['New_Signal_XLY_Relative'] = df['XLY_close'] / df['Close']
    df['h_l'] = df['High'] - df['Low']
    df['c_o'] = df['Close'] - df['Open']
    df['ret'] = df['Close'].pct_change()
    df['New_Signal_Yield_Vol'] = df['DGS10'].rolling(20).std()
    df['New_Overnight_Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    df['New_Returns_Skew'] = df['ret'].rolling(20).skew()
    df['New_Shadow_Ratio'] = df['h_l'] / df['c_o'].abs().replace(0, 0.001)
    df['New_PV_Corr'] = df['Close'].rolling(10).corr(df['Volume'])
    df['Stability_ratio'] = df['New_Signal_Yield_Vol']/df['DGS10']
    log_ret = np.log(df['Close'] / df['Close'].shift(1))
    
    # Calculate Stat_Vol for Meta Model
    for w in [10, 21, 63]:
        df[f'Stat_Vol_{w}'] = log_ret.rolling(w).std() * np.sqrt(252) * 100 # Scaled to 0-100
    return df

# =========================================================
# METRICS & PIPELINES
# =========================================================
def sharpe_252(r):
    r = r.dropna()
    if r.std() == 0 or len(r) == 0: return 0.0
    return (r.mean() * 252) / (r.std() * np.sqrt(252))

def max_drawdown(eq):
    eq = eq.dropna()
    if len(eq) == 0: return 0.0
    return (eq / eq.cummax() - 1).min()

def load_and_prep_qqq():
    print("Loading QQQ...")
    qqq = pd.read_csv(QQQ_FILE)
    qqq.columns = qqq.columns.str.strip()
    qqq['Date'] = pd.to_datetime(qqq['Time'], errors='coerce')
    qqq = qqq.dropna(subset=['Date']).set_index('Date').sort_index()
    qqq = qqq[['Open', 'High', 'Low', 'Latest', 'Volume']].rename(columns={'Latest': 'Close'})

    blind = pd.read_csv(BLIND_FILE)
    blind.columns = blind.columns.str.strip()
    blind['Date'] = pd.to_datetime(blind['Time'], errors='coerce')
    blind = blind.dropna(subset=['Date']).set_index('Date').sort_index()
    if 'Latest' in blind.columns: blind = blind.rename(columns={'Latest': 'Close'})
    blind = blind[['Open', 'High', 'Low', 'Close', 'Volume']]

    macro = pd.read_csv(MACRO_FILE)
    macro.columns = macro.columns.str.strip()
    macro['Date'] = pd.to_datetime(macro['observation_date'], errors='coerce')
    macro = macro.dropna(subset=['Date']).set_index('Date').sort_index()

    df_main = qqq.join(macro, how='left').fillna(method='ffill')
    df_blind = blind.join(macro, how='left').fillna(method='ffill')
    return df_main, df_blind

def run_qqq_strategy():
    optimal_features = ["New_Returns_Skew", "New_Panic_Indicator", "New_Shadow_Ratio", "New_SKEW_Level", "New_Overnight_Gap", 
                        "New_Signal_Inflation_Exp", "Sig_NFCI_Delta", "New_Tech_Stress_Rel", "New_Signal_SMH_Relative",
                          "New_Yield_Stock_Corr", "New_VIX_Stock_Corr", "New_NFCI_Level", "New_MeanRev_VXN", "Sig_VRP", 
                          "Strat_Hurst_Proxy", "Sig_Price_SMA200", "New_Signal_Yield_Vol", "New_Z_SKEW", "New_VVIX_VIX_Ratio", 
                          "Strat_VWAP_Dev_5", "New_Z_VVIX", "Sig_CMF_Proxy", "Sig_Vol_ROC_5", "New_Signal_SMH_MOM", "New_PV_Corr", 
                          "New_NFCI_Accel", "New_SKEW_MA_Div"]

    df_main, df_blind = load_and_prep_qqq()
    df_full = pd.concat([df_main, df_blind])
    df_full = df_full[~df_full.index.duplicated(keep="first")].sort_index()
    df_full = calculate_features(df_full).replace([np.inf, -np.inf], np.nan)
    train = df_full.loc[TRAIN_START:TRAIN_END].dropna(subset=optimal_features + ['Target', 'Next_Return'])
    val   = df_full.loc[VAL_START:VAL_END].dropna(subset=optimal_features + ['Target', 'Next_Return'])
    blind = df_full.loc[BLIND_START:].dropna(subset=optimal_features + ['Target', 'Next_Return'])

    rf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=50, random_state=42, n_jobs=-1)
    rf.fit(train[optimal_features], train['Target'])

    probs_train = rf.predict_proba(train[optimal_features])[:, 1]
    probs_val   = rf.predict_proba(val[optimal_features])[:, 1]
    probs_blind = rf.predict_proba(blind[optimal_features])[:, 1]

    probs_full = pd.concat([
        pd.Series(probs_train, index=train.index),
        pd.Series(probs_val, index=val.index),
        pd.Series(probs_blind, index=blind.index)
    ]).sort_index()
    nextret_full = pd.concat([train['Next_Return'],val['Next_Return'],blind['Next_Return']]).sort_index()

    return {"name": "QQQ", "probs": probs_full, "next_ret": nextret_full, "feat_df": df_full.drop(columns=["Target","Next_Return"], errors="ignore")}

def run_tlt_strategy():
    print("Loading TLT...")
    TLT = pd.read_csv(TLT_FILE)
    macro = pd.read_csv(MACRO_FILE)
    def clean_and_merge(stock_df, macro_df):
        stock_df.columns = stock_df.columns.str.strip()
        stock_df['Date'] = pd.to_datetime(stock_df['Date'], errors='coerce')
        stock_df = stock_df.dropna(subset=['Date']).set_index('Date').sort_index()
        if 'Latest' in stock_df.columns: stock_df = stock_df.rename(columns={'Latest': 'Close'})
        stock_df = stock_df[['Open', 'High', 'Low', 'Close', 'Volume']]
        merged = stock_df.join(macro_df, how='left').fillna(method='ffill')
        return merged

    macro.columns = macro.columns.str.strip()
    macro['Date'] = pd.to_datetime(macro['observation_date'], errors='coerce')
    macro = macro.dropna(subset=['Date']).set_index('Date').sort_index()
    df_2 = clean_and_merge(TLT, macro)
    df_2.index = pd.to_datetime(df_2.index)

    optimal_features = ["Sig_Vol_ROC_5", "New_Signal_Real_Yield", "New_VXN_MA_Div", 
                        "Sig_CCI", "Stability_ratio", "New_Yield_Curve_Slope", "Sig_CMF_Proxy"]

    df_full = calculate_features(df_2).replace([np.inf, -np.inf], np.nan)
    train = df_full.loc[TRAIN_START:TRAIN_END].dropna(subset=optimal_features + ['Target', 'Next_Return'])
    val   = df_full.loc[VAL_START:VAL_END].dropna(subset=optimal_features + ['Target', 'Next_Return'])
    blind = df_full.loc[BLIND_START:].dropna(subset=optimal_features + ['Target', 'Next_Return'])

    rf = RandomForestClassifier(n_estimators=250, max_depth=5, min_samples_leaf=100, max_features=None, bootstrap=True, random_state=42, n_jobs=-1)
    rf.fit(train[optimal_features], train['Target'])

    probs_train = rf.predict_proba(train[optimal_features])[:, 1]
    probs_val   = rf.predict_proba(val[optimal_features])[:, 1]
    probs_blind = rf.predict_proba(blind[optimal_features])[:, 1]

    probs_full = pd.concat([pd.Series(probs_train, index=train.index), pd.Series(probs_val, index=val.index), pd.Series(probs_blind, index=blind.index)]).sort_index()
    nextret_full = pd.concat([train['Next_Return'],val['Next_Return'],blind['Next_Return']]).sort_index()
    return {"name": "TLT", "probs": probs_full, "next_ret": nextret_full, "feat_df": df_full.drop(columns=["Target","Next_Return"], errors="ignore")}

def apply_leverage(probs: pd.Series, next_ret: pd.Series, lev_func):
    df = pd.concat([probs.rename("p"), next_ret.rename("r")], axis=1).dropna()
    pos = df["p"].apply(lev_func)
    return pos * df["r"]

def build_meta_X(q, t):
    base = pd.concat([q["probs"].rename("P_Q"), t["probs"].rename("P_T")], axis=1).dropna()
    q_feat_df, t_feat_df = q["feat_df"], t["feat_df"]
    base['P_Spread'] = base['P_Q'] - base['P_T']
    base['P_Q_Mom'] = base['P_Q'].diff(5)
    base['P_T_Mom'] = base['P_T'].diff(5)
    q_log_ret = q_feat_df['Log_Ret'].reindex(base.index).fillna(0)
    t_log_ret = t_feat_df['Log_Ret'].reindex(base.index).fillna(0)

    for w in [10, 30, 90]:
        base[f'Corr_Q_T_{w}'] = q_log_ret.rolling(w).corr(t_log_ret)
        base[f'Cov_Q_T_{w}'] = q_log_ret.rolling(w).cov(t_log_ret)

    q_price = q_feat_df['Close'].reindex(base.index)
    t_price = t_feat_df['Close'].reindex(base.index)
    price_ratio = q_price / t_price
    base['Ratio_Trend_20'] = price_ratio.pct_change(20)
    base['Ratio_Trend_5'] = price_ratio.pct_change(5)

    vol_q = q_feat_df['Stat_Vol_21'].reindex(base.index)
    vol_t = t_feat_df['Stat_Vol_21'].reindex(base.index)
    base['Rel_Vol_Q_T'] = vol_q / (vol_t + 1e-9)

    qX = q["feat_df"].reindex(base.index).add_prefix("Q_")
    tX = t["feat_df"].reindex(base.index).add_prefix("T_")

    X_all = pd.concat([base, qX, tX], axis=1)
    X_all = X_all.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    X_all = X_all[META_FEATURES].dropna()
    return X_all

def build_meta_y(q, t, idx):
    rQ0 = q["next_ret"].reindex(idx)
    rT0 = t["next_ret"].reindex(idx)
    y = (rQ0 > rT0).astype(int)
    return y

def fit_meta_model(X_all, y_all):
    train_idx = X_all.loc[TRAIN_START:TRAIN_END].index
    Xtr = X_all.loc[train_idx]
    ytr = y_all.loc[train_idx]
    meta = RandomForestClassifier(n_estimators=250, max_depth=4, min_samples_leaf=100, random_state=42, n_jobs=-1)
    meta.fit(Xtr, ytr)
    return meta

# =========================================================
# NEW META WEIGHTS WITH OVERRIDES
# =========================================================
def meta_weights(meta, X_all, smooth=1, w_min=0.2, w_max=0.8, overrides=None):
    probs = pd.Series(meta.predict_proba(X_all)[:,1], index=X_all.index)

    wQ = probs.rolling(smooth).mean().fillna(0.5).clip(w_min, w_max)
    
    if overrides:
        mask_high = (probs > overrides['P_HIGH_THRESH'])
        wQ.loc[mask_high] = wQ.loc[mask_high] + overrides['W_BOOST']

        if 'Q_Stat_Vol_21' in X_all.columns:
            vol = X_all['Q_Stat_Vol_21']
            mask_crash = (vol > overrides['VOL_TRIGGER'])
            wQ.loc[mask_crash] = wQ.loc[mask_crash] - overrides['W_PENALTY']
        

        if 'P_LOW_THRESH' in overrides:
             mask_low = (probs < overrides['P_LOW_THRESH'])
             wQ.loc[mask_low] = wQ.loc[mask_low] + overrides['W_ADD']
    return wQ.clip(0.1, 0.9)

# =========================================================
# MAIN EXECUTION
# =========================================================
def main():
    q = run_qqq_strategy()
    t = run_tlt_strategy()
    X_all = build_meta_X(q, t)
    y_all = build_meta_y(q, t, X_all.index)
    meta = fit_meta_model(X_all, y_all)
    
    TUNABLE_OVERRIDES = {
        'P_HIGH_THRESH': 0.62, 
        'W_BOOST':       0.2, 
        
        'VOL_TRIGGER':   85, 
        'W_PENALTY':     0.35, 

        'P_LOW_THRESH':  0.4,
        'W_ADD':        0.2
    }
    
    # Pass the overrides dict here
    wQ_meta = meta_weights(meta, X_all, smooth=1, w_min=0.2, w_max=0.8, 
                           overrides=TUNABLE_OVERRIDES)

    # Leverage Functions (Same as your original)
    def QQQ_LEV(p):
        if p > 0.58: return 1.5
        if p>0.56: return 1.25
        if p > 0.55: return 1.0
        if p>0.545: return 0.5
        if p > 0.50: return 0.0
        if p>0.48: return -0.25
        if p>0.46: return -0.5
        return -1.0

    def TLT_LEV(p):
        if p > 0.58: return 1.5
        if p > 0.55: return 1.0
        if p>0.54: return 0.5
        if p > 0.45: return 0.0
        return -1.0

    r_q = apply_leverage(q["probs"], q["next_ret"], QQQ_LEV)
    r_t = apply_leverage(t["probs"], t["next_ret"], TLT_LEV)
    
    common = wQ_meta.index.intersection(r_q.index).intersection(r_t.index)
    w = wQ_meta.loc[common]
    r_meta = (w * r_q.loc[common] + (1 - w) * r_t.loc[common]).dropna()
    
    r_train = r_meta.loc[TRAIN_START:TRAIN_END]
    r_val   = r_meta.loc[VAL_START:VAL_END]
    r_blind = r_meta.loc[BLIND_START:]

    eq_train = (1 + r_train).cumprod()
    eq_val   = (1 + r_val).cumprod()
    eq_blind = (1 + r_blind).cumprod()

    print("\nMETA COMBO PERFORMANCE (With Heuristics)")
    print("-" * 40)
    print(f"Train SR: {sharpe_252(r_train):.3f} | MDD: {max_drawdown(eq_train):.2%}")
    print(f"Val   SR: {sharpe_252(r_val):.3f} | MDD: {max_drawdown(eq_val):.2%}")
    print(f"Blind SR: {sharpe_252(r_blind):.3f} | MDD: {max_drawdown(eq_blind):.2%}")

    r_meta.loc[BLIND_START:].to_csv('meta_returns_blind.csv')

    # ---- plots ----
    plt.figure(figsize=(12, 4))
    plt.plot(eq_train, label="Meta Combo (Train)")
    plt.yscale("log"); plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(); plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(eq_val, label="Meta Combo (Validation)")
    plt.yscale("log"); plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(); plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(eq_blind, label="Meta Combo (Blind)")
    plt.yscale("log"); plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(); plt.show()

    plt.figure(figsize=(12, 3))
    plt.plot(w, label="Weight in QQQ (Meta)")
    plt.axhline(TUNABLE_OVERRIDES['P_HIGH_THRESH'], color='r', linestyle='--', alpha=0.3, label="Conf Thresh")
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend(); plt.show()

if __name__ == "__main__":
    main()