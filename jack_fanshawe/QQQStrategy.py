import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import joblib

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
QQQ_FILE = 'Quanta Fellowship Train & Validate.csv'
MACRO_FILE = 'master_df_2.csv'
BLIND_FILE = 'QQQ Fellowship Blind Out of Sample.csv'

# ---------------------------------------------------------
# 2. DATA LOADING & PREP
# ---------------------------------------------------------
def load_and_prep():
    print("Loading data...")
    # Load Main Data
    qqq = pd.read_csv(QQQ_FILE)
    qqq.columns = qqq.columns.str.strip()
    qqq['Date'] = pd.to_datetime(qqq['Time'], errors='coerce')
    qqq = qqq.dropna(subset=['Date']).set_index('Date').sort_index()
    qqq = qqq[['Open', 'High', 'Low', 'Latest', 'Volume']].rename(columns={'Latest': 'Close'})

    # Load Blind Data
    blind = pd.read_csv(BLIND_FILE)
    blind.columns = blind.columns.str.strip()
    blind['Date'] = pd.to_datetime(blind['Time'], errors='coerce')
    blind = blind.dropna(subset=['Date']).set_index('Date').sort_index()
    if 'Latest' in blind.columns:
        blind = blind.rename(columns={'Latest': 'Close'})
    blind = blind[['Open', 'High', 'Low', 'Close', 'Volume']]

    # Load Macro
    macro = pd.read_csv(MACRO_FILE)
    macro.columns = macro.columns.str.strip()
    macro['Date'] = pd.to_datetime(macro['observation_date'], errors='coerce')
    macro = macro.dropna(subset=['Date']).set_index('Date').sort_index()

    # Merge
    df_main = qqq.join(macro, how='left').fillna(method='ffill')
    df_blind = blind.join(macro, how='left').fillna(method='ffill')
    
    return df_main, df_blind

# ---------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------
def calculate_features(df_in):
    df = df_in.copy()
    
    # Lag macro data 1 day
    ohlcv = ['Open', 'High', 'Low', 'Close', 'Volume']
    macro_cols = [c for c in df.columns if c not in ohlcv]
    df[macro_cols] = df[macro_cols].shift(1)

    # Target
    df['Next_Return'] = df['Close'].shift(-1) / df['Close'] - 1
    df['Target'] = (df['Next_Return'] > 0).astype(int)
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))

    # --- BASE FEATURES ---
    df['Sig_ROC_20'] = df['Close'].pct_change(20)
    tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift(1)).abs(), (df['Low']-df['Close'].shift(1)).abs()], axis=1).max(axis=1)
    df['Sig_ATR_Ratio'] = tr.rolling(5).mean() / tr.rolling(20).mean()
    df['Sig_Price_SMA200'] = df['Close'] / df['Close'].rolling(200).mean() - 1
    
    realized_vol = df['Log_Ret'].rolling(21).std() * np.sqrt(252) * 100
    df['Sig_VRP'] = (df['VIX_Close'] - realized_vol) if 'VIX_Close' in df.columns else 0
    
    delta = df['Close'].diff()
    gain = delta.where(delta>0, 0).rolling(14).mean()
    loss = (-delta.where(delta<0, 0)).rolling(14).mean().replace(0, 0.001)
    rsi = 100 - (100 / (1 + gain/loss))
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
    
    # Macro/Credit
    if 'NFCI' in df.columns:
        df['Sig_Fin_Stress'] = df['NFCI']
        df['Sig_NFCI_Delta'] = df['NFCI'].diff(5)
        df['New_NFCI_Level'] = df['NFCI']
        df['New_NFCI_Accel'] = df['NFCI'].diff().diff()
    else:
        for c in ['Sig_Fin_Stress', 'Sig_NFCI_Delta', 'New_NFCI_Level', 'New_NFCI_Accel']: df[c] = 0
        
    if 'BAA10Y' in df.columns:
        df['Sig_Corr_Credit_QQQ'] = df['Close'].rolling(60).corr(df['BAA10Y'])
        df['New_Credit_ROC_20'] = df['BAA10Y'].pct_change(20)
        df['New_Credit_VIX_Mult'] = df['BAA10Y'] * df['VIX_Close'] if 'VIX_Close' in df.columns else 0
    else:
        for c in ['Sig_Corr_Credit_QQQ', 'New_Credit_ROC_20', 'New_Credit_VIX_Mult']: df[c] = 0

    # VIX Derived
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

    # --- NEW REGIME SIGNALS ---
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

# ---------------------------------------------------------
# 4. EXECUTION
# ---------------------------------------------------------
def run_backtest():
    # 1. Load & PreP
    df_main, df_blind = load_and_prep()
    
    # STITCH FIRST
    df_full = pd.concat([df_main, df_blind])
    df_full = df_full[~df_full.index.duplicated(keep="first")].sort_index()
    
    # CALCULATE ONCE
    df_full = calculate_features(df_full).replace([np.inf, -np.inf], np.nan)

    # SPLIT SECOND
    train_data = df_full.loc['2000-01-01':'2015-12-31'].dropna()
    val_data   = df_full.loc['2016-01-01':'2021-12-31'].dropna()
    blind_data = df_full.loc['2022-01-01':].dropna()

    # 4. Define Feature Set
    optimal_features = ["New_Returns_Skew", "New_Panic_Indicator", "New_Shadow_Ratio", "New_SKEW_Level", "New_Overnight_Gap", 
                        "New_Signal_Inflation_Exp", "Sig_NFCI_Delta", "New_Tech_Stress_Rel", "New_Signal_SMH_Relative",
                          "New_Yield_Stock_Corr", "New_VIX_Stock_Corr", "New_NFCI_Level", "New_MeanRev_VXN", "Sig_VRP", 
                          "Strat_Hurst_Proxy", "Sig_Price_SMA200", "New_Signal_Yield_Vol", "New_Z_SKEW", "New_VVIX_VIX_Ratio", 
                          "Strat_VWAP_Dev_5", "New_Z_VVIX", "Sig_CMF_Proxy", "Sig_Vol_ROC_5", "New_Signal_SMH_MOM", "New_PV_Corr", 
                          "New_NFCI_Accel", "New_SKEW_MA_Div"]
    
    # 4. Define Feature Set
    optimal_features = ["New_Returns_Skew", "New_Panic_Indicator", "New_Shadow_Ratio", "New_SKEW_Level", "New_Overnight_Gap", 
                        "New_Signal_Inflation_Exp", "New_Tech_Stress_Rel", "New_Signal_SMH_Relative",
                          "New_Yield_Stock_Corr", "New_VIX_Stock_Corr", "New_MeanRev_VXN", "Sig_VRP", 
                          "Strat_Hurst_Proxy", "Sig_Price_SMA200", "New_Signal_Yield_Vol", "New_Z_SKEW", "New_VVIX_VIX_Ratio", 
                          "Strat_VWAP_Dev_5", "New_Z_VVIX", "Sig_CMF_Proxy", "Sig_Vol_ROC_5", "New_Signal_SMH_MOM", "New_PV_Corr", 
                          "New_SKEW_MA_Div"]
    
    

    # Check for missing features
    # missing = [f for f in optimal_features if f not in train_data.columns]
    # if missing:
    #     print(f"Error: Missing features: {missing}")
    #     return

    # 5. Train Model
    print(f"Training Random Forest on {len(train_data)} samples...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=50, random_state=42, n_jobs=-1)
    rf.fit(train_data[optimal_features], train_data['Target'])

    # 6. Predictions
    print("Generating predictions...")
    probs_train = rf.predict_proba(train_data[optimal_features])[:, 1]
    probs_val = rf.predict_proba(val_data[optimal_features])[:, 1]
    probs_blind = rf.predict_proba(blind_data[optimal_features])[:, 1]

    # 7. Apply Leverage Logic
    def lev_5(p):
        if p > 0.58: return 1.5
        if p>0.56: return 1.25
        if p > 0.55: return 1.0
        if p>0.545: return 0.5
        if p > 0.50: return 0.0
        if p>0.48: return -0.25
        if p>0.46: return -0.5
        return -1.0

    pos_train = pd.Series([lev_5(p) for p in probs_train], index=train_data.index)
    pos_val = pd.Series([lev_5(p) for p in probs_val], index=val_data.index)
    pos_blind = pd.Series([lev_5(p) for p in probs_blind], index=blind_data.index)
    
    # ---------------------------------------------------------
    # OOS EXPOSURE PLOT
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 4))
    plt.step(pos_blind.index, pos_blind.values, where='post')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title("Blind OOS Daily Exposure (Leverage)")
    plt.ylabel("Exposure")
    plt.xlabel("Date")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


    # 8. Returns & Equity
    ret_train = pos_train * train_data['Next_Return']
    ret_val = pos_val * val_data['Next_Return']
    ret_blind = pos_blind * blind_data['Next_Return']

    eq_train_val = (1 + pd.concat([ret_train, ret_val])).cumprod()
    eq_blind = (1 + ret_blind).cumprod()

    # 9. Sharpe Ratios
    def get_sharpe_ratio(r):
        if r.std() == 0: return 0
        return (r.mean() * 252) / (r.std() * np.sqrt(252))

    sr_train = get_sharpe_ratio(ret_train)
    sr_val = get_sharpe_ratio(ret_val)
    sr_blind = get_sharpe_ratio(ret_blind)

    active_mask = (pos_blind != 0)
    active_ret = ret_blind[active_mask]
    win_rate_active = (active_ret > 0).mean() if len(active_ret) > 0 else 0

    # 10. Reporting
    print("\n" + "="*40)
    print("STRATEGY PERFORMANCE REPORT")
    print("="*40)
    print(f"Training Sharpe (2000-2015):   {sr_train:.4f}")
    print(f"Validation Sharpe (2016-2021): {sr_val:.4f}")
    print(f"Blind OOS Sharpe (2022-Present): {sr_blind:.4f}")
    
    print("\n" + "="*40)
    print("BLIND OOS STATS")
    print("="*40)
    print(f"Total Return: {(eq_blind.iloc[-1]-1)*100:.2f}%")
    print(f"Active Win Rate:  {win_rate_active:.2%}")
    print(f"Max Drawdown: {(eq_blind / eq_blind.cummax() - 1).min():.2%}")

    # 11. Plotting
    # Plot 1: Train & Validation
    plt.figure(figsize=(12, 6))
    plt.plot(eq_train_val, label='Strategy Equity (Train+Val)')
    plt.yscale('log')
    plt.title(f"In-Sample Equity Curve (Train SR: {sr_train:.2f}, Val SR: {sr_val:.2f})")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

    # Plot 2: Blind OOS
    plt.figure(figsize=(12, 6))
    plt.plot(eq_blind, label='Strategy Equity (Blind OOS)', color='green')
    plt.plot((1 + blind_data['Next_Return']).cumprod(), label='Buy & Hold QQQ', color='gray', alpha=0.5, linestyle='--')
    plt.yscale('log')
    plt.title(f"Blind Out-of-Sample Performance (SR: {sr_blind:.2f})")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

if __name__ == "__main__":
    run_backtest()