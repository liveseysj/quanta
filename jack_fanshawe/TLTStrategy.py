import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import joblib

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
TLT = pd.read_csv('TLT.csv')
macro = pd.read_csv('master_df_2.csv')

def sharpe_252(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) == 0:
        return 0.0
    s = r.std()
    if s == 0 or np.isnan(s):
        return 0.0
    return (r.mean() * 252.0) / (s * np.sqrt(252.0))


# ---------------------------------------------------------
# 2. PREPROCESSING & MERGING FUNCTIONS
# ---------------------------------------------------------
def clean_and_merge(stock_df, macro_df):
    # 1. Standardize Columns
    stock_df.columns = stock_df.columns.str.strip()
    
    # 2. Fix Dates
    stock_df['Date'] = pd.to_datetime(stock_df['Date'], errors='coerce')
    stock_df = stock_df.dropna(subset=['Date']).set_index('Date').sort_index()
    
    # 3. Rename Target Column if needed
    if 'Latest' in stock_df.columns:
        stock_df = stock_df.rename(columns={'Latest': 'Close'})
        
    # 4. Keep only necessary columns to avoid join conflicts
    stock_df = stock_df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # 5. Merge with Macro (Forward Fill macro to match stock dates)
    merged = stock_df.join(macro_df, how='left').fillna(method='ffill')
    return merged

# Prep Global Macro
macro.columns = macro.columns.str.strip()
macro['Date'] = pd.to_datetime(macro['observation_date'], errors='coerce')
macro = macro.dropna(subset=['Date']).set_index('Date').sort_index()

# Create Main Dataset and Blind Dataset
df_2 = clean_and_merge(TLT, macro)
df_2.index = pd.to_datetime(df_2.index)
df_main = df_2.loc['2002-07-30':'2021-12-31'].copy()
df_blind  = df_2.loc['2021-12-31':'2025-06-30'].copy()

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

    return df

# ---------------------------------------------------------
# 4. EXECUTION
# ---------------------------------------------------------
def run_backtest(df_main, df_blind):
    # 2. Features
    print("Calculating features...")

    optimal_features = ["Sig_Vol_ROC_5", "New_Signal_Real_Yield", "New_VXN_MA_Div", "Sig_CCI", "Stability_ratio", 
                        "New_Yield_Curve_Slope", "Sig_CMF_Proxy"]
    df_full = calculate_features(df_2).replace([np.inf, -np.inf], np.nan)

    train_data = df_full.loc['2000-01-01':'2015-12-31'].dropna(subset=optimal_features+['Target','Next_Return'])
    val_data   = df_full.loc['2016-01-01':'2021-12-31'].dropna(subset=optimal_features+['Target','Next_Return'])
    blind_data = df_full.loc['2022-01-01':].dropna(subset=optimal_features+['Target','Next_Return'])

    # Check for missing features
    missing = [f for f in optimal_features if f not in train_data.columns]
    if missing:
        print(f"Error: Missing features: {missing}")
        return

    # 5. Train Model
    print(f"Training Random Forest on {len(train_data)} samples...")
    rf_params = dict(
        n_estimators=250,
        max_depth=5,
        min_samples_leaf=100,
        max_features = None,
        bootstrap=True
        # add any other RF params you're using (min_samples_split, max_features, etc.)
        )
    rf = RandomForestClassifier(**rf_params, random_state=42, n_jobs=-1)
    rf.fit(train_data[optimal_features], train_data['Target'])

    # 6. Predictions
    print("Generating predictions...")
    probs_train = rf.predict_proba(train_data[optimal_features])[:, 1]
    probs_val = rf.predict_proba(val_data[optimal_features])[:, 1]
    probs_blind = rf.predict_proba(blind_data[optimal_features])[:, 1]


    # 7. Apply Leverage Logic
    def lev_3(p):
        if p > 0.58: return 1.5
        if p > 0.55: return 1.0
        if p >0.54: return 0.5
        if p > 0.45: return 0.0
        return -1.0


    pos_train = pd.Series([lev_3(p) for p in probs_train], index=train_data.index)
    pos_val = pd.Series([lev_3(p) for p in probs_val], index=val_data.index)
    pos_blind = pd.Series([lev_3(p) for p in probs_blind], index=blind_data.index)

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
    active = pos_blind != 0
    win_rate_active = (ret_blind[active] > 0).mean()
    print(f"Win Rate (active days only): {win_rate_active:.2%}")
    print(f"Active days: {active.mean():.2%}")
    print(f"Max Drawdown: {(eq_blind / eq_blind.cummax() - 1).min():.2%}")
    print("\n" + "="*40)

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
    run_backtest(df_main, df_blind)