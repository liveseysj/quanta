import pandas as pd
import numpy as np

# --- STANDALONE QQQ STRATEGY (V11 HYBRID) ---
def get_qqq_final_v11():
    df0 = pd.read_csv('QQQ Train & Validate.csv')
    df1 = pd.read_csv('QQQ Blind Out of Sample.csv')
    df = pd.concat([df0, df1]).drop_duplicates(subset=['Time'])
    df['Time'] = pd.to_datetime(df['Time'], format='mixed')
    df = df.sort_values('Time').set_index('Time')
    df.index = df.index.normalize()  # Remove time component
    df['Ret'] = df['Latest'].pct_change()

    def get_rsi(s, p):
        delta = s.diff()
        g = delta.where(delta > 0, 0)
        l = -delta.where(delta < 0, 0)
        rs = g.rolling(p).mean() / (l.rolling(p).mean().replace(0, 1e-9))
        return 100 - (100 / (1 + rs))

    df['IBS'] = (df['Latest'] - df['Low']) / (df['High'] - df['Low']).replace(0, 1e-9)
    df['RV21'] = df['Ret'].rolling(21).std() * np.sqrt(252)
    df['SMA20'] = df['Latest'].rolling(20).mean()
    df['SMA50'] = df['Latest'].rolling(50).mean()
    df['SMA60'] = df['Latest'].rolling(60).mean()
    df['SMA200'] = df['Latest'].rolling(200).mean()
    df['ER'] = (df['Latest'] - df['Latest'].shift(21)).abs() / (df['Ret'].abs().rolling(21).sum() * df['Latest'].shift(21)).replace(0, 1e-9)

    # Signal Generation (Shifted 1 day to prevent lookahead)
    df['Sig_D'] = (1.5 / (1 + np.exp(15 * (df['IBS'] - 0.2)))).shift(1)
    df['MT_Final_E'] = (np.where(df['SMA20'] > df['SMA60'], 1.0, np.where(df['Latest'] < df['SMA200'], -1.0, 0.0)) * (0.15 / df['RV21'].replace(0, 1e-9))).shift(1)
    df['Sig_E'] = pd.Series(np.where((df['IBS'] < 0.2) & ((df['Latest'] - df['SMA20'])/df['Latest'].rolling(20).std().replace(0, 1e-9) < -1.5), 1.5, 0.0), index=df.index).shift(1)
    df['Sig_C_Connors'] = pd.Series(np.where(((get_rsi(df['Latest'], 3) + get_rsi(df['Ret'], 2) + df['Ret'].rolling(100).rank(pct=True)*100)/3) < 25, 1.5, 0.0), index=df.index).shift(1)
    df['Sig_A_Sniper'] = pd.Series(np.where((df['IBS'] < 0.2) & (df['Latest'] > df['SMA200']), 1.5, 0.0), index=df.index).shift(1)
    df['Sig_MT_ER'] = (df['ER'] * np.sign(df['Latest'].pct_change(21)) * 2.0).shift(1)
    df['Iter_T3_D'] = (np.where(df['SMA50'] > df['SMA200'], 1.0, -1.0) * (0.15 / df['RV21'].replace(0, 1e-9))).shift(1)
    df['Iter_D_E'] = pd.Series(np.where(df['SMA200'].diff(5) > 0, (np.sign(df['Latest'].pct_change(5)) + np.sign(df['Latest'].pct_change(21)) + np.sign(df['Latest'].pct_change(126)))/3 + 0.5, -0.5), index=df.index).shift(1)

    df['Sig_S_Headfake'] = pd.Series(np.where((df['Latest'] < df['SMA200']) & (get_rsi(df['Latest'], 2) > 90), -1.0, 0.0), index=df.index).shift(1)
    df['Range'] = df['High'] - df['Low']
    df['Sig_S_Ignition'] = pd.Series(np.where((df['Range'] > df['Range'].rolling(10).mean() * 2.0) & (df['Latest'] < df['Low'].shift(1)), -1.2, 0.0), index=df.index).shift(1)
    df['Z20'] = (df['Latest'] - df['SMA20']) / df['Latest'].rolling(20).std().replace(0, 1e-9)
    df['Sig_S_Blowoff'] = pd.Series(np.where(df['Z20'] > 2.5, -1.0, 0.0), index=df.index).shift(1)
    df['ER_High'] = df['ER'].rolling(10).max() > 0.6
    df['Sig_S_EffBreak'] = pd.Series(np.where(df['ER_High'] & (df['ER'] < 0.3) & (df['Latest'] < df['Latest'].rolling(10).min()), -1.5, 0.0), index=df.index).shift(1)
    df['Skew63'] = df['Ret'].rolling(63).skew()
    df['Sig_S_SkewDiv'] = pd.Series(np.where((df['Latest'] > df['Latest'].rolling(252).max() * 0.98) & (df['Skew63'] < df['Skew63'].rolling(252).quantile(0.1)), -1.0, 0.0), index=df.index).shift(1)

    mr_cols = ['Sig_D', 'Sig_E', 'Sig_C_Connors', 'Sig_A_Sniper']
    tf_cols = ['MT_Final_E', 'Iter_T3_D', 'Iter_D_E', 'Sig_MT_ER']
    short_cols = ['Sig_S_Headfake', 'Sig_S_Ignition', 'Sig_S_Blowoff', 'Sig_S_EffBreak', 'Sig_S_SkewDiv']

    for col in mr_cols + tf_cols + short_cols:
        df[col] = df[col].fillna(0)

    cluster_rets = pd.DataFrame({
        'MR': df[mr_cols].multiply(df['Ret'], axis=0).mean(axis=1),
        'TF': df[tf_cols].multiply(df['Ret'], axis=0).mean(axis=1),
        'SH': df[short_cols].multiply(df['Ret'], axis=0).mean(axis=1)
    })

    c_corr = cluster_rets.rolling(252).corr().groupby(level=0).mean().shift(1).fillna(0.5)
    w_corr = (1 - c_corr).div((1 - c_corr).sum(axis=1), axis=0).fillna(0.33)

    v11_raw = (df[mr_cols].mean(axis=1) * w_corr['MR'] +
               df[tf_cols].mean(axis=1) * w_corr['TF'] +
               df[short_cols].mean(axis=1) * w_corr['SH'])

    return (v11_raw.clip(-1.0, 1.5).fillna(0) * df['Ret']).dropna()

# --- STANDALONE SPY STRATEGY ---
def get_spy_final():
    d2 = pd.read_csv('SPY Train and Validate.csv')
    d3 = pd.read_csv('SPY Blind Out of Sample.csv')
    df = pd.concat([d2, d3]).drop_duplicates(subset=['Time'])
    df['Date'] = pd.to_datetime(df['Time'], errors='coerce')
    df = df.dropna(subset=['Date']).set_index('Date').sort_index()
    df.index = df.index.normalize()  # Remove time component
    if 'Latest' in df.columns: df.rename(columns={'Latest': 'Close'}, inplace=True)
    
    ind = pd.DataFrame(index=df.index)
    ind['SMA_200'] = df['Close'].rolling(200).mean()
    ind['SMA_252'] = df['Close'].rolling(252).mean()
    ind['Min_20_Low'] = df['Low'].rolling(20).min()
    ind['ROC_63'] = df['Close'].pct_change(63)
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/2, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/2, adjust=False).mean().replace(0, 1e-9)
    ind['RSI_2'] = 100 - (100 / (1 + gain / loss))

    gain3 = (delta.where(delta > 0, 0)).ewm(alpha=1/3, adjust=False).mean()
    loss3 = (-delta.where(delta < 0, 0)).ewm(alpha=1/3, adjust=False).mean().replace(0, 1e-9)
    ind['RSI_3'] = 100 - (100 / (1 + gain3 / loss3))
    
    vol_sma = df['Volume'].rolling(100).mean()
    ind['RVOL'] = df['Volume'] / (vol_sma.replace(0, 1e-9))
    
    log_ret = np.log(df['Close'] / df['Close'].shift(1))
    vol_20 = log_ret.rolling(20).std() * np.sqrt(252)
    ind['VV_15'] = vol_20.rolling(15).std()
    
    ind['Bull_Month'] = df.index.month.isin([4, 7, 11, 12])
    ind['Day_Of_Week'] = df.index.dayofweek 
    ind['Daily_Ret'] = df['Close'].pct_change()
    ind['Panic'] = ind['RSI_2'] < 10

    s1 = np.where(ind['Bull_Month'] | ind['Panic'], 1.5, 0.0)
    s2 = np.where(df['Close'] > ind['SMA_200'], 1.0, 0.0)
    s2 = np.where((ind['RVOL'] > 0.8) & (ind['RVOL'] < 1.2), 1.5, s2)
    s2 = np.where(ind['Panic'], 1.5, s2)
    s3 = np.where(df['Close'] > ind['SMA_252'], 1.5, 0.0)
    s3 = np.where((df['Close'] < ind['SMA_252']) & ind['Panic'], 1.5, s3)
    s4 = np.where(ind['VV_15'] < 0.05, 1.5, 0.0)
    s4 = np.where((df['Close'] < ind['SMA_200']) & (s4 == 1.5), 1.0, s4)
    s4 = np.where(ind['Bull_Month'] | ind['Panic'], 1.5, s4)
    s5 = np.where(df['Close'] > ind['SMA_252'], 1.0, 0.0)
    s5 = np.where(ind['Panic'], 1.5, s5)
    s6 = np.where((ind['Day_Of_Week'] == 0) & (ind['Daily_Ret'] < -0.005) & (ind['RSI_2'] < 30), 1.5, 0.0)
    s7 = np.where((ind['RSI_3'] < 15) | (ind['Bull_Month'] & (ind['RSI_3'] < 45)), 1.5, 0.0)
    s8 = np.where(ind['ROC_63'] > 0, 1.5, 0.0)
    
    votes = pd.DataFrame({'S1': s1, 'S2': s2, 'S3': s3, 'S4': s4, 'S5': s5, 'S6': s6, 'S7': s7, 'S8': s8}, index=df.index)
    vote_count = (votes > 0).sum(axis=1)
    base_signal = np.where(vote_count >= 5, 1.5, 0.0)
    veto_triggered = df['Close'] <= ind['Min_20_Low']
    final_exposure = np.where(veto_triggered, -0.5, base_signal)
    
    return (pd.Series(final_exposure, index=df.index).shift(1) * ind['Daily_Ret']).dropna()

# --- STANDALONE GLD STRATEGY ---
def get_gld_final():
    df = pd.read_csv('GLD.csv')
    df['Date'] = pd.to_datetime(df['Time'], errors='coerce')
    df = df.dropna(subset=['Date']).set_index('Date').sort_index()
    df.index = df.index.normalize()  # Remove time component
    if 'Latest' in df.columns: df.rename(columns={'Latest': 'Close'}, inplace=True)
    df['Return'] = df['Close'].pct_change(); TRADING_DAYS = 252
    df['SMA_50'] = df['Close'].rolling(50).mean(); df['SMA_200'] = df['Close'].rolling(200).mean(); df['Vol_21'] = df['Return'].rolling(21).std() * np.sqrt(TRADING_DAYS); df['Vol_Rank'] = df['Vol_21'].rolling(252).rank(pct=True)
    df['ATR_14'] = (df['High'] - df['Low']).rolling(14).mean(); df['Dens_Rank'] = (df['Volume'] / df['ATR_14'].replace(0, np.nan)).rolling(252).rank(pct=True); df['Vol_Z_Score'] = (df['Volume'] - df['Volume'].rolling(50).mean()) / df['Volume'].rolling(50).std()
    df['Parkinson_21'] = np.sqrt((1.0 / (4.0 * np.log(2.0))) * (np.log(df['High'] / df['Low'])**2).rolling(21).mean()) * np.sqrt(TRADING_DAYS)
    df['SMA_200_Slope'] = df['SMA_200'].diff(10); delta = df['Close'].diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean(); df['RSI'] = 100 - (100 / (1 + gain/loss.replace(0, 1e-9)))
    df['IBS'] = (df['Close'] - df['Low']) / (df['High'] - df['Low']).replace(0, 0.01)
    macd = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean(); sig = macd.ewm(span=9, adjust=False).mean()
    def calc_rsi_custom(window): d = df['Close'].diff(); g = (d.where(d > 0, 0)).rolling(window).mean(); l = (-d.where(d < 0, 0)).rolling(window).mean(); return 100 - (100 / (1 + g / l.replace(0, 0.0001)))
    df['Ult_Osc'] = (calc_rsi_custom(7) * 4 + calc_rsi_custom(14) * 2 + calc_rsi_custom(28)) / 7
    df['AD_Line'] = (((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']).replace(0, 0.01) * df['Volume']).cumsum(); df['AD_SMA_50'] = df['AD_Line'].rolling(50).mean()
    df['BD_Month'] = df.groupby([df.index.year, df.index.month]).cumcount() + 1; dates_count = df.groupby([df.index.year, df.index.month])['Close'].transform('count'); df['BD_Month_Rev'] = dates_count - df['BD_Month']
    s = pd.DataFrame(index=df.index)
    s['S5_Loose_Vol'] = np.where((df['Dens_Rank'] > 0.8) & (df['Vol_Rank'] < 0.4).rolling(5).max() > 0, np.where(df['Close'] > df['SMA_50'], 1.5, -0.5), 0.0)
    s['S21_Vol_Regime_GC'] = np.where((df['SMA_50'] > df['SMA_200']) & (df['Close'] > df['SMA_200']), np.where(df['Vol_21'] < 0.10, 1.5, np.where(df['Vol_21'] > 0.18, 0.5, 1.0)), 0.0)
    s['Trend_Filter_Low'] = np.select([(df['Vol_21'] < 0.10) & (df['Close'] < df['SMA_200']), (df['Vol_21'] < 0.10) & (df['Close'] > df['SMA_200']), (df['Vol_21'] >= 0.10) & (df['Vol_21'] < 0.18), (df['Vol_21'] >= 0.18)], [0.0, 1.5, 1.0, 0.5], default=1.0)
    s['S17_Vol_Climax'] = np.where(df['Vol_Z_Score'] > 3, -1.0 * np.sign(df['Return']), 1.0); s['S19_Stable_Spread'] = np.where(df['Parkinson_21'] < df['Vol_21'], 1.5, 0.5)
    s['S7_Mom_Scale'] = np.where(((df['BD_Month'] <= 3) | (df['BD_Month_Rev'] < 4)) & (df['Close'] > df['SMA_200']), (0.12 / df['Vol_21'].replace(0, 0.01) * np.where(df['SMA_200_Slope'] > 0, 1.2, 0.8)).clip(0, 1.5), 0.0)
    def hybrid_base(row):
        if pd.isna(row['Vol_21']): return 0.0
        if row['Vol_21'] < 0.10: return 1.5 if row['Close'] > row['SMA_50'] else 0.0
        elif row['Vol_21'] > 0.18:
            if row['RSI'] < 30: return 1.5
            elif row['RSI'] > 70: return -1.0
            else: return 0.5
        return 1.0
    s['S5_Hybrid_Base'] = df.apply(hybrid_base, axis=1)
    s['S27_MACD_Regime'] = np.where(macd > sig, np.where(df['IBS'] < 0.1, 1.5, 0.0), np.where(df['IBS'] > 0.9, -1.0, 0.0))
    s['N6_Tue_Rev'] = np.where((df.index.dayofweek == 1) & (df['Return'].shift(1) < 0), 1.5, 0.0)
    s['N5_Inside_Break'] = np.where((df['High'].shift(1) < df['High'].shift(2)) & (df['Low'].shift(1) > df['Low'].shift(2)) & (df['Close'] > df['High'].shift(1)), 1.5, 0.0)
    s['N24_Ult_Osc'] = np.where(df['Ult_Osc'] < 40, 1.5, np.where(df['Ult_Osc'] > 60, -0.5, 0.0))
    s['N37_Band_Fade'] = np.where(df['Close'] > df['SMA_50']*1.05, -0.5, np.where(df['Close'] < df['SMA_50']*0.95, 1.5, 0.0))
    s['N35_AD_Trend'] = np.where(df['AD_Line'] > df['AD_SMA_50'], 1.5, -0.5); s['S5_Parabolic_Short'] = np.where(((df['Close'] - df['SMA_50']) / df['SMA_50'] > 0.10) & (df['Return'] < 0), -1.0, 0.0)
    exposure = s.shift(1); sig_rets = exposure.multiply(df['Return'], axis=0).iloc[252:].dropna(); train_rets = sig_rets.loc[:'2019-12-31']
    inv_vol = 1.0 / (train_rets.std() * np.sqrt(252) + 1e-9); weights = inv_vol / inv_vol.sum()
    return (exposure.iloc[252:].dropna().dot(weights).clip(-1.0, 1.5) * df['Return']).dropna()

# --- PORTFOLIO EXECUTION ---
r_q = get_qqq_final_v11()
r_s = get_spy_final()
r_g = get_gld_final()

print(f"QQQ returns: {len(r_q)} rows, mean={r_q.mean():.6f}, range: {r_q.index.min()} to {r_q.index.max()}")
print(f"SPY returns: {len(r_s)} rows, mean={r_s.mean():.6f}, range: {r_s.index.min()} to {r_s.index.max()}")
print(f"GLD returns: {len(r_g)} rows, mean={r_g.mean():.6f}, range: {r_g.index.min()} to {r_g.index.max()}")

p_df = pd.DataFrame({'QQQ': r_q, 'SPY': r_s, 'GLD': r_g}).dropna()
print(f"\nCombined portfolio: {len(p_df)} rows")
if len(p_df) > 0:
    print(f"Date range: {p_df.index.min()} to {p_df.index.max()}")
else:
    print("No overlapping dates!")
    p_df_all = pd.DataFrame({'QQQ': r_q, 'SPY': r_s, 'GLD': r_g})
    print(f"Without dropna: {len(p_df_all)} rows")
    print(f"Non-null counts: QQQ={p_df_all['QQQ'].notna().sum()}, SPY={p_df_all['SPY'].notna().sum()}, GLD={p_df_all['GLD'].notna().sum()}")


W = 180
vol_roll = p_df.rolling(W).std()
v_eq = (vol_roll['QQQ'] + vol_roll['SPY']) / 2.0; v_g = vol_roll['GLD']
inv_eq = 1.0 / (v_eq + 1e-9); inv_g = 1.0 / (v_g + 1e-9); total = inv_eq + inv_g
weights = pd.DataFrame({'QQQ': (inv_eq/total)*0.5, 'SPY': (inv_eq/total)*0.5, 'GLD': inv_g/total}, index=p_df.index).shift(1).dropna()
final_ret = (weights * p_df.loc[weights.index]).sum(axis=1)

final_ret.to_csv('final_portfolio_returns.csv')

def get_stats(returns):
    if len(returns) == 0: return 0.0, 0.0, 0.0
    ann_ret = returns.mean() * 252; ann_vol = returns.std() * np.sqrt(252); sharpe = ann_ret / (ann_vol + 1e-9)
    cum = (1 + returns).cumprod(); dd = (cum / cum.cummax()) - 1; return sharpe, ann_ret, dd.min()

print(f"{'Period':<10} | {'Sharpe':<10} | {'Ann. Ret':<10} | {'Max DD':<10}")
print("-" * 50)
for name, mask in {'Train': final_ret.index < '2020-01-01', 'Valid': (final_ret.index >= '2020-01-01') & (final_ret.index < '2022-01-01'), 'Holdout': final_ret.index >= '2022-01-01'}.items():
    s, r, d = get_stats(final_ret.loc[mask])
    print(f"{name:<10} | {s:<10.2f} | {r:<10.1%} | {d:<10.1%}")
    
    import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- EQUITY CURVES ---

def equity_curve(returns):
    return (1 + returns).cumprod()

eq_qqq = equity_curve(r_q)
eq_spy = 0 # equity_curve(r_s)
eq_gld = equity_curve(r_g)
eq_port = equity_curve(final_ret)

# --- PLOT ---
plt.figure(figsize=(12, 7))
plt.plot(eq_qqq, label='QQQ Strategy')
plt.plot(eq_spy, label='SPY Strategy')
plt.plot(eq_gld, label='GLD Strategy')
plt.plot(eq_port, label='Combined Portfolio', linewidth=2)

plt.title('Equity Curves')
plt.ylabel('Growth of $1')
plt.xlabel('Date')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
