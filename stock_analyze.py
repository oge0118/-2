import streamlit as st
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import sys
from datetime import datetime, timedelta

# --- 頁面設定 ---
st.set_page_config(page_title="量化交易策略系統", layout="wide")

# --- 快取函數 ---
@st.cache_data(ttl=300) 
def load_stock_data(ticker, start_date, interval):
    try:
        # 5分鐘線限制
        if interval.endswith('m'):
            limit_date = datetime.now() - timedelta(days=59)
            # 轉換 start_date (date) 為 datetime
            start_datetime = datetime.combine(start_date, datetime.min.time())
            if start_datetime < limit_date:
                start_date = limit_date.date()
        
        df = yf.download(ticker, start=start_date, interval=interval, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        return pd.DataFrame()

# --- 策略 1: 通用波段策略 (General Swing) ---
def calculate_general_strategy(df, sell_threshold_val, use_sl, trailing_val, commission):
    # 1. 指標計算
    df['MA90'] = ta.sma(df['Close'], length=90)
    
    # MACD (12, 26, 9)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    col_macd = 'MACD_12_26_9'
    col_macdh = 'MACDh_12_26_9' 
    col_macds = 'MACDs_12_26_9' 
    
    # RSI (14)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    # KDJ (9, 3)
    kdj = df.ta.kdj(length=9, signal=3)
    df = pd.concat([df, kdj], axis=1)
    col_k = 'K_9_3'
    col_d = 'D_9_3'
    
    # MTM (10)
    df['MTM'] = df['Close'] - df['Close'].shift(10)
    
    # OSC (10, 20, 10)
    sma_short = ta.sma(df['Close'], length=10)
    sma_long = ta.sma(df['Close'], length=20)
    df['OSC'] = sma_short - sma_long
    df['OSCEMA'] = ta.ema(df['OSC'], length=10)
    
    # OBV (20)
    df['OBV'] = ta.obv(df['Close'], df['Volume'])
    df['OBVMA'] = ta.sma(df['OBV'], length=20)
    
    # BB (20, 2)
    bb = ta.bbands(df['Close'], length=20, std=2.0)
    df = pd.concat([df, bb], axis=1)
    # 動態抓取 BB 欄位
    cols = df.columns
    col_bbl = [c for c in cols if c.startswith('BBL')][0]
    col_bbm = [c for c in cols if c.startswith('BBM')][0]
    col_bbu = [c for c in cols if c.startswith('BBU')][0]
    
    df.dropna(inplace=True)
    
    # 2. 邏輯定義
    df['Signal_Trigger'] = (df[col_k] > df[col_d]) & (df[col_k].shift(1) < df[col_d].shift(1))
    df['Trend_OK'] = ((df[col_macd] > df[col_macds]) & (df[col_macdh] > 0)) | (df['OSC'] > 0)
    df['Volume_OK'] = df['OBV'] > df['OBVMA']
    cond_loc_low = df['Close'] < df[col_bbm]
    cond_loc_high_mom = (df['Close'] >= df[col_bbm]) & (df['MTM'] > 0)
    df['Location_OK'] = cond_loc_low | cond_loc_high_mom
    df['Condition_Safe'] = df['RSI'] < 85
    
    # 3. 買入訊號
    df['Raw_Buy'] = df['Signal_Trigger'] & df['Condition_Safe'] & (df['Trend_OK'] | df['Volume_OK'] | df['Location_OK'])
    
    # 4. 賣出訊號
    is_kdj_dead = (df[col_k] < df[col_d]) & (df[col_k].shift(1) > df[col_d].shift(1))
    cond_sell_1 = is_kdj_dead & (df[col_k] > 80)
    cond_sell_2 = df['OSC'] < 0
    cond_sell_3 = df['RSI'] > 85
    sell_count = cond_sell_1.astype(int) + cond_sell_2.astype(int) + cond_sell_3.astype(int)
    df['Raw_Sell'] = sell_count >= sell_threshold_val

    # 5. 回測執行 (共用邏輯)
    return run_backtest_engine(df, use_sl, trailing_val, commission, mode="General")

# --- 策略 2: 短線狙擊策略 (Short-Term) ---
def calculate_short_term_strategy(df, short_mode, callback, max_days, profit_limit, commission):
    # 1. 基礎指標
    # BB (20, 2)
    bb = ta.bbands(df['Close'], length=20, std=2.0)
    df = pd.concat([df, bb], axis=1)
    cols = df.columns
    col_bbu = [c for c in cols if c.startswith('BBU')][0]
    col_bbm = [c for c in cols if c.startswith('BBM')][0]
    col_bbl = [c for c in cols if c.startswith('BBL')][0]
    
    # Vol MA5
    df['Vol_MA5'] = ta.sma(df['Volume'], length=5)
    
    # MACD
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    col_macd = 'MACD_12_26_9'
    col_macds = 'MACDs_12_26_9'
    
    # KDJ
    kdj = df.ta.kdj(length=9, signal=3)
    df = pd.concat([df, kdj], axis=1)
    col_k = 'K_9_3'
    col_d = 'D_9_3'
    
    # RSI
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    df.dropna(inplace=True)
    
    # 2. 進場訊號
    df['Raw_Buy'] = False
    
    if "模式 A" in short_mode: # 突破
        cond_break = df['Close'] > df[col_bbu]
        cond_vol = df['Volume'] > (2 * df['Vol_MA5'])
        cond_macd = df[col_macd] > df[col_macds]
        cond_kdj = df[col_k] > df[col_d]
        df['Raw_Buy'] = cond_break & cond_vol & cond_macd & cond_kdj
    else: # 抄底
        cond_low = df['Close'] < df[col_bbl]
        cond_os = (df['RSI'] < 20) | (df[col_k] < 15)
        cond_rev = (df['Close'] > df['Open']) & (df[col_k] > df[col_d])
        df['Raw_Buy'] = cond_low & cond_os & cond_rev
        
    # 3. 賣出訊號 (這裡設為 False，完全由回測引擎的停損停利控制)
    df['Raw_Sell'] = False 
    
    # 4. 回測執行
    # 傳入特殊參數給引擎使用
    return run_backtest_engine(df, True, callback, commission, mode="Short", max_days=max_days, time_profit=profit_limit)

# --- 共用回測引擎 (Backtest Engine) ---
def run_backtest_engine(df, use_stop, stop_val, commission, mode="General", max_days=5, time_profit=0.01):
    # 初始化變數
    buy_signals = []
    sell_signals = []
    buy_reasons = []
    sell_reasons = []
    sell_profits = []
    
    holding = False
    cash = 100000.0
    position_size = 0.0
    asset_history = []
    
    entry_price = 0.0
    highest_price = 0.0
    entry_idx = 0
    
    raw_buys = df['Raw_Buy'].values
    raw_sells = df['Raw_Sell'].values
    closes = df['Close'].values
    opens = df['Open'].values
    lows = df['Low'].values
    dates = df.index
    
    trade_log = []
    
    for i in range(len(df)):
        price = closes[i]
        low_price = lows[i]
        open_price = opens[i]
        
        curr_buy = 0
        curr_sell = 0
        reason_buy = None
        reason_sell = None
        curr_profit = 0.0
        
        # 買入訊號標記 (即使持倉也標記)
        if raw_buys[i]:
            curr_buy = 1
            reason_buy = "訊號觸發"
            
        if not holding:
            if raw_buys[i]:
                holding = True
                entry_price = price
                entry_idx = i
                highest_price = price
                
                # 買入 (扣成本)
                cost = cash * commission
                position_size = (cash - cost) / price
                cash = 0
        else:
            # 持倉管理
            if price > highest_price:
                highest_price = price
                
            is_exit = False
            exit_price = price
            exit_reason = ""
            
            # --- 策略分歧 ---
            if mode == "General":
                # 通用策略: 移動停損 + 技術指標賣出
                trailing_price = highest_price * (1 - stop_val) if use_stop else 0
                is_trailing = use_stop and (low_price <= trailing_price)
                is_indicator = raw_sells[i] and (price > entry_price) # 賺錢才賣
                
                if is_trailing:
                    is_exit = True
                    # 模擬觸價
                    exit_price = open_price if open_price < trailing_price else trailing_price
                    p_pct = (exit_price - entry_price)/entry_price
                    if p_pct > 0: exit_reason = f"移動停利 ({p_pct*100:.1f}%)"
                    else: exit_reason = f"移動停損 ({p_pct*100:.1f}%)"
                elif is_indicator:
                    is_exit = True
                    exit_price = price
                    exit_reason = "指標轉弱獲利"
                    
            elif mode == "Short":
                # 短線策略: 移動停利 + 時間停損
                trailing_price = highest_price * (1 - stop_val) # stop_val 這裡是 callback
                is_trailing = low_price <= trailing_price
                
                # 時間停損計算
                # 簡單計算 K 線根數差
                bars_held = i - entry_idx
                # 轉換為概略天數 (如果是日線=天數, 分鐘線=根數/此處簡化邏輯)
                # 假設 max_days 對應 interval 的 bar 數更合理，但這裡先用 index 差
                # 如果是日線，index 差就是交易日數
                
                curr_pnl = (price - entry_price) / entry_price
                is_time = (bars_held >= max_days) and (curr_pnl < time_profit)
                
                if is_trailing:
                    is_exit = True
                    exit_price = open_price if open_price < trailing_price else trailing_price
                    exit_reason = "移動停利出場"
                elif is_time:
                    is_exit = True
                    exit_price = price
                    exit_reason = f"時間停損 ({bars_held}天未達標)"

            if is_exit:
                curr_sell = 1
                holding = False
                reason_sell = exit_reason
                
                gross_val = position_size * exit_price
                fee = gross_val * commission
                cash = gross_val - fee
                position_size = 0
                
                pnl = (exit_price - entry_price) / entry_price
                trade_log.append(pnl)
                curr_profit = pnl
                
                entry_price = 0
                highest_price = 0
        
        buy_signals.append(curr_buy)
        sell_signals.append(curr_sell)
        buy_reasons.append(reason_buy)
        sell_reasons.append(reason_sell)
        sell_profits.append(curr_profit)
        
        # 資產計算
        curr_val = (position_size * price) if holding else cash
        asset_history.append(curr_val)
        
    df['Buy_Signal'] = buy_signals
    df['Sell_Signal'] = sell_signals
    df['Buy_Reason'] = buy_reasons
    df['Sell_Reason'] = sell_reasons
    df['Sell_Profit'] = sell_profits
    df['Total_Asset'] = asset_history
    
    # 績效計算
    metrics = {}
    if len(trade_log) > 0:
        trades = np.array(trade_log)
        wins = trades[trades > 0]
        losses = trades[trades <= 0]
        win_rate = len(wins) / len(trades) * 100
        pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 else float('inf')
        
        asset_s = pd.Series(asset_history)
        mdd = ((asset_s.cummax() - asset_s) / asset_s.cummax()).max() * 100
        
        metrics = {
            "win_rate": win_rate, "profit_factor": pf, "max_drawdown": mdd, "total_trades": len(trades)
        }
    else:
        metrics = {"win_rate": 0, "profit_factor": 0, "max_drawdown": 0, "total_trades": 0}
        
    return df, metrics

# --- 主程式介面 ---
# 側邊欄輸入
st.sidebar.subheader("📋 策略選擇")
strategy_type = st.sidebar.selectbox(
    "選擇策略模組", 
    ["通用波段策略 (General Swing)", "短線狙擊策略 (Short-Term)"]
)

ticker_input = st.sidebar.text_input("輸入股票代碼", value="") 
st.sidebar.caption("範例: QQQ, NVDA, 2330")
ticker = ticker_input.strip().upper()
if ticker.isdigit(): ticker = f"{ticker}.TW"

interval_option = st.sidebar.selectbox("K 線週期", ["日線 (1 Day)", "5 分鐘 (5 Minutes)"], index=0)
interval_map = {"日線 (1 Day)": "1d", "5 分鐘 (5 Minutes)": "5m"}
interval = interval_map[interval_option]

# 根據策略顯示不同參數
if strategy_type == "通用波段策略 (General Swing)":
    st.sidebar.markdown("---")
    st.sidebar.write("🔧 **波段參數設定**")
    with st.sidebar.expander("技術指標細節 (MACD/RSI/BB...)", expanded=False):
        # 這裡簡化，實際可放回所有原本的指標參數輸入
        st.write("使用預設參數 (MACD 12/26/9, RSI 14, BB 20/2)")
        
    sell_thresh = st.sidebar.number_input("賣出訊號門檻 (3選幾)", 1, 3, 1)
    use_sl = st.sidebar.checkbox("啟用停損 (Trailing Stop)", True)
    if use_sl:
        trailing_stop = st.sidebar.number_input("移動停損 (%)", 5.0, 50.0, 15.0, step=0.5) / 100.0
    else:
        trailing_stop = None
    
    # 短線參數設為 None
    short_mode = None
    callback_rate = None
    max_days = None
    profit_limit = None

else: # 短線狙擊
    st.sidebar.markdown("---")
    st.sidebar.write("🚀 **短線參數設定**")
    short_mode = st.sidebar.radio("模式選擇", ["模式 A: 突破追價", "模式 B: 乖離抄底"])
    callback_rate = st.sidebar.number_input("移動停利回檔 (%)", 0.5, 10.0, 3.0, step=0.5) / 100.0
    max_days = st.sidebar.number_input("最大耐心天數 (Bars)", 1, 100, 5)
    profit_limit = st.sidebar.number_input("時間到期獲利門檻 (%)", 0.0, 10.0, 1.0, step=0.5) / 100.0
    
    # 通用參數設為 None
    sell_thresh = None
    use_sl = None
    trailing_stop = None

commission_rate = st.sidebar.number_input("手續費率 (%)", 0.0, 1.0, 0.1425) / 100.0

if interval == "5m":
    d_start = datetime.now() - timedelta(days=5)
    min_d = datetime.now() - timedelta(days=59)
else:
    d_start = datetime(2023, 1, 1)
    min_d = datetime(2000, 1, 1)
    
start_date = st.sidebar.date_input("開始日期", d_start, min_value=min_d, max_value=datetime.now())

if st.sidebar.button("開始回測", type="primary"):
    if not ticker:
        st.warning("請輸入代碼")
        st.stop()
    
    with st.spinner("策略運算中..."):
        df = load_stock_data(ticker, start_date, interval)
        
        if df.empty:
            st.error("無數據")
        else:
            # 根據選擇呼叫不同函數
            if strategy_type == "通用波段策略 (General Swing)":
                df, metrics = calculate_general_strategy(df, sell_thresh, use_sl, trailing_stop, commission_rate)
            else:
                df, metrics = calculate_short_term_strategy(df, short_mode, callback_rate, max_days, profit_limit, commission_rate)
            
            # --- 繪圖與結果 (共用) ---
            # 處理 Index 字串化 (Category Axis)
            fmt = '%Y-%m-%d %H:%M' if interval == "5m" else '%Y-%m-%d'
            df.index_str = df.index.strftime(fmt)
            
            curr = df.iloc[-1]
            ret = (curr['Total_Asset'] - 100000) / 100000 * 100
            bh_ret = (curr['Close'] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100
            
            st.markdown(f"### 📊 {ticker} 回測結果 ({strategy_type})")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("最新收盤", f"{curr['Close']:.2f}")
            col2.metric("策略報酬", f"{ret:.2f}%", f"資產: ${curr['Total_Asset']:.0f}")
            col3.metric("買入持有", f"{bh_ret:.2f}%")
            col4.metric("交易次數", f"{metrics['total_trades']} (勝率 {metrics['win_rate']:.1f}%)")
            
            st.subheader("📈 資產曲線")
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=df.index_str, y=df['Total_Asset'], name='策略', line=dict(color='red')))
            fig_eq.add_trace(go.Scatter(x=df.index_str, y=df['Close']/df['Close'].iloc[0]*100000, name='Buy&Hold', line=dict(color='gray', dash='dash')))
            
            # 自動範圍
            lookback = 300 if interval == '5m' else 250
            start_idx = max(0, len(df)-lookback)
            end_idx = len(df)-1
            
            fig_eq.update_xaxes(type='category', range=[start_idx, end_idx], nticks=10)
            st.plotly_chart(fig_eq, use_container_width=True)
            
            st.subheader("🕯️ K線交易圖")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
            
            # K線
            fig.add_trace(go.Candlestick(x=df.index_str, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
            
            # 布林通道 (共用)
            # 需重新計算或從 df 抓取正確欄位 (因為不同策略欄位名可能不同，這裡重新抓取 BBU/BBL)
            cols = df.columns
            try:
                bbu = [c for c in cols if 'BBU' in c][0]
                bbl = [c for c in cols if 'BBL' in c][0]
                fig.add_trace(go.Scatter(x=df.index_str, y=df[bbu], line=dict(color='gray', width=1, dash='dot'), name='Upper'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index_str, y=df[bbl], line=dict(color='gray', width=1, dash='dot'), name='Lower'), row=1, col=1)
            except:
                pass # 如果該策略沒算 BB 就不畫
                
            # 買賣點
            buys = df[df['Buy_Signal']==1]
            sells = df[df['Sell_Signal']==1]
            sells_win = sells[sells['Sell_Profit']>0]
            sells_loss = sells[sells['Sell_Profit']<=0]
            
            fig.add_trace(go.Scatter(x=df.index_str[df['Buy_Signal']==1], y=buys['Low']*0.99, mode='markers', marker=dict(symbol='triangle-up', size=10, color='red'), name='Buy', hovertext=buys['Buy_Reason']), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index_str[(df['Sell_Signal']==1) & (df['Sell_Profit']>0)], y=sells_win['High']*1.01, mode='markers', marker=dict(symbol='triangle-down', size=10, color='orange'), name='Sell(Win)', hovertext=sells_win['Sell_Reason']), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index_str[(df['Sell_Signal']==1) & (df['Sell_Profit']<=0)], y=sells_loss['High']*1.01, mode='markers', marker=dict(symbol='triangle-down', size=10, color='green'), name='Sell(Loss)', hovertext=sells_loss['Sell_Reason']), row=1, col=1)

            # 交易量
            colors = ['red' if c>=o else 'green' for c,o in zip(df['Close'], df['Open'])]
            fig.add_trace(go.Bar(x=df.index_str, y=df['Volume'], marker_color=colors, name='Volume'), row=2, col=1)
            
            # 視野設定
            # 自動計算 Y 軸範圍
            df_view = df.iloc[start_idx:]
            ymin = df_view['Low'].min() * 0.95
            ymax = df_view['High'].max() * 1.05
            
            fig.update_xaxes(type='category', range=[start_idx, end_idx], nticks=10)
            fig.update_yaxes(range=[ymin, ymax], side='right', row=1, col=1)
            fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_dark", dragmode='pan', hovermode='x unified')
            
            st.plotly_chart(fig, use_container_width=True)