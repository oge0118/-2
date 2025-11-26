import streamlit as st
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np  # 新增 numpy 用於計算標準差
import sys
from datetime import datetime, timedelta

# --- 頁面設定 ---
st.set_page_config(page_title="量化交易策略系統", layout="wide")

# --- 快取函數 (效能優化) ---
@st.cache_data(ttl=3600) 
def load_stock_data(ticker, start_date):
    try:
        df = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        return pd.DataFrame()

# --- 側邊欄：參數設定 ---
st.sidebar.header("📊 參數設定 (Parameters)")

ticker_input = st.sidebar.text_input("輸入股票代碼", value="") 
st.sidebar.caption("範例: QQQ, VOO, NVDA, 2330 (台股輸入數字即可)")

# 自動判斷邏輯
ticker = ticker_input.strip().upper()
if ticker.isdigit():
    # 如果輸入全是數字，預設為台股，自動加上 .TW
    ticker = f"{ticker}.TW"

run_button = st.sidebar.button("開始策略回測", type="primary")

with st.sidebar.expander("🔧 技術指標參數", expanded=False):
    st.write("**MACD**")
    macd_fast = st.number_input("Fast Period", value=12)
    macd_slow = st.number_input("Slow Period", value=26)
    macd_signal = st.number_input("Signal Period", value=9)
    
    st.write("**RSI**")
    rsi_period = st.number_input("RSI Period", value=14)
    rsi_safe_limit = st.number_input("RSI Safe Limit (<)", value=85, help="買入時 RSI 不能超過此值")
    rsi_exit_limit = st.number_input("RSI Exit Limit (>)", value=85, help="賣出時 RSI 超過此值強制出場")
    
    st.write("**KDJ**")
    kdj_period = st.number_input("KDJ Period", value=9)
    kdj_signal = st.number_input("Signal Period", value=3)
    kdj_high_level = st.number_input("KDJ High Level (>)", value=80, help="賣出時 KDJ 死叉需高於此值")
    
    st.write("**MTM (動量)**")
    mtm_n = st.number_input("MTM Period (N)", value=10)
    mtm_ma = st.number_input("MTMMA Period", value=10)
    
    st.write("**OSC (震盪)**")
    osc_short = st.number_input("OSC Short MA", value=10)
    osc_long = st.number_input("OSC Long MA", value=20)
    osc_ema_len = st.number_input("OSC EMA Period", value=10)
    
    st.write("**OBV (能量潮)**")
    obv_ma_len = st.number_input("OBV MA Period", value=20)
    
    st.write("**Bollinger Bands (布林通道)**")
    bb_len = st.number_input("BB Period", value=20)
    bb_std = st.number_input("BB StdDev", value=2.0)

st.sidebar.subheader("⚙️ 賣出策略設定")
sell_threshold = st.sidebar.number_input("賣出訊號觸發門檻 (3選幾)", min_value=1, max_value=3, value=1, help="滿足幾項賣出條件才觸發賣出？(KDJ死叉/OSC轉弱/RSI過熱)")

# 修改: 僅保留移動停損，移除硬性停損
use_stop_loss = st.sidebar.checkbox("啟用停損機制 (Stop Loss)", value=True, help="若關閉，則只會依據技術指標賣出(且需獲利)，可能會造成深度套牢。")

if use_stop_loss:
    # 只保留移動停損
    trailing_stop_pct = st.sidebar.number_input("移動停損比例 (%)", value=15.0, step=0.5, help="從波段最高價回落超過此比例時賣出 (同時作為停損與獲利保護)") / 100.0
else:
    trailing_stop_pct = None

# 新增: 交易成本設定
st.sidebar.subheader("💸 交易成本設定")
commission_rate = st.sidebar.number_input("單邊手續費率 (%)", value=0.1425, step=0.01, help="每次買入或賣出扣除的百分比成本 (台股約 0.1425%, 美股可設為 0)") / 100.0

# 修改: 移除了「圖表顯示設定」區塊 (高度與對數座標)

# 修改: 明確設定 min_value 和 max_value，解決無法選擇全部時間的問題
start_date = st.sidebar.date_input(
    "開始日期", 
    value=datetime(2000, 1, 1), 
    min_value=datetime(1970, 1, 1), 
    max_value=datetime.now()
)

# --- 核心邏輯函數 ---
def calculate_strategy(df, sell_threshold_val, use_sl, trailing_val, commission): # 移除了 sl_val (硬性停損參數)
    # 1. 計算技術指標
    df['MA5'] = ta.sma(df['Close'], length=5)
    df['MA90'] = ta.sma(df['Close'], length=90)

    df.ta.macd(fast=macd_fast, slow=macd_slow, signal=macd_signal, append=True)
    col_macd = f'MACD_{macd_fast}_{macd_slow}_{macd_signal}'
    col_macdh = f'MACDh_{macd_fast}_{macd_slow}_{macd_signal}' 
    col_macds = f'MACDs_{macd_fast}_{macd_slow}_{macd_signal}' 
    
    df['RSI'] = ta.rsi(df['Close'], length=rsi_period)
    
    kdj = df.ta.kdj(length=kdj_period, signal=kdj_signal)
    df = pd.concat([df, kdj], axis=1)
    col_k = f'K_{kdj_period}_{kdj_signal}'
    col_d = f'D_{kdj_period}_{kdj_signal}'
    
    df['MTM'] = df['Close'] - df['Close'].shift(mtm_n)
    
    sma_short = ta.sma(df['Close'], length=osc_short)
    sma_long = ta.sma(df['Close'], length=osc_long)
    df['OSC'] = sma_short - sma_long
    df['OSCEMA'] = ta.ema(df['OSC'], length=osc_ema_len)
    
    df['OBV'] = ta.obv(df['Close'], df['Volume'])
    df['OBVMA'] = ta.sma(df['OBV'], length=obv_ma_len)
    
    bb = ta.bbands(df['Close'], length=bb_len, std=bb_std)
    df = pd.concat([df, bb], axis=1)
    col_bb_lower = [c for c in bb.columns if c.startswith('BBL')][0]
    col_bb_mid = [c for c in bb.columns if c.startswith('BBM')][0]
    col_bb_upper = [c for c in bb.columns if c.startswith('BBU')][0]
    
    df.dropna(inplace=True)
    
    # --- 關鍵修改：將 Index 轉為字串，改用 Category Axis ---
    df.index = df.index.strftime('%Y-%m-%d')
    
    # 2. 定義邏輯狀態
    df['Signal_Trigger'] = (df[col_k] > df[col_d]) & (df[col_k].shift(1) < df[col_d].shift(1))
    
    df['Trend_OK'] = ((df[col_macd] > df[col_macds]) & (df[col_macdh] > 0)) | (df['OSC'] > 0)
    df['Volume_OK'] = df['OBV'] > df['OBVMA']
    
    cond_loc_low = df['Close'] < df[col_bb_mid]
    cond_loc_high_mom = (df['Close'] >= df[col_bb_mid]) & (df['MTM'] > 0)
    df['Location_OK'] = cond_loc_low | cond_loc_high_mom
    
    df['Condition_Safe'] = df['RSI'] < rsi_safe_limit
    
    # 3. 原始訊號計算
    df['Raw_Buy'] = (
        df['Signal_Trigger'] & 
        df['Condition_Safe'] & 
        (df['Trend_OK'] | df['Volume_OK'] | df['Location_OK'])
    )
    
    is_kdj_dead = (df[col_k] < df[col_d]) & (df[col_k].shift(1) > df[col_d].shift(1))
    cond_sell_1 = is_kdj_dead & (df[col_k] > kdj_high_level)
    cond_sell_2 = df['OSC'] < 0
    cond_sell_3 = df['RSI'] > rsi_exit_limit
    
    sell_condition_count = (
        cond_sell_1.astype(int) + 
        cond_sell_2.astype(int) + 
        cond_sell_3.astype(int)
    )
    
    df['Raw_Sell'] = sell_condition_count >= sell_threshold_val
    
    # 4. 訊號過濾、資金回測與原因記錄
    buy_signals = []
    sell_signals = []
    buy_reasons = [] 
    sell_reasons = []
    sell_profits = [] 
    holding = False 
    
    initial_capital = 100000.0
    cash = initial_capital
    position_size = 0.0 
    asset_history = [] 
    entry_price = 0.0 
    last_entry_price_record = 0.0
    
    # 新增: 追蹤持倉期間最高價 (for Trailing Stop)
    highest_price_record = 0.0
    
    trade_log = [] 
    
    raw_buy_list = df['Raw_Buy'].values
    raw_sell_list = df['Raw_Sell'].values
    close_prices = df['Close'].values
    low_prices = df['Low'].values 
    open_prices = df['Open'].values 
    
    trend_ok_vals = df['Trend_OK'].values
    vol_ok_vals = df['Volume_OK'].values
    loc_ok_vals = df['Location_OK'].values
    
    sell_1_vals = cond_sell_1.values
    sell_2_vals = cond_sell_2.values
    sell_3_vals = cond_sell_3.values
    
    for i in range(len(df)):
        is_buy_raw = raw_buy_list[i]
        is_sell_raw = raw_sell_list[i]
        current_price = close_prices[i]
        current_low = low_prices[i]
        current_open = open_prices[i]
        
        current_buy_signal = 0
        current_sell_signal = 0
        current_buy_reason = None
        current_sell_reason = None
        current_sell_profit = 0.0 
        
        if is_buy_raw:
            current_buy_signal = 1
            reasons = []
            if trend_ok_vals[i]: reasons.append("趨勢")
            if vol_ok_vals[i]: reasons.append("量能")
            if loc_ok_vals[i]: reasons.append("位置")
            current_buy_reason = f"KDJ金叉 + ({', '.join(reasons)})"
        
        if not holding:
            if is_buy_raw:
                holding = True
                entry_price = current_price
                last_entry_price_record = entry_price
                highest_price_record = current_price # 初始化最高價
                
                # 計算手續費
                cost = cash * commission
                position_val = cash - cost
                
                position_size = position_val / current_price
                cash = 0
        else:
            # 持倉中，更新最高價
            if current_price > highest_price_record:
                highest_price_record = current_price
            
            if is_buy_raw:
                last_entry_price_record = current_price 
            
            is_trailing_stop = False
            
            # 如果啟用了停損機制，才計算移動停損
            if use_sl and trailing_val is not None:
                # 計算移動停損價格 (從最高點回落)
                trailing_stop_price = highest_price_record * (1 - trailing_val)
                # 檢查是否觸發 (使用最低價)
                is_trailing_stop = current_low <= trailing_stop_price
            
            # 3. 檢查是否觸發技術指標賣訊 (且有獲利)
            is_take_profit = is_sell_raw and (current_price > entry_price)
            
            if is_trailing_stop or is_take_profit:
                current_sell_signal = 1
                holding = False
                
                # 決定賣出價格與原因
                sell_price = current_price
                reason_str = ""
                
                if is_trailing_stop:
                    # 移動停損賣出
                    sell_price = current_open if current_open < trailing_stop_price else trailing_stop_price
                    # 移動停損通常是獲利的，但也可能虧損
                    if sell_price >= entry_price:
                        reason_str = f"移動停損 (獲利回吐 {(trailing_val*100):.1f}%)"
                    else:
                        reason_str = f"移動停損 (虧損出場 {(trailing_val*100):.1f}%)"
                else:
                    # 技術指標賣出
                    reasons = []
                    if sell_1_vals[i]: reasons.append("KDJ死叉")
                    if sell_2_vals[i]: reasons.append("OSC轉弱")
                    if sell_3_vals[i]: reasons.append("RSI過熱")
                    reason_str = f"指標轉弱: {', '.join(reasons)}"
                
                current_sell_reason = reason_str
                
                # 計算手續費
                gross_revenue = position_size * sell_price
                fee = gross_revenue * commission
                cash = gross_revenue - fee
                
                pnl_pct = (sell_price - entry_price) / entry_price
                trade_log.append(pnl_pct)
                current_sell_profit = pnl_pct 
                
                position_size = 0
                entry_price = 0
                highest_price_record = 0 # 重置最高價
        
        buy_signals.append(current_buy_signal)
        sell_signals.append(current_sell_signal)
        buy_reasons.append(current_buy_reason)
        sell_reasons.append(current_sell_reason)
        sell_profits.append(current_sell_profit) 
        
        if holding:
            current_asset = position_size * current_price
        else:
            current_asset = cash
            
        asset_history.append(current_asset)
            
    df['Buy_Signal'] = buy_signals
    df['Sell_Signal'] = sell_signals
    df['Buy_Reason'] = buy_reasons
    df['Sell_Reason'] = sell_reasons
    df['Sell_Profit'] = sell_profits 
    df['Total_Asset'] = asset_history 
    
    # 計算 Buy & Hold 資產曲線
    initial_close = df['Close'].iloc[0]
    df['Buy_Hold_Asset'] = initial_capital * (df['Close'] / initial_close)
    
    metrics = {}
    if len(trade_log) > 0:
        trade_log = np.array(trade_log)
        wins = trade_log[trade_log > 0]
        losses = trade_log[trade_log <= 0]
        
        win_rate = len(wins) / len(trade_log) * 100
        
        gross_profit = wins.sum() if len(wins) > 0 else 0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 1e-9 
        profit_factor = gross_profit / gross_loss
        
        asset_series = pd.Series(asset_history)
        rolling_max = asset_series.cummax()
        drawdown = (asset_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        daily_returns = asset_series.pct_change().dropna()
        if daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() - (0.02/252)) / daily_returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0
            
        metrics = {
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "total_trades": len(trade_log)
        }
    else:
        metrics = {
            "win_rate": 0, "profit_factor": 0, "max_drawdown": 0, "sharpe_ratio": 0, "total_trades": 0
        }

    return df, col_k, col_d, last_entry_price_record, col_bb_upper, col_bb_mid, col_bb_lower, col_macd, col_macdh, col_macds, metrics

# --- 主程式執行 ---
if run_button:
    if not ticker:
        st.warning("⚠️ 請輸入股票代碼！")
        st.stop()
        
    with st.spinner(f'正在運算 {ticker} 的交易策略...'):
        try:
            stock_name = ticker
            try:
                stock_info = yf.Ticker(ticker).info
                stock_name = stock_info.get('longName', stock_info.get('shortName', ticker))
            except:
                pass

            df = load_stock_data(ticker, start_date)

            if df.empty:
                st.error(f"找不到數據 ({ticker})")
            else:
                # 修改: 移除 stop_loss_pct 傳入
                df, col_k, col_d, last_entry_price, col_bbu, col_bbm, col_bbl, col_macd, col_macdh, col_macds, metrics = calculate_strategy(df, sell_threshold, use_stop_loss, trailing_stop_pct, commission_rate)
                
                curr = df.iloc[-1]
                initial_capital = 100000.0
                
                # 策略報酬
                strategy_return = (curr['Total_Asset'] - initial_capital) / initial_capital * 100
                
                # Buy & Hold 報酬
                bh_return = (curr['Buy_Hold_Asset'] - initial_capital) / initial_capital * 100
                
                # 超額報酬 (Alpha)
                alpha = strategy_return - bh_return
                
                st.markdown("---")
                st.markdown(f"### 🪙 {ticker} - {stock_name}")
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("最新收盤價", f"${curr['Close']:.2f}")
                
                last_buy_idx = df[df['Buy_Signal']==1].index.max()
                last_sell_idx = df[df['Sell_Signal']==1].index.max()
                is_holding = False
                if pd.notna(last_buy_idx):
                    buy_loc = df.index.get_loc(last_buy_idx)
                    if pd.isna(last_sell_idx):
                        is_holding = True
                    else:
                        sell_loc = df.index.get_loc(last_sell_idx)
                        is_holding = buy_loc > sell_loc

                if is_holding:
                    if last_entry_price > 0:
                        unrealized_pnl = (curr['Close'] - last_entry_price) / last_entry_price * 100
                        c2.metric("目前倉位", "🟢 持倉中", f"{unrealized_pnl:.2f}% (最近買入: {last_entry_price:.2f})")
                    else:
                        c2.metric("目前倉位", "🟢 持倉中", "成本計算中")
                else:
                    c2.metric("目前倉位", "⚪ 空手", "等待買點")

                # 顯示策略總回報與 Buy & Hold 對比
                c3.metric("策略總報酬", f"{strategy_return:.2f}%", f"總資產: ${curr['Total_Asset']:.0f}")
                c4.metric("買入持有報酬", f"{bh_return:.2f}%", delta=f"Alpha: {alpha:.2f}%")

                # 進階績效
                st.markdown("#### 📊 策略進階績效")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("交易次數", f"{metrics['total_trades']} 次")
                m2.metric("勝率 (Win Rate)", f"{metrics['win_rate']:.1f}%")
                m3.metric("獲利因子 (PF)", f"{metrics['profit_factor']:.2f}")
                m4.metric("最大回撤 (MDD)", f"{metrics['max_drawdown']:.2f}%", delta_color="inverse")
                m5.metric("夏普比率 (Sharpe)", f"{metrics['sharpe_ratio']:.2f}")
                st.markdown("---")

                # --- 新增圖表區塊：資產曲線 ---
                st.subheader("📈 資產增長曲線 (Equity Curve)")
                fig_equity = go.Figure()
                fig_equity.add_trace(go.Scatter(x=df.index, y=df['Total_Asset'], mode='lines', name='策略資產', line=dict(color='red', width=2)))
                fig_equity.add_trace(go.Scatter(x=df.index, y=df['Buy_Hold_Asset'], mode='lines', name='買入持有', line=dict(color='gray', width=1, dash='dash')))
                
                # 同樣使用 Category Axis 確保對齊
                total_bars = len(df)
                show_bars = 250
                range_start_idx = max(0, total_bars - show_bars)
                range_end_idx = total_bars - 1
                
                fig_equity.update_layout(
                    height=400, 
                    xaxis_title="日期", 
                    yaxis_title="總資產 ($)",
                    template="plotly_white",
                    xaxis=dict(
                        type='category', 
                        range=[range_start_idx, range_end_idx],
                        tickmode='auto', nticks=10
                    ),
                    hovermode='x unified'
                )
                st.plotly_chart(fig_equity, use_container_width=True)

                # 主圖表
                st.subheader(f"📊 {stock_name} ({ticker}) 策略訊號圖")
                
                buy_points = df[df['Buy_Signal'] == 1]
                
                sell_points_win = df[(df['Sell_Signal'] == 1) & (df['Sell_Profit'] > 0)]
                sell_points_loss = df[(df['Sell_Signal'] == 1) & (df['Sell_Profit'] <= 0)]
                
                fig = make_subplots(rows=5, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.04, 
                                    row_heights=[0.4, 0.15, 0.15, 0.15, 0.15],
                                    subplot_titles=("價格與訊號 (含BB通道)", "交易量", "MACD", "KDJ", "RSI"))
                
                fig.update_annotations(font_size=10)

                # 1. Price & BB & Signals
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df[col_bbu].values, line=dict(color='gray', width=1, dash='dash'), name='BB Upper'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df[col_bbm].values, line=dict(color='cyan', width=1), name='BB Mid'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df[col_bbl].values, line=dict(color='gray', width=1, dash='dash'), name='BB Lower', fill='tonexty', fillcolor='rgba(0,200,200,0.05)'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MA5'].values, line=dict(color='yellow', width=1), name='MA5'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MA90'].values, line=dict(color='purple', width=1), name='MA90'), row=1, col=1)

                fig.add_trace(go.Scatter(
                    x=buy_points.index, y=buy_points['Low']*0.98, 
                    mode='markers', marker=dict(symbol='triangle-up', size=12, color='blue'), 
                    name='Buy',
                    hovertext=buy_points['Buy_Reason'], 
                    hoverinfo='x+y+text'
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=sell_points_win.index, y=sell_points_win['High']*1.02, 
                    mode='markers', marker=dict(symbol='triangle-down', size=10, color='orange'), 
                    name='Sell (Win)',
                    hovertext=sell_points_win['Sell_Reason'], 
                    hoverinfo='x+y+text'
                ), row=1, col=1)

                fig.add_trace(go.Scatter(
                    x=sell_points_loss.index, y=sell_points_loss['High']*1.02, 
                    mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'), 
                    name='Sell (Loss)',
                    hovertext=sell_points_loss['Sell_Reason'], 
                    hoverinfo='x+y+text'
                ), row=1, col=1)

                # 2. Volume
                vol_colors = ['green' if c >= o else 'red' for c, o in zip(df['Close'], df['Open'])]
                fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=vol_colors, name='Volume'), row=2, col=1)

                # 3. MACD
                fig.add_trace(go.Scatter(x=df.index, y=df[col_macd], line=dict(color='blue', width=1), name='DIF'), row=3, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df[col_macds], line=dict(color='orange', width=1), name='DEA'), row=3, col=1)
                fig.add_trace(go.Bar(x=df.index, y=df[col_macdh], marker_color=['red' if v < 0 else 'green' for v in df[col_macdh]], name='Hist'), row=3, col=1)

                # 4. KDJ
                fig.add_trace(go.Scatter(x=df.index, y=df[col_k], line=dict(color='purple', width=1), name='K'), row=4, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df[col_d], line=dict(color='orange', width=1), name='D'), row=4, col=1)
                fig.add_hline(y=20, line_color="gray", line_dash="dot", row=4, col=1)
                fig.add_hline(y=80, line_color="gray", line_dash="dot", row=4, col=1)

                # 5. RSI
                fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='blue', width=1), name='RSI'), row=5, col=1)
                fig.add_hline(y=50, line_color="gray", line_dash="dot", row=5, col=1)
                fig.add_hline(y=80, line_color="red", line_dash="dash", row=5, col=1)
                fig.add_hline(y=20, line_color="green", line_dash="dash", row=5, col=1)

                # 決定 Y 軸類型 (固定為線性)
                y_axis_type = "linear"

                # --- 計算初始可視範圍內的價格區間 (Auto Y-Range) ---
                df_visible = df.iloc[range_start_idx:]
                
                price_min = df_visible[['Low', col_bbl, 'MA5', 'MA90']].min().min()
                price_max = df_visible[['High', col_bbu, 'MA5', 'MA90']].max().max()
                
                padding = (price_max - price_min) * 0.05
                y_min_limit = price_min - padding
                y_max_limit = price_max + padding
                
                initial_y_range = [y_min_limit, y_max_limit]

                fig.update_yaxes(side='right')

                fig.update_xaxes(
                    type='category', 
                    range=[range_start_idx, range_end_idx],
                    showspikes=True, spikemode='across', spikesnap='cursor', showline=True, showgrid=True,
                    tickmode='auto', nticks=20 
                )
                fig.update_xaxes(showticklabels=True, row=1, col=1)
                # 修改: 高度固定為 1200 (移除了 chart_height 變數)
                fig.update_layout(
                    height=1200, 
                    xaxis_rangeslider_visible=False, 
                    template="plotly_white", 
                    dragmode='pan', 
                    hovermode='x unified',
                    yaxis=dict(
                        type=y_axis_type, 
                        range=initial_y_range, 
                        fixedrange=False,
                        side='right'
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error("發生錯誤：")
            st.exception(e)