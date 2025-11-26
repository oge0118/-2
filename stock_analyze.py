import streamlit as st
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import sys
from datetime import datetime, timedelta

# --- 頁面設定 ---
st.set_page_config(page_title="量化交易策略系統", layout="wide")
st.title("📈 智能股票交易系統")

# --- 側邊欄：參數設定 ---
st.sidebar.header("📊 參數設定 (Parameters)")

# 修改 1: 預設值改為空白
ticker = st.sidebar.text_input("輸入股票代碼", value="") 
st.sidebar.caption("範例: QQQ, VOO, NVDA, 2330.TW")

with st.sidebar.expander("🔧 技術指標參數", expanded=True):
    # MACD
    st.write("**MACD**")
    macd_fast = st.number_input("Fast Period", value=12)
    macd_slow = st.number_input("Slow Period", value=26)
    macd_signal = st.number_input("Signal Period", value=9)
    
    # RSI
    st.write("**RSI**")
    rsi_period = st.number_input("RSI Period", value=14)
    rsi_overbought = st.number_input("Overbought (>)", value=80)
    
    # KDJ
    st.write("**KDJ**")
    kdj_period = st.number_input("KDJ Period", value=9)
    kdj_signal = st.number_input("Signal Period", value=3)
    kdj_high = st.number_input("KDJ High Level (>)", value=80) 
    
    # MTM
    st.write("**MTM (動量)**")
    mtm_n = st.number_input("MTM Period (N)", value=10)
    mtm_ma = st.number_input("MTMMA Period", value=10)
    
    # OSC
    st.write("**OSC (震盪)**")
    osc_short = st.number_input("OSC Short MA", value=10)
    osc_long = st.number_input("OSC Long MA", value=20)
    osc_ema_len = st.number_input("OSC EMA Period", value=10)

# 修改 2: 新增圖表高度設定
st.sidebar.subheader("🎨 圖表顯示設定")
chart_height = st.sidebar.number_input("圖表高度 (px)", value=1200, min_value=600, max_value=3000, step=100)

# 修改 3: 開始日期預設為 2023/1/1
start_date = st.sidebar.date_input("開始日期", value=datetime(2023, 1, 1))

# --- 核心邏輯函數 ---
def calculate_strategy(df):
    """
    計算所有指標與買賣訊號，並進行資金回測
    """
    # 1. 計算技術指標
    df['MA5'] = ta.sma(df['Close'], length=5)
    df['MA90'] = ta.sma(df['Close'], length=90)

    # MACD
    df.ta.macd(fast=macd_fast, slow=macd_slow, signal=macd_signal, append=True)
    col_macd = f'MACD_{macd_fast}_{macd_slow}_{macd_signal}'
    col_macdh = f'MACDh_{macd_fast}_{macd_slow}_{macd_signal}' 
    col_macds = f'MACDs_{macd_fast}_{macd_slow}_{macd_signal}' 
    
    # RSI
    df['RSI'] = ta.rsi(df['Close'], length=rsi_period)
    
    # KDJ
    kdj = df.ta.kdj(length=kdj_period, signal=kdj_signal)
    df = pd.concat([df, kdj], axis=1)
    col_k = f'K_{kdj_period}_{kdj_signal}'
    col_d = f'D_{kdj_period}_{kdj_signal}'
    
    # MTM
    df['MTM'] = df['Close'] - df['Close'].shift(mtm_n)
    df['MTMMA'] = ta.sma(df['MTM'], length=mtm_ma)
    
    # OSC
    sma_short = ta.sma(df['Close'], length=osc_short)
    sma_long = ta.sma(df['Close'], length=osc_long)
    df['OSC'] = sma_short - sma_long
    df['OSCEMA'] = ta.ema(df['OSC'], length=osc_ema_len)
    
    df.dropna(inplace=True)
    
    # 2. 定義邏輯狀態
    
    # A. 趨勢過濾器
    cond_macd_bull = (df[col_macd] > df[col_macds]) & (df[col_macdh] > 0)
    cond_osc_bull = df['OSC'] > 0
    df['Trend_Bull'] = cond_macd_bull | cond_osc_bull
    
    # B. 動能過濾器
    df['Momentum_Strong'] = (df['MTM'] > 0) & (df['MTM'] > df['MTMMA'])
    df['Safe_Zone'] = df['RSI'] < rsi_overbought
    
    # C. 觸發訊號 - KDJ
    df['KDJ_Gold'] = (df[col_k] > df[col_d]) & (df[col_k].shift(1) < df[col_d].shift(1))
    df['KDJ_Dead'] = (df[col_k] < df[col_d]) & (df[col_k].shift(1) > df[col_d].shift(1))
    
    # 3. 原始訊號計算 (Raw Signals)
    
    # 🔵 原始買入條件
    raw_buy = (
        df['Trend_Bull'] & 
        df['Momentum_Strong'] & 
        df['Safe_Zone'] & 
        df['KDJ_Gold']
    )
    
    # 🔴 原始賣出條件 (3選2)
    cond_kdj_dead_high = df['KDJ_Dead'] & (df[col_d] > kdj_high)
    osc_cross_down = (df['OSC'] < df['OSCEMA']) & (df['OSC'].shift(1) > df['OSCEMA'].shift(1))
    cond_osc_weak_high = osc_cross_down & (df['OSC'] > 0)
    cond_rsi_hot = df['RSI'] > rsi_overbought
    
    sell_condition_count = (
        cond_kdj_dead_high.astype(int) + 
        cond_osc_weak_high.astype(int) + 
        cond_rsi_hot.astype(int)
    )
    
    raw_sell = sell_condition_count >= 2
    
    # 4. 訊號過濾與資金回測 (Updated Logic)
    
    buy_signals = []
    sell_signals = []
    holding = False 
    
    initial_capital = 100000.0
    cash = initial_capital
    position_size = 0.0 
    asset_history = [] 
    entry_price = 0.0 
    last_entry_price_record = 0.0 
    
    raw_buy_list = raw_buy.values
    raw_sell_list = raw_sell.values
    close_prices = df['Close'].values
    
    for i in range(len(df)):
        is_buy_raw = raw_buy_list[i]
        is_sell_raw = raw_sell_list[i]
        current_price = close_prices[i]
        
        # 修改 2: 買入訊號不再受 holding 狀態限制，只要符合條件就標記
        current_buy_signal = 1 if is_buy_raw else 0
        current_sell_signal = 0
        
        if not holding:
            # 空倉狀態
            if is_buy_raw:
                # 執行買入
                holding = True
                entry_price = current_price
                last_entry_price_record = entry_price
                position_size = cash / current_price
                cash = 0
        else:
            # 持倉狀態
            if is_sell_raw:
                # 執行賣出
                current_sell_signal = 1 # 只有在持倉時且觸發賣出條件，才標記賣出
                holding = False
                cash = position_size * current_price
                position_size = 0
                entry_price = 0
            
            # 註：如果在持倉時遇到 is_buy_raw，我們會標記買入訊號 (current_buy_signal=1)，
            # 但因為資金已滿 (All-in)，所以不會有額外的資金動作，視為「持倉/加倉建議」。
        
        buy_signals.append(current_buy_signal)
        sell_signals.append(current_sell_signal)
        
        # 計算當日總資產
        if holding:
            current_asset = position_size * current_price
        else:
            current_asset = cash
            
        asset_history.append(current_asset)
            
    df['Buy_Signal'] = buy_signals
    df['Sell_Signal'] = sell_signals
    df['Total_Asset'] = asset_history 
    
    return df, col_k, col_d, last_entry_price_record

# --- 主程式執行 ---
if st.button("開始策略回測", type="primary"):
    # 修改 3: 檢查是否輸入股票代碼
    if not ticker.strip():
        st.warning("⚠️ 請輸入股票代碼 (例如 NVDA) 才能開始分析！")
        st.stop()
        
    with st.spinner(f'正在運算 {ticker} 的交易策略...'):
        try:
            # 1. 下載數據
            df = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if df.empty:
                st.error("找不到數據，請確認股票代碼。")
            else:
                # 2. 執行策略計算
                df, col_k, col_d, last_entry_price = calculate_strategy(df)
                
                # 取得最後一天的狀態
                curr = df.iloc[-1]
                
                # 計算報酬率
                initial_capital = 100000.0
                total_return = (curr['Total_Asset'] - initial_capital) / initial_capital * 100
                
                # 3. 顯示結果
                st.markdown("---")
                
                c1, c2, c3, c4 = st.columns(4)
                
                # 判斷持倉狀態
                last_buy_idx = df[df['Buy_Signal']==1].index.max()
                last_sell_idx = df[df['Sell_Signal']==1].index.max()
                
                is_holding = False
                if pd.isna(last_buy_idx):
                    is_holding = False
                elif pd.isna(last_sell_idx):
                    is_holding = True # 有買沒賣
                else:
                    is_holding = last_buy_idx > last_sell_idx 
                
                c1.metric("最新收盤價", f"${curr['Close']:.2f}")

                if is_holding:
                    if last_entry_price > 0:
                        unrealized_pnl = (curr['Close'] - last_entry_price) / last_entry_price * 100
                        c2.metric("目前倉位", "🟢 持倉中", f"{unrealized_pnl:.2f}% (成本: {last_entry_price:.2f})")
                    else:
                        c2.metric("目前倉位", "🟢 持倉中", "成本計算中")
                else:
                    c2.metric("目前倉位", "⚪ 空手", "等待買點")

                c3.metric("策略總報酬率", f"{total_return:.2f}%", f"總資產: ${curr['Total_Asset']:.0f}")

                with c4.container():
                    st.write("📋 **當前邏輯:**")
                    st.write(f"- 趨勢多頭: {'✅' if curr['Trend_Bull'] else '❌'}")
                    st.write(f"- 動能強勁: {'✅' if curr['Momentum_Strong'] else '❌'}")
                    st.write(f"- RSI 安全: {'✅' if curr['Safe_Zone'] else '❌'}")
                    st.write(f"- KDJ 交叉: {'🟡 金叉' if curr['KDJ_Gold'] else ('⚫ 死叉' if curr['KDJ_Dead'] else '無')}")

                # --- 繪圖 (Plotly) ---
                st.subheader(f"📊 {ticker} 策略訊號圖")
                
                buy_points = df[df['Buy_Signal'] == 1]
                sell_points = df[df['Sell_Signal'] == 1]
                
                fig = make_subplots(rows=5, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.02, 
                                    row_heights=[0.4, 0.15, 0.15, 0.15, 0.15],
                                    subplot_titles=("價格與交易訊號", "交易量", "MACD", "KDJ", "RSI"))

                # 1. K線圖
                fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'].values, high=df['High'].values,
                                low=df['Low'].values, close=df['Close'].values, name='Price'), row=1, col=1)
                
                fig.add_trace(go.Scatter(x=df.index, y=df['MA5'].values, line=dict(color='yellow', width=1), name='MA5'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['MA90'].values, line=dict(color='purple', width=1), name='MA90'), row=1, col=1)

                # 買入標記
                fig.add_trace(go.Scatter(x=buy_points.index, y=buy_points['Low'].values*0.98, 
                                        mode='markers', marker=dict(symbol='triangle-up', size=12, color='blue'),
                                        name='Buy Signal'), row=1, col=1)
                
                # 賣出標記
                fig.add_trace(go.Scatter(x=sell_points.index, y=sell_points['High'].values*1.02, 
                                        mode='markers', marker=dict(symbol='triangle-down', size=10, color='orange'),
                                        name='Exit Signal'), row=1, col=1)

                # 2. 交易量
                vol_colors = ['green' if c >= o else 'red' for c, o in zip(df['Close'].values, df['Open'].values)]
                fig.add_trace(go.Bar(x=df.index, y=df['Volume'].values, marker_color=vol_colors, name='Volume'), row=2, col=1)

                # 3. MACD
                col_macd_val = f'MACD_{macd_fast}_{macd_slow}_{macd_signal}'
                col_macdh_val = f'MACDh_{macd_fast}_{macd_slow}_{macd_signal}'
                col_macds_val = f'MACDs_{macd_fast}_{macd_slow}_{macd_signal}'
                
                fig.add_trace(go.Scatter(x=df.index, y=df[col_macd_val].values, line=dict(color='blue', width=1), name='DIF'), row=3, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df[col_macds_val].values, line=dict(color='orange', width=1), name='DEA'), row=3, col=1)
                fig.add_trace(go.Bar(x=df.index, y=df[col_macdh_val].values, name='MACD Hist', marker_color=['red' if v < 0 else 'green' for v in df[col_macdh_val].values]), row=3, col=1)

                # 4. KDJ
                fig.add_trace(go.Scatter(x=df.index, y=df[col_k].values, line=dict(color='purple', width=1), name='K'), row=4, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df[col_d].values, line=dict(color='orange', width=1), name='D'), row=4, col=1)
                fig.add_hline(y=20, line_color="gray", line_dash="dot", row=4, col=1)
                fig.add_hline(y=80, line_color="gray", line_dash="dot", row=4, col=1)

                # 5. RSI
                fig.add_trace(go.Scatter(x=df.index, y=df['RSI'].values, line=dict(color='blue', width=1), name='RSI'), row=5, col=1)
                fig.add_hline(y=50, line_color="gray", line_dash="dot", row=5, col=1)
                fig.add_hline(y=80, line_color="red", line_dash="dash", row=5, col=1)
                fig.add_hline(y=20, line_color="green", line_dash="dash", row=5, col=1)
                
                # 修改 4: 使用自訂高度
                fig.update_layout(height=chart_height, xaxis_rangeslider_visible=False, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("📋 訊號數據明細 (Data Log)")
                
                output_cols = ['Close', 'Total_Asset', 'MA5', 'MA90', 'Volume', 'RSI', 'MTM', 'MTMMA', 'OSC', 'OSCEMA', col_k, col_d, 'Buy_Signal', 'Sell_Signal']
                
                st.dataframe(df[output_cols].tail(50))
                
                csv = df.to_csv().encode('utf-8')
                st.download_button(
                    label="📥 下載完整 CSV",
                    data=csv,
                    file_name=f'{ticker}_strategy_result.csv',
                    mime='text/csv',
                )

        except Exception as e:
            st.error("發生錯誤：")
            st.exception(e)