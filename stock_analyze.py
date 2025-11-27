import streamlit as st
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta

# --- 頁面設定 ---
st.set_page_config(page_title="專業量化回測系統", layout="wide")

# --- 輔助函數 ---
@st.cache_data(ttl=300)
def load_data(ticker, start_date, interval):
    try:
        if interval.endswith('m'):
            limit_date = datetime.now() - timedelta(days=59)
            start_datetime = datetime.combine(start_date, datetime.min.time())
            if start_datetime < limit_date:
                start_date = limit_date.date()
        
        df = yf.download(ticker, start=start_date, interval=interval, progress=False, auto_adjust=True)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            
        return df
    except Exception as e:
        return pd.DataFrame()

def calculate_cagr(total_return, days):
    if days <= 0: return 0
    return ((1 + total_return/100) ** (365/days) - 1) * 100

def calculate_sharpe_sortino(returns, risk_free_rate=0.02):
    if len(returns) < 2: return 0, 0
    excess_returns = returns - (risk_free_rate / 252)
    std = returns.std()
    downside_std = returns[returns < 0].std()
    sharpe = (excess_returns.mean() / std * np.sqrt(252)) if std > 0 else 0
    sortino = (excess_returns.mean() / downside_std * np.sqrt(252)) if downside_std > 0 else 0
    return sharpe, sortino

# --- 側邊欄設定 ---
st.sidebar.header("⚙️ 策略參數設定")

# 1. 模式選擇
strategy_mode = st.sidebar.radio("選擇回測模式", ["模式一：長線回測 (趨勢與資金)", "模式二：短線回測 (5分K + ATR風控)"])

# 2. 股票設定
ticker_input = st.sidebar.text_input("股票代碼", value="NVDA")
ticker = ticker_input.strip().upper()
if ticker.isdigit(): ticker = f"{ticker}.TW"

# 按鈕
run_button = st.sidebar.button("開始回測", type="primary")

commission = st.sidebar.number_input("單邊手續費率 (%)", 0.0, 1.0, 0.1425, step=0.01) / 100.0

# 初始化變數
interval = "1d" 
start_date = datetime.now()

# 3. 動態參數與日期
if strategy_mode.startswith("模式一"):
    st.sidebar.subheader("📅 長線設定")
    interval = "1d"
    start_date = st.sidebar.date_input("開始日期", datetime(2020, 1, 1))
    
    with st.sidebar.expander("🔧 長線指標參數", expanded=True):
        # MACD (加入 key 避免重複 ID)
        st.write("**MACD (趨勢)**")
        macd_fast = st.number_input("Fast", 5, 50, 15, key="lt_macd_fast")
        macd_slow = st.number_input("Slow", 10, 100, 30, key="lt_macd_slow")
        macd_sig = st.number_input("Signal", 5, 50, 9, key="lt_macd_sig")
        
        # RSI
        st.write("**RSI (動能)**")
        rsi_len = st.number_input("Length", 5, 50, 21, key="lt_rsi_len")
        # 修正: max_value 調高到 60，避免 50 超出範圍
        rsi_lower = st.number_input("Lower Bound", 10, 60, 50, help="買入需大於此值", key="lt_rsi_lower")
        rsi_upper = st.number_input("Upper Bound", 60, 90, 70, help="買入需小於此值", key="lt_rsi_upper")
        
        # OBV
        st.write("**OBV (資金)**")
        obv_ma_len = st.number_input("OBV MA Length", 5, 100, 20, key="lt_obv_len")
        
        # BB
        st.write("**Bollinger Bands (位置)**")
        bb_len = st.number_input("BB Length", 5, 50, 20, key="lt_bb_len")
        bb_std = st.number_input("BB Std", 1.0, 3.0, 2.0, key="lt_bb_std")
        
        # Stop Loss
        st.write("**風控**")
        hard_stop_pct = st.number_input("硬性停損 (%)", 1.0, 20.0, 7.0, step=0.5, key="lt_sl") / 100.0

else:
    st.sidebar.subheader("⚡ 短線設定")
    interval = "5m"
    # 修改: 預設抓取過去 30 天數據，讓使用者有更多歷史 K 線可滑動
    d_start = datetime.now() - timedelta(days=30)
    min_d = datetime.now() - timedelta(days=59)
    start_date = st.sidebar.date_input("開始日期 (限最近60天)", d_start, min_value=min_d, max_value=datetime.now())
    
    with st.sidebar.expander("🔧 短線指標參數", expanded=True):
        # KDJ
        st.write("**KDJ (觸發)**")
        kdj_k = st.number_input("Period", 3, 30, 9, key="st_kdj_k")
        kdj_smooth = st.number_input("Smooth", 1, 10, 3, key="st_kdj_s")
        
        # RSI
        st.write("**RSI (動能)**")
        rsi_short_len = st.number_input("Length", 2, 20, 6, key="st_rsi_len")
        
        # MACD
        st.write("**MACD (背景)**")
        macd_s_fast = st.number_input("Fast", 2, 20, 5, key="st_macd_fast")
        macd_s_slow = st.number_input("Slow", 5, 50, 10, key="st_macd_slow")
        macd_s_sig = st.number_input("Signal", 2, 20, 3, key="st_macd_sig")
        
        # ADX
        st.write("**ADX (趨勢強度)**")
        adx_len = st.number_input("Length", 5, 50, 10, key="st_adx_len")
        adx_limit = st.number_input("Threshold", 10, 50, 25, help="大於此值視為有趨勢", key="st_adx_lim")
        
        # BB
        st.write("**BB (價格空間)**")
        bb_s_len = st.number_input("BB Length", 5, 50, 14, key="st_bb_len")
        bb_s_std = st.number_input("BB Std", 1.0, 3.0, 2.0, key="st_bb_std")
        
        # OBV
        st.write("**OBV (資金驗證)**")
        obv_s_len = st.number_input("OBV MA", 5, 100, 20, key="st_obv_len")
        
        # ATR
        st.write("**ATR (波動風控)**")
        atr_len = st.number_input("ATR Length", 5, 50, 14, key="st_atr_len")
        atr_mult_sl = st.number_input("停損 ATR 倍數", 1.0, 5.0, 1.5, key="st_atr_mult")
        
        # Time
        st.write("**時間風控**")
        max_hold_hours = st.number_input("最大持倉小時", 0.5, 6.0, 2.0, key="st_time_hold")

# --- 策略邏輯 ---

def run_long_term_strategy(df):
    # 1. 計算指標
    # MACD
    macd = df.ta.macd(fast=macd_fast, slow=macd_slow, signal=macd_sig)
    df['DIF'] = macd[f'MACD_{macd_fast}_{macd_slow}_{macd_sig}']
    df['DEA'] = macd[f'MACDs_{macd_fast}_{macd_slow}_{macd_sig}']
    # RSI
    df['RSI'] = ta.rsi(df['Close'], length=rsi_len)
    # OBV
    df['OBV'] = ta.obv(df['Close'], df['Volume'])
    df['OBV_MA'] = ta.sma(df['OBV'], length=obv_ma_len)
    # BB
    bb = ta.bbands(df['Close'], length=bb_len, std=bb_std)
    # 動態抓取 BB 欄位
    cols = bb.columns
    col_bbu = [c for c in cols if c.startswith('BBU')][0]
    col_bbm = [c for c in cols if c.startswith('BBM')][0]
    col_bbl = [c for c in cols if c.startswith('BBL')][0]
    df['BBU'] = bb[col_bbu]
    df['BBM'] = bb[col_bbm]
    df['BBL'] = bb[col_bbl]
    
    df.dropna(inplace=True)
    
    # 2. 訊號判斷 (向量化預算)
    # 買入條件
    # A. MACD 金叉且在0軸上 (簡化：金叉即可，但在0軸上更強，這裡依照您的要求：金叉且DIF>0)
    cond_macd_buy = (df['DIF'] > df['DEA']) & (df['DIF'] > 0)
    # B. OBV 支持
    cond_obv_buy = (df['OBV'] > df['OBV_MA'])
    # C. 位置優勢 (中軌之上，RSI 50-70)
    cond_pos_buy = (df['Close'] > df['BBM']) & (df['RSI'] > rsi_lower) & (df['RSI'] < rsi_upper)
    
    df['Signal_Buy'] = cond_macd_buy & cond_obv_buy & cond_pos_buy
    
    # 賣出條件 (僅標記技術面賣訊，硬停損在回測迴圈處理)
    # A. MACD 死叉 或 DIF < 0
    cond_macd_sell = (df['DIF'] < df['DEA']) | (df['DIF'] < 0)
    # B. 資金外流
    cond_obv_sell = (df['OBV'] < df['OBV_MA'])
    
    df['Signal_Sell_Tech'] = cond_macd_sell | cond_obv_sell
    
    return df

def run_short_term_strategy(df):
    # 補上 MA 計算 (修復繪圖錯誤)
    df['MA5'] = ta.sma(df['Close'], length=5)
    df['MA90'] = ta.sma(df['Close'], length=90)
    
    # 計算指標
    kdj = df.ta.kdj(length=kdj_k, signal=kdj_smooth)
    df['K'] = kdj[f'K_{kdj_k}_{kdj_smooth}']
    df['D'] = kdj[f'D_{kdj_k}_{kdj_smooth}']
    df['J'] = kdj[f'J_{kdj_k}_{kdj_smooth}']
    
    df['RSI'] = ta.rsi(df['Close'], length=rsi_short_len)
    
    macd = df.ta.macd(fast=macd_s_fast, slow=macd_s_slow, signal=macd_s_sig)
    df['DIF'] = macd[f'MACD_{macd_s_fast}_{macd_s_slow}_{macd_s_sig}']
    df['DEA'] = macd[f'MACDs_{macd_s_fast}_{macd_s_slow}_{macd_s_sig}']
    
    adx = df.ta.adx(length=adx_len)
    df['ADX'] = adx[f'ADX_{adx_len}']
    
    bb = ta.bbands(df['Close'], length=bb_s_len, std=bb_s_std)
    cols = bb.columns
    col_bbu = [c for c in cols if c.startswith('BBU')][0]
    col_bbm = [c for c in cols if c.startswith('BBM')][0]
    col_bbl = [c for c in cols if c.startswith('BBL')][0]
    df['BBU'] = bb[col_bbu]
    df['BBM'] = bb[col_bbm]
    df['BBL'] = bb[col_bbl]
    
    df['OBV'] = ta.obv(df['Close'], df['Volume'])
    df['OBV_MA'] = ta.sma(df['OBV'], length=obv_s_len)
    
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=atr_len)
    
    df.dropna(inplace=True)
    
    # 訊號
    trigger = (df['K'] > df['D']) & (df['K'].shift(1) < df['D'].shift(1))
    vol_ok = df['OBV'] > df['OBV_MA']
    
    cond_breakout = (df['Close'] >= df['BBU'] * 0.995) & (df['RSI'] > 60) & (df['ADX'] > adx_limit)
    cond_reversion = (df['Close'] <= df['BBL'] * 1.005) & (df['RSI'] < 40) & (df['J'] < 0)
    
    df['Signal_Buy'] = trigger & vol_ok & (cond_breakout | cond_reversion)
    df['Buy_Type'] = np.select([cond_breakout, cond_reversion], ['突破追價', '反彈抄底'], default='')
    
    return df

# --- 回測引擎 ---
def run_backtest(df, mode):
    initial_capital = 100000
    cash = initial_capital
    position = 0
    cost_basis = 0 
    
    history = [] 
    trades = []
    
    entry_price = 0
    entry_time = None
    highest_price = 0
    stop_loss_price = 0
    
    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    opens = df['Open'].values
    dates = df.index
    
    buys = df['Signal_Buy'].values
    tech_sells = df['Signal_Sell_Tech'].values if mode == "Long" else np.zeros(len(df), dtype=bool)
    atrs = df['ATR'].values if mode == "Short" else np.zeros(len(df))
    
    action_buy_idx = []
    action_sell_idx = []
    action_reasons = []
    
    for i in range(len(df)):
        current_price = closes[i]
        current_time = dates[i]
        
        # 1. 檢查出場 (如果持有)
        if position > 0:
            # 更新最高價 (用於移動停利)
            if highs[i] > highest_price:
                highest_price = highs[i]
            
            is_exit = False
            exit_reason = ""
            exit_price = current_price
            
            if mode == "Long":
                # 長線出場: 1. 技術轉弱 2. 硬停損
                if tech_sells[i]:
                    is_exit = True
                    exit_reason = "技術轉弱 (MACD/OBV)"
                elif current_price < entry_price * (1 - hard_stop_pct):
                    is_exit = True
                    exit_reason = f"硬性停損 (-{hard_stop_pct*100}%)"
                    exit_price = min(opens[i], entry_price * (1 - hard_stop_pct)) # 模擬跳空
            
            else: # Short
                # 短線出場: 1. ATR 停損 2. 移動停利 (這裡簡化邏輯，假設回落 1.5 ATR 為移動停利點，或使用外部參數)
                # 使用 ATR 動態停損
                if lows[i] < stop_loss_price:
                    is_exit = True
                    exit_reason = "ATR 波動停損"
                    exit_price = stop_loss_price # 觸價
                
                # 移動停利 (假設回落 2 ATR 出場，或可自訂)
                trailing_stop_p = highest_price - (2.0 * atrs[i])
                if lows[i] < trailing_stop_p and current_price > entry_price: # 確保是獲利才叫移動停利
                    is_exit = True
                    exit_reason = "ATR 移動停利"
                    exit_price = max(opens[i], trailing_stop_p)
                
                # 時間停損 (持倉 > 2 小時)
                time_held = (current_time - entry_time).total_seconds() / 3600
                if time_held > max_hold_hours and (current_price - entry_price)/entry_price < 0.005: # 沒賺多少就跑
                    is_exit = True
                    exit_reason = "時間停損 (超時)"
                
                # 收盤強制平倉 (13:25) - 原本的邏輯保留或移除，此處保留基本檢查

            # 執行賣出
            if is_exit:
                # 計算手續費
                gross_val = position * exit_price
                fee = gross_val * commission
                cash = gross_val - fee
                
                # 紀錄
                pnl = cash - cost_basis # cost_basis 包含買入成本
                pnl_pct = (exit_price - entry_price) / entry_price
                
                trades.append({
                    'Entry Time': entry_time,
                    'Exit Time': current_time,
                    'Entry Price': entry_price,
                    'Exit Price': exit_price,
                    'PnL': pnl,
                    'PnL %': pnl_pct,
                    'Reason': exit_reason
                })
                
                action_sell_idx.append(current_time)
                action_reasons.append(exit_reason)
                
                position = 0
                entry_price = 0
                highest_price = 0

        # 2. 檢查進場 (如果空手)
        if position == 0:
            if buys[i]:
                # 買入
                cost = cash * commission
                position_val = cash - cost
                position = position_val / current_price
                cost_basis = cash # 紀錄投入的現金
                cash = 0
                
                entry_price = current_price
                entry_time = current_time
                highest_price = current_price
                
                # 設定短線 ATR 停損價
                if mode == "Short":
                    stop_loss_price = entry_price - (atr_mult_sl * atrs[i])
                
                action_buy_idx.append(current_time)

        # 3. 更新資產淨值
        equity = cash + (position * current_price)
        history.append(equity)
        
    df['Equity'] = history
    return df, pd.DataFrame(trades), action_buy_idx, action_sell_idx, action_reasons

# --- 主程式執行 ---
if run_button:
    if not ticker:
        st.warning("請輸入股票代碼")
        st.stop()
        
    with st.spinner(f"正在分析 {ticker} ..."):
        # 載入數據
        df = load_data(ticker, start_date, interval)
        
        if df.empty:
            st.error("找不到數據或數據不足。")
        else:
            # 策略計算
            if strategy_mode.startswith("模式一"):
                df = run_long_term_strategy(df)
                df_res, trades_df, buys, sells, sell_reasons = run_backtest(df, "Long")
            else:
                df = run_short_term_strategy(df)
                df_res, trades_df, buys, sells, sell_reasons = run_backtest(df, "Short")
            
            # --- 績效統計 ---
            initial_capital = 100000
            final_equity = df_res['Equity'].iloc[-1]
            total_return_pct = (final_equity - initial_capital) / initial_capital * 100
            
            # 每日/每條回測數據的報酬率 (用於 Sharpe)
            equity_curve = df_res['Equity']
            returns = equity_curve.pct_change().dropna()
            
            sharpe, sortino = calculate_sharpe_sortino(returns)
            
            # 最大回撤
            running_max = equity_curve.cummax()
            drawdown = (equity_curve - running_max) / running_max
            max_drawdown = drawdown.min() * 100
            
            # 交易統計
            total_trades = len(trades_df)
            if total_trades > 0:
                win_rate = len(trades_df[trades_df['PnL'] > 0]) / total_trades * 100
                gross_profit = trades_df[trades_df['PnL'] > 0]['PnL'].sum()
                gross_loss = abs(trades_df[trades_df['PnL'] < 0]['PnL'].sum())
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                avg_pnl = trades_df['PnL'].mean()
                
                # 連續虧損 (簡單版)
                loss_streak = 0
                max_loss_streak = 0
                curr_streak = 0
                for pnl in trades_df['PnL']:
                    if pnl < 0:
                        curr_streak += 1
                    else:
                        max_loss_streak = max(max_loss_streak, curr_streak)
                        curr_streak = 0
                max_loss_streak = max(max_loss_streak, curr_streak)
            else:
                win_rate = 0; profit_factor = 0; avg_pnl = 0; max_loss_streak = 0
            
            # --- 顯示儀表板 ---
            st.markdown(f"### 📊 {ticker} 回測報告")
            
            # 1. 核心績效
            st.subheader("一、核心績效指標 (KPIs)")
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("總報酬率", f"{total_return_pct:.2f}%", f"${final_equity:.0f}")
            # CAGR 需要天數
            days = (df.index[-1] - df.index[0]).days
            cagr = calculate_cagr(total_return_pct, days)
            k2.metric("年化報酬 (CAGR)", f"{cagr:.2f}%")
            k3.metric("夏普比率", f"{sharpe:.2f}")
            k4.metric("索提諾比率", f"{sortino:.2f}")
            k5.metric("獲利因子", f"{profit_factor:.2f}")
            
            # 2. 風險指標
            st.subheader("二、風險與穩定性")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("最大回撤 (MDD)", f"{max_drawdown:.2f}%", delta_color="inverse")
            r2.metric("勝率", f"{win_rate:.1f}%")
            r3.metric("波動率 (Std)", f"{returns.std()*100:.2f}%")
            r4.metric("最大連虧次數", f"{max_loss_streak} 次", delta_color="inverse")
            
            # 3. 交易細節
            st.subheader("三、交易執行細節")
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("總交易次數", f"{total_trades}")
            e2.metric("平均每筆盈虧", f"${avg_pnl:.0f}")
            e3.metric("總淨利", f"${trades_df['PnL'].sum():.0f}" if total_trades > 0 else "$0")
            e4.metric("手續費設定", f"{commission*100}%")

            # --- 圖表區 ---
            st.subheader("四、圖形化結果")
            
            # 格式化 Index 避免 Plotly 缺口 (使用 Category Axis)
            fmt = '%Y-%m-%d %H:%M' if interval == "5m" else '%Y-%m-%d'
            df.index_str = df.index.strftime(fmt)
            
            # A. 淨值與回撤
            fig_equity = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
            fig_equity.add_trace(go.Scatter(x=df.index_str, y=df['Equity'], name="淨值曲線", line=dict(color='cyan')), row=1, col=1)
            fig_equity.add_trace(go.Scatter(x=df.index_str, y=drawdown*100, name="回撤幅度 %", fill='tozeroy', line=dict(color='red')), row=2, col=1)
            
            # 設定顯示範圍 (最近一年或全部)
            total_len = len(df)
            zoom_len = 150 # 稍微縮小預設範圍，確保 K 線夠粗
            start_idx = max(0, total_len - zoom_len)
            
            fig_equity.update_xaxes(type='category', range=[start_idx, total_len-1], nticks=10)
            fig_equity.update_layout(height=500, title="淨值與回撤圖", template="plotly_dark", hovermode="x unified")
            st.plotly_chart(fig_equity, use_container_width=True, config={'scrollZoom': True})
            
            # B. 交易標記圖
            fig_chart = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03, subplot_titles=("價格與交易點", "成交量"))
            
            # K線
            fig_chart.add_trace(go.Candlestick(x=df.index_str, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
            
            # 布林 / 均線 (視模式畫)
            if strategy_mode.startswith("模式一"):
                fig_chart.add_trace(go.Scatter(x=df.index_str, y=df['BBU'], line=dict(color='gray', dash='dot'), name='BBU'), row=1, col=1)
                fig_chart.add_trace(go.Scatter(x=df.index_str, y=df['BBL'], line=dict(color='gray', dash='dot'), name='BBL'), row=1, col=1)
            else:
                fig_chart.add_trace(go.Scatter(x=df.index_str, y=df['BBU'], line=dict(color='gray', dash='dot'), name='BBU'), row=1, col=1)
                fig_chart.add_trace(go.Scatter(x=df.index_str, y=df['BBL'], line=dict(color='gray', dash='dot'), name='BBL'), row=1, col=1)

            # 建立日期對照表 (將 datetime 轉為 string index)
            dt_to_str = {dt: s for dt, s in zip(df.index, df.index_str)}
            
            # 處理買入訊號 (維持藍色)
            buy_x = [dt_to_str[dt] for dt in buys if dt in dt_to_str]
            buy_y = df.loc[buys]['Low'] * 0.99
            
            fig_chart.add_trace(go.Scatter(
                x=buy_x, y=buy_y, 
                mode='markers', 
                marker=dict(symbol='triangle-up', size=12, color='blue'), 
                name='買進', 
                hovertext=df.loc[buys]['Buy_Reason'] if 'Buy_Reason' in df.columns else None
            ), row=1, col=1)
            
            # 處理賣出訊號
            if not trades_df.empty:
                wins = trades_df[trades_df['PnL'] > 0]
                losses = trades_df[trades_df['PnL'] <= 0]
                
                win_x = [dt_to_str[dt] for dt in wins['Exit Time'] if dt in dt_to_str]
                win_y = [row['Exit Price'] * 1.01 for _, row in wins.iterrows()]
                
                loss_x = [dt_to_str[dt] for dt in losses['Exit Time'] if dt in dt_to_str]
                loss_y = [row['Exit Price'] * 1.01 for _, row in losses.iterrows()]
                
                fig_chart.add_trace(go.Scatter(
                    x=win_x, y=win_y, 
                    mode='markers', 
                    marker=dict(symbol='triangle-down', size=12, color='orange'), 
                    name='賣出(獲利)', 
                    hovertext=wins['Reason']
                ), row=1, col=1)

                fig_chart.add_trace(go.Scatter(
                    x=loss_x, y=loss_y, 
                    mode='markers', 
                    marker=dict(symbol='triangle-down', size=12, color='#FF00FF'), 
                    name='賣出(虧損)', 
                    hovertext=losses['Reason']
                ), row=1, col=1)
            
            # Volume
            colors = ['red' if c >= o else 'green' for c, o in zip(df['Close'], df['Open'])]
            fig_chart.add_trace(go.Bar(x=df.index_str, y=df['Volume'], marker_color=colors, name='Volume'), row=2, col=1)
            
            # X軸設定 (保留初始縮放，但允許滑動)
            fig_chart.update_xaxes(type='category', range=[start_idx, total_len-1], nticks=10)
            
            # --- Y 軸縮放邏輯 (自動計算初始可視範圍) ---
            # 計算初始可視範圍內的最佳 Y 軸範圍
            df_view = df.iloc[start_idx:]
            cols_to_check = ['Low', 'High']
            if 'BBL' in df.columns: cols_to_check.append('BBL')
            if 'BBU' in df.columns: cols_to_check.append('BBU')
            if 'MA5' in df.columns: cols_to_check.append('MA5')
            
            valid_cols = [c for c in cols_to_check if c in df_view.columns]
            
            if valid_cols:
                p_min = df_view[valid_cols].min().min()
                p_max = df_view[valid_cols].max().max()
                # 加上 5% 緩衝，避免 K 線頂到天花板
                pad = (p_max - p_min) * 0.05
                # 強制設定初始範圍，解決 K 線扁平問題。同時保留 fixedrange=False，讓使用者可以手動拖曳 Y 軸
                fig_chart.update_yaxes(range=[p_min - pad, p_max + pad], fixedrange=False, side='right', row=1, col=1)
            else:
                fig_chart.update_yaxes(autorange=True, fixedrange=False, side='right', row=1, col=1)
            
            fig_chart.update_layout(
                height=700, 
                title=f"{ticker} 交易訊號詳情", 
                template="plotly_dark", 
                dragmode='pan', 
                hovermode="x unified",
                xaxis_rangeslider_visible=False  # 隱藏下方的 Range Slider
            )
            
            # 重要: 加入 scrollZoom 設定，讓滑鼠滾輪可以縮放，拖曳可以平移
            st.plotly_chart(fig_chart, use_container_width=True, config={'scrollZoom': True})
            
            st.caption("💡 操作提示：預設視角已自動最佳化。若滑動至歷史區間發現 K 線超出畫面，請「雙擊圖表」重置，或按住右側 Y 軸上下拖曳即可調整高度。")

            # --- 交易明細表格 ---
            with st.expander("📋 查看詳細交易紀錄"):
                if not trades_df.empty:
                    cols_map = {
                        'Entry Time': '進場時間', 
                        'Exit Time': '出場時間',
                        'Entry Price': '進場價', 
                        'Exit Price': '出場價', 
                        'PnL': '獲利金額', 
                        'PnL %': '獲利 %', 
                        'Reason': '出場原因'
                    }
                    
                    available_cols = [c for c in cols_map.keys() if c in trades_df.columns]
                    t_df = trades_df[available_cols].rename(columns=cols_map).copy()
                    
                    def color_pnl(val):
                        color = 'green' if val > 0 else 'red'
                        return f'color: {color}'
                    
                    st.dataframe(
                        t_df.style.format({
                            '進場價': '{:.2f}', 
                            '出場價': '{:.2f}', 
                            '獲利金額': '{:.0f}', 
                            '獲利 %': '{:.2%}'
                        }).map(color_pnl, subset=['獲利金額', '獲利 %'])
                    )
                else:
                    st.write("此回測區間內無交易產生。")