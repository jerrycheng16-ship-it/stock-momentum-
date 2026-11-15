from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# 頁面設定
st.set_page_config(page_title="波動度濾網與均線策略回測", layout="wide")
st.title("📈 波動度狀態切換 × 均線策略回測系統")

# --- 側邊欄參數設定 ---
with st.sidebar:
    st.header("⚙️ 策略參數設定")
    ticker = st.text_input(
        "輸入股票或 ETF 代號",
        value="0050.TW",
        help="台股請加 .TW 或 .TWO (例如: 0050.TW, 2330.TW)，美股直接輸入 (例如: SPY, QQQ)",
    )

    timeframe = st.radio(
        "資料頻率（觀察週期）",
        options=["日線 (Daily)", "月線 (Monthly)"],
        index=0,
        horizontal=True,
        help="日線：每日收盤計算指標，次日執行；月線：每月最後交易日收盤計算指標，次月執行。",
    )
    is_monthly = timeframe == "月線 (Monthly)"
    periods_per_year = 12 if is_monthly else 252
    unit_label = "月" if is_monthly else "交易日"

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("開始日期", value=datetime(2015, 1, 1))
    with col2:
        end_date = st.date_input("結束日期", value=datetime.today())

    st.markdown("---")
    st.subheader(f"📐 指標週期設定（單位：{unit_label}）")

    default_vol_win = 12 if is_monthly else 252
    default_ma_win = 3 if is_monthly else 63

    vol_window = st.number_input(
        f"波動度計算週期 ({unit_label})",
        min_value=2 if is_monthly else 5,
        max_value=120 if is_monthly else 1000,
        value=default_vol_win,
        step=1,
        help=f"目前為{timeframe}模式：預設 {default_vol_win} {unit_label}（約 12 個月）。",
    )

    ma_window = st.number_input(
        f"價格均線週期 ({unit_label})",
        min_value=2,
        max_value=60 if is_monthly else 500,
        value=default_ma_win,
        step=1,
        help=f"目前為{timeframe}模式：預設 {default_ma_win} {unit_label}（約 3 個月季線）。",
    )

    st.markdown("---")
    st.subheader("🌊 波動度狀態（趨勢盤）判斷模式")

    vol_mode = st.selectbox(
        "選擇波動度判斷方法",
        options=[
            "1. 原始波動度 (%) vs 固定門檻",
            "2. 標準化波動度 (Z-Score) vs 門檻",
            "3. 波動度與其均線交叉 (波動度均線)",
        ],
        index=0,
        help="標準化波動度可消除不同個股天生波動率高低的差異；波動度均線交叉則觀察波動度本身的趨勢變化。",
    )

    # 根據選擇的模式動態顯示對應參數
    extra_lookback = 0
    if vol_mode == "1. 原始波動度 (%) vs 固定門檻":
        trend_condition = st.radio(
            "趨勢盤判斷條件",
            options=[
                "波動度「低於 (<)」門檻為趨勢盤",
                "波動度「高於 (>)」門檻為趨勢盤",
            ],
            index=0,
        )
        is_low_vol_trend = trend_condition == "波動度「低於 (<)」門檻為趨勢盤"
        vol_threshold = (
            st.slider(
                "年化波動度門檻 (%)",
                min_value=5.0,
                max_value=80.0,
                value=20.0,
                step=0.5,
            )
            / 100.0
        )
        op_symbol = "<" if is_low_vol_trend else ">"
        regime_desc = f"原始波動度 {op_symbol} {vol_threshold*100:.1f}%"

    elif vol_mode == "2. 標準化波動度 (Z-Score) vs 門檻":
        z_window = st.number_input(
            f"標準化滾動基準週期 ({unit_label})",
            min_value=5 if is_monthly else 20,
            max_value=120 if is_monthly else 1000,
            value=12 if is_monthly else 252,
            step=1,
            help="用來計算波動度平均值與標準差的滾動窗口，讓高波動股與低波動 ETF 站在同一基準比較。",
        )
        extra_lookback = z_window
        trend_condition = st.radio(
            "趨勢盤判斷條件",
            options=[
                "標準化波動度「低於 (<)」門檻為趨勢盤",
                "標準化波動度「高於 (>)」門檻為趨勢盤",
            ],
            index=0,
        )
        is_low_vol_trend = (
            trend_condition == "標準化波動度「低於 (<)」門檻為趨勢盤"
        )
        z_threshold = st.slider(
            "標準化波動度門檻 (Z-Score 標準差倍數)",
            min_value=-2.5,
            max_value=2.5,
            value=0.0,
            step=0.1,
            help="0.0 代表處於歷史平均水準；負值代表比平時平靜，正值代表比平時劇烈。",
        )
        op_symbol = "<" if is_low_vol_trend else ">"
        regime_desc = f"標準化波動度 {op_symbol} {z_threshold:.1f}σ"

    else:  # 3. 波動度與其均線交叉
        vol_ma_window = st.number_input(
            f"波動度均線週期 ({unit_label})",
            min_value=2,
            max_value=60 if is_monthly else 500,
            value=3 if is_monthly else 63,
            step=1,
            help="計算波動度曲線本身的移動平均線，用來判斷波動度正在升溫還是降溫。",
        )
        extra_lookback = vol_ma_window
        trend_condition = st.radio(
            "趨勢盤判斷條件",
            options=[
                "波動度「上穿 (> 均線)」為趨勢盤",
                "波動度「下破 (< 均線)」為趨勢盤",
            ],
            index=0,
            help="上穿均線為趨勢盤：波動放大時啟動均線保護；下破均線為趨勢盤：波動收斂回穩時啟動均線保護。",
        )
        is_low_vol_trend = (
            trend_condition == "波動度「下破 (< 均線)」為趨勢盤"
        )
        op_symbol = "<" if is_low_vol_trend else ">"
        regime_desc = f"波動度 {op_symbol} 波動度{vol_ma_window}{unit_label}均線"

    st.markdown("---")
    st.subheader("🎯 盤勢動作設定")

    breakdown_action = st.radio(
        "趨勢盤跌破價格均線時的動作",
        options=[
            "平倉空手 (部位 = 0)",
            "反向做空 (部位 = -1)",
        ],
        index=0,
        help="平倉空手：跌破均線時持有現金（報酬 0%）；反向做空：跌破均線時放空標的（股價下跌時獲利）。",
    )
    short_on_breakdown = breakdown_action == "反向做空 (部位 = -1)"
    breakdown_text = "反向做空 (-1)" if short_on_breakdown else "平倉空手 (0)"

    consolidation_action = st.radio(
        "盤整盤時的交易動作",
        options=[
            "維持買入部位 (部位 = +1)",
            "維持前一日部位 (不動作)"
            if not is_monthly
            else "維持前一月部位 (不動作)",
            "全部平倉空手 (部位 = 0)",
        ],
        index=0,
        help="維持買入部位：忽略均線、無條件做多；維持前期部位：沿用進入盤整盤前的多空/空手狀態；全部平倉空手：盤整期間退出觀望。",
    )

    st.markdown("---")
    exec_delay_text = "次月" if is_monthly else "次一交易日"
    st.markdown(f"""
    **目前交易邏輯（{timeframe}）：**
    * **趨勢盤 (`{regime_desc}`)**：收盤價 > `{ma_window}` {unit_label}均線買入 (`+1`)，跌破則**{breakdown_text}**。
    * **盤整盤（非趨勢盤時）**：**{consolidation_action}**。
    * **執行延遲**：當期收盤確認訊號，**{exec_delay_text}**生效。
    """)


# --- 數據獲取與快取 ---
@st.cache_data(ttl=3600)
def load_data(symbol, start, end, lookback_periods, monthly_mode):
    if monthly_mode:
        buffer_calendar_days = int(lookback_periods * 32) + 90
    else:
        buffer_calendar_days = int(lookback_periods * 1.6) + 30

    fetch_start = start - timedelta(days=buffer_calendar_days)
    df = yf.download(symbol, start=fetch_start, end=end, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if monthly_mode:
        try:
            df = df.resample("ME").last()
        except ValueError:
            df = df.resample("M").last()
        df = df.dropna(subset=["Close"])

    return df


# --- 績效指標計算函數 ---
def calculate_metrics(returns_series, position_series=None, ann_factor=252):
    cum_returns = (1 + returns_series).cumprod()
    total_return = cum_returns.iloc[-1] - 1

    days = (returns_series.index[-1] - returns_series.index[0]).days
    years = max(days / 365.25, 1 / ann_factor)
    cagr = (1 + total_return) ** (1 / years) - 1

    ann_vol = returns_series.std() * np.sqrt(ann_factor)

    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    max_dd = drawdown.min()

    if position_series is not None:
        active_periods = returns_series[position_series != 0]
        period_win_rate = (
            (active_periods > 0).mean() if len(active_periods) > 0 else 0.0
        )

        trade_groups = (position_series != position_series.shift()).cumsum()
        trade_returns = []
        for _, grp_pos in position_series.groupby(trade_groups):
            pos_val = grp_pos.iloc[0]
            if pos_val != 0:
                grp_ret = returns_series.loc[grp_pos.index]
                t_ret = (1 + grp_ret).prod() - 1
                trade_returns.append(t_ret)

        trade_win_rate = (
            np.mean([r > 0 for r in trade_returns])
            if len(trade_returns) > 0
            else 0.0
        )
        num_trades = len(trade_returns)
    else:
        period_win_rate = (returns_series > 0).mean()
        trade_win_rate = 1.0 if total_return > 0 else 0.0
        num_trades = 1

    return {
        "總報酬率": total_return,
        "年化報酬率": cagr,
        "年化波動度": ann_vol,
        "最大回撤 (MDD)": max_dd,
        "持倉期勝率": period_win_rate,
        "波段交易勝率": trade_win_rate,
        "總交易次數": num_trades,
    }


# --- 主程式執行 ---
if ticker:
    total_lookback = max(vol_window + extra_lookback, ma_window)
    raw_df = load_data(ticker, start_date, end_date, total_lookback, is_monthly)

    if raw_df.empty:
        st.error(f"找不到代號 `{ticker}` 的資料，請確認股票代號與日期範圍是否正確。")
    else:
        df = raw_df[["Close"]].copy()
        df["Ret"] = df["Close"].pct_change()

        # 1. 計算基礎年化波動度與價格均線
        df["Vol"] = df["Ret"].rolling(window=int(vol_window)).std() * np.sqrt(
            periods_per_year
        )
        df["MA"] = df["Close"].rolling(window=int(ma_window)).mean()

        # 2. 根據選擇的波動度模式計算衍生指標與趨勢盤狀態
        if vol_mode == "1. 原始波動度 (%) vs 固定門檻":
            if is_low_vol_trend:
                df["Is_Trend"] = df["Vol"] < vol_threshold
            else:
                df["Is_Trend"] = df["Vol"] > vol_threshold

        elif vol_mode == "2. 標準化波動度 (Z-Score) vs 門檻":
            vol_mean = df["Vol"].rolling(window=int(z_window)).mean()
            vol_std = df["Vol"].rolling(window=int(z_window)).std()
            df["Vol_Z"] = (df["Vol"] - vol_mean) / vol_std.replace(0, np.nan)
            if is_low_vol_trend:
                df["Is_Trend"] = df["Vol_Z"] < z_threshold
            else:
                df["Is_Trend"] = df["Vol_Z"] > z_threshold

        else:  # 3. 波動度與其均線交叉
            df["Vol_MA"] = df["Vol"].rolling(window=int(vol_ma_window)).mean()
            if is_low_vol_trend:
                df["Is_Trend"] = df["Vol"] < df["Vol_MA"]
            else:
                df["Is_Trend"] = df["Vol"] > df["Vol_MA"]

        # 過濾至使用者指定的回測起始日並清除前置 NaN
        df = df[df.index >= pd.to_datetime(start_date)].dropna().copy()

        min_required_bars = 5 if is_monthly else 10
        if len(df) < min_required_bars:
            st.warning("有效回測資料過少，請將開始日期往前調整，或縮短指標計算週期。")
        else:
            # 3. 趨勢盤與盤整盤交易訊號
            below_ma_pos = -1 if short_on_breakdown else 0
            trend_signal = np.where(df["Close"] > df["MA"], 1, below_ma_pos)

            if consolidation_action == "維持買入部位 (部位 = +1)":
                df["Signal"] = np.where(df["Is_Trend"], trend_signal, 1)
            elif consolidation_action == "全部平倉空手 (部位 = 0)":
                df["Signal"] = np.where(df["Is_Trend"], trend_signal, 0)
            else:
                raw_sig = pd.Series(
                    np.where(df["Is_Trend"], trend_signal, np.nan),
                    index=df.index,
                )
                df["Signal"] = raw_sig.ffill().fillna(1)

            # 實際部位：上期收盤確認訊號，本期持有
            df["Position"] = df["Signal"].shift(1).fillna(df["Signal"].iloc[0])

            # 偵測部位切換點
            df["Pos_Change"] = df["Position"].diff()
            df.iloc[0, df.columns.get_loc("Pos_Change")] = df["Position"].iloc[0]

            buy_signals = df[(df["Pos_Change"] != 0) & (df["Position"] == 1)]
            flat_signals = df[(df["Pos_Change"] != 0) & (df["Position"] == 0)]
            short_signals = df[(df["Pos_Change"] != 0) & (df["Position"] == -1)]

            # 4. 計算策略與 Buy & Hold 報酬與淨值
            df["Strat_Ret"] = df["Position"] * df["Ret"]
            df["BH_Ret"] = df["Ret"]

            df["Strat_Equity"] = (1 + df["Strat_Ret"]).cumprod()
            df["BH_Equity"] = (1 + df["BH_Ret"]).cumprod()

            df["Strat_DD"] = (
                df["Strat_Equity"] - df["Strat_Equity"].cummax()
            ) / df["Strat_Equity"].cummax()
            df["BH_DD"] = (
                df["BH_Equity"] - df["BH_Equity"].cummax()
            ) / df["BH_Equity"].cummax()

            # 5. 計算績效統計表
            strat_metrics = calculate_metrics(
                df["Strat_Ret"], df["Position"], periods_per_year
            )
            bh_metrics = calculate_metrics(df["BH_Ret"], None, periods_per_year)

            st.subheader(
                f"📊 {ticker.upper()} 績效指標比較【{timeframe}】 ({regime_desc} 為趨勢盤)"
            )

            period_win_label = "持倉月勝率" if is_monthly else "持倉日勝率"

            metrics_df = pd.DataFrame(
                {
                    "策略表現": [
                        f"{strat_metrics['總報酬率']*100:.2f}%",
                        f"{strat_metrics['年化報酬率']*100:.2f}%",
                        f"{strat_metrics['年化波動度']*100:.2f}%",
                        f"{strat_metrics['最大回撤 (MDD)']*100:.2f}%",
                        f"{strat_metrics['波段交易勝率']*100:.2f}% ({strat_metrics['總交易次數']}次)",
                        f"{strat_metrics['持倉期勝率']*100:.2f}%",
                    ],
                    "Buy & Hold": [
                        f"{bh_metrics['總報酬率']*100:.2f}%",
                        f"{bh_metrics['年化報酬率']*100:.2f}%",
                        f"{bh_metrics['年化波動度']*100:.2f}%",
                        f"{bh_metrics['最大回撤 (MDD)']*100:.2f}%",
                        "100.00% (1次)"
                        if bh_metrics["總報酬率"] > 0
                        else "0.00% (1次)",
                        f"{bh_metrics['持倉期勝率']*100:.2f}%",
                    ],
                },
                index=[
                    "總報酬率",
                    "年化報酬率 (CAGR)",
                    "年化波動度",
                    "最大回撤 (MDD)",
                    "波段交易勝率 (交易次數)",
                    period_win_label,
                ],
            )

            st.dataframe(metrics_df, use_container_width=True)

            # 6. 繪製 4 層互動式圖表
            st.subheader(f"📈 {timeframe}價格均線訊號、淨值走勢、回撤與盤勢分析")

            if vol_mode == "1. 原始波動度 (%) vs 固定門檻":
                vol_subplot_title = f"{vol_window}{unit_label}滾動年化波動度 (%) vs 門檻 ({vol_threshold*100:.1f}%)"
                vol_y_title = "波動度 (%)"
            elif vol_mode == "2. 標準化波動度 (Z-Score) vs 門檻":
                vol_subplot_title = f"{vol_window}{unit_label}標準化波動度 (Z-Score, {z_window}{unit_label}基準) vs 門檻 ({z_threshold:.1f}σ)"
                vol_y_title = "標準差 (σ)"
            else:
                vol_subplot_title = f"{vol_window}{unit_label}年化波動度 (%) vs 波動度 {vol_ma_window}{unit_label}均線"
                vol_y_title = "波動度 (%)"

            fig = make_subplots(
                rows=4,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.32, 0.32, 0.18, 0.18],
                subplot_titles=(
                    f"{ticker.upper
