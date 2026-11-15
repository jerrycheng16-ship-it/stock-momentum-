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

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("開始日期", value=datetime(2015, 1, 1))
    with col2:
        end_date = st.date_input("結束日期", value=datetime.today())

    st.markdown("---")
    st.subheader("📐 指標天數與門檻設定")

    vol_window = st.number_input(
        "波動度計算天數 (交易日)",
        min_value=5,
        max_value=1000,
        value=252,
        step=1,
        help="預設 252 個交易日（約 12 個月），63 天約為 3 個月，21 天約為 1 個月。",
    )

    ma_window = st.number_input(
        "均線計算天數 (交易日)",
        min_value=2,
        max_value=500,
        value=63,
        step=1,
        help="預設 63 個交易日（約 3 個月季線），20 天約為月線，200 天為年線。",
    )

    trend_condition = st.radio(
        "趨勢盤判斷條件（啟用均線濾網的時機）",
        options=[
            "波動度「低於 (<)」門檻為趨勢盤",
            "波動度「高於 (>)」門檻為趨勢盤",
        ],
        index=0,
        help="趨勢盤時會依據均線操作；盤整盤時則無條件維持做多持有。",
    )
    is_low_vol_trend = trend_condition == "波動度「低於 (<)」門檻為趨勢盤"
    op_symbol = "<" if is_low_vol_trend else ">"
    opposite_symbol = ">=" if is_low_vol_trend else "<="

    vol_threshold = (
        st.slider(
            "波動度門檻 (%)",
            min_value=5.0,
            max_value=60.0,
            value=20.0,
            step=0.5,
        )
        / 100.0
    )

    breakdown_action = st.radio(
        "趨勢盤跌破均線時的動作",
        options=[
            "平倉空手 (部位 = 0)",
            "反向做空 (部位 = -1)",
        ],
        index=0,
        help="平倉空手：跌破均線時持有現金（報酬 0%）；反向做空：跌破均線時放空標的（股價下跌時獲利）。",
    )
    short_on_breakdown = breakdown_action == "反向做空 (部位 = -1)"
    breakdown_text = "反向做空" if short_on_breakdown else "平倉賣出"

    st.markdown("---")
    st.markdown(f"""
    **目前交易邏輯：**
    * **波動度計算**：過去 `{vol_window}` 個交易日之年化波動度。
    * **趨勢盤 (`波動度 {op_symbol} {vol_threshold*100:.1f}%`)**：股價 > `{ma_window}` 日均線買入，跌破則**{breakdown_text}**。
    * **盤整盤 (`波動度 {opposite_symbol} {vol_threshold*100:.1f}%`)**：維持買入持有（做多）。
    * **執行延遲**：當日收盤確認訊號，次一交易日生效。
    """)


# --- 數據獲取與快取 ---
@st.cache_data(ttl=3600)
def load_data(symbol, start, end, lookback_days):
    buffer_calendar_days = int(lookback_days * 1.6) + 30
    fetch_start = start - timedelta(days=buffer_calendar_days)
    df = yf.download(symbol, start=fetch_start, end=end, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# --- 績效指標計算函數（支援多頭、空手與反向做空波段計算） ---
def calculate_metrics(returns_series, position_series=None):
    cum_returns = (1 + returns_series).cumprod()
    total_return = cum_returns.iloc[-1] - 1

    days = (returns_series.index[-1] - returns_series.index[0]).days
    years = max(days / 365.25, 1 / 252)
    cagr = (1 + total_return) ** (1 / years) - 1

    ann_vol = returns_series.std() * np.sqrt(252)

    # 計算回撤
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # 勝率計算
    if position_series is not None:
        # 持有部位（做多或做空）期間的日勝率
        active_days = returns_series[position_series != 0]
        daily_win_rate = (
            (active_days > 0).mean() if len(active_days) > 0 else 0.0
        )

        # 依照連續相同部位（+1 多單波段 / -1 空單波段）切分每一筆完整交易
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
        daily_win_rate = (returns_series > 0).mean()
        trade_win_rate = 1.0 if total_return > 0 else 0.0
        num_trades = 1

    return {
        "總報酬率": total_return,
        "年化報酬率": cagr,
        "年化波動度": ann_vol,
        "最大回撤 (MDD)": max_dd,
        "交易日勝率": daily_win_rate,
        "波段交易勝率": trade_win_rate,
        "總交易次數": num_trades,
    }


# --- 主程式執行 ---
if ticker:
    max_lookback = max(vol_window, ma_window)
    raw_df = load_data(ticker, start_date, end_date, max_lookback)

    if raw_df.empty:
        st.error(f"找不到代號 `{ticker}` 的資料，請確認股票代號與日期範圍是否正確。")
    else:
        df = raw_df[["Close"]].copy()
        df["Ret"] = df["Close"].pct_change()

        # 1. 動態計算自訂天數的波動度與均線
        df["Vol"] = df["Ret"].rolling(window=int(vol_window)).std() * np.sqrt(252)
        df["MA"] = df["Close"].rolling(window=int(ma_window)).mean()

        # 過濾至使用者指定的回測起始日
        df = df[df.index >= pd.to_datetime(start_date)].dropna().copy()

        if len(df) < 10:
            st.warning("有效回測資料過少，請將開始日期往前調整，或縮短指標計算天數。")
        else:
            # 2. 依據使用者選擇的方向判斷趨勢盤
            if is_low_vol_trend:
                df["Is_Trend"] = df["Vol"] < vol_threshold
            else:
                df["Is_Trend"] = df["Vol"] > vol_threshold

            # 跌破均線時的目標部位：空手為 0，做空為 -1
            below_ma_pos = -1 if short_on_breakdown else 0

            # 訊號邏輯：趨勢盤看均線 (Close > MA -> 1 else below_ma_pos)，盤整盤維持買入 (1)
            df["Signal"] = np.where(
                df["Is_Trend"], np.where(df["Close"] > df["MA"], 1, below_ma_pos), 1
            )

            # 實際部位：昨日收盤確認訊號，今日持有 (避免前視偏誤)
            df["Position"] = df["Signal"].shift(1).fillna(1)

            # 3. 計算策略與 Buy & Hold 報酬與淨值
            df["Strat_Ret"] = df["Position"] * df["Ret"]
            df["BH_Ret"] = df["Ret"]

            df["Strat_Equity"] = (1 + df["Strat_Ret"]).cumprod()
            df["BH_Equity"] = (1 + df["BH_Ret"]).cumprod()

            # 計算 Drawdown
            df["Strat_DD"] = (
                df["Strat_Equity"] - df["Strat_Equity"].cummax()
            ) / df["Strat_Equity"].cummax()
            df["BH_DD"] = (
                df["BH_Equity"] - df["BH_Equity"].cummax()
            ) / df["BH_Equity"].cummax()

            # 4. 計算績效統計表
            strat_metrics = calculate_metrics(df["Strat_Ret"], df["Position"])
            bh_metrics = calculate_metrics(df["BH_Ret"], None)

            st.subheader(
                f"📊 {ticker.upper()} 績效指標比較 (波動度 {op_symbol} {vol_threshold*100:.1f}% 為趨勢盤｜跌破均線：{breakdown_text})"
            )

            metrics_df = pd.DataFrame(
                {
                    "策略表現": [
                        f"{strat_metrics['總報酬率']*100:.2f}%",
                        f"{strat_metrics['年化報酬率']*100:.2f}%",
                        f"{strat_metrics['年化波動度']*100:.2f}%",
                        f"{strat_metrics['最大回撤 (MDD)']*100:.2f}%",
                        f"{strat_metrics['波段交易勝率']*100:.2f}% ({strat_metrics['總交易次數']}次)",
                        f"{strat_metrics['交易日勝率']*100:.2f}%",
                    ],
                    "Buy & Hold": [
                        f"{bh_metrics['總報酬率']*100:.2f}%",
                        f"{bh_metrics['年化報酬率']*100:.2f}%",
                        f"{bh_metrics['年化波動度']*100:.2f}%",
                        f"{bh_metrics['最大回撤 (MDD)']*100:.2f}%",
                        "100.00% (1次)"
                        if bh_metrics["總報酬率"] > 0
                        else "0.00% (1次)",
                        f"{bh_metrics['交易日勝率']*100:.2f}%",
                    ],
                },
                index=[
                    "總報酬率",
                    "年化報酬率 (CAGR)",
                    "年化波動度",
                    "最大回撤 (MDD)",
                    "波段交易勝率 (交易次數)",
                    "持倉日勝率",
                ],
            )

            st.dataframe(metrics_df, use_container_width=True)

            # 5. 繪製互動式圖表
            st.subheader("📈 淨值走勢、回撤 (Drawdown) 與盤勢狀態分析")

            fig = make_subplots(
                rows=3,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.06,
                row_heights=[0.5, 0.25, 0.25],
                subplot_titles=(
                    f"累積淨值走勢圖 (背景色：淺藍=趨勢盤 [波動度{op_symbol}{vol_threshold*100:.1f}%] / 淺橘=盤整盤)",
                    "回撤走勢圖 (Drawdown %)",
                    f"{vol_window}日滾動年化波動度 (%)",
                ),
            )

            # 淨值曲線
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["Strat_Equity"],
                    name="波動度濾網策略",
                    line=dict(color="#1f77b4", width=2.2),
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["BH_Equity"],
                    name="Buy & Hold",
                    line=dict(color="#7f7f7f", width=1.8, dash="dot"),
                ),
                row=1,
                col=1,
            )

            # Drawdown 曲線
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["Strat_DD"] * 100,
                    name="策略 Drawdown",
                    fill="tozeroy",
                    line=dict(color="#1f77b4", width=1.2),
                    fillcolor="rgba(31, 119, 180, 0.2)",
                ),
                row=2,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["BH_DD"] * 100,
                    name="Buy & Hold Drawdown",
                    line=dict(color="#d62728", width=1.2),
                ),
                row=2,
                col=1,
            )

            # 波動度與門檻線
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["Vol"] * 100,
                    name=f"{vol_window}日年化波動度",
                    line=dict(color="#9467bd", width=1.5),
                ),
                row=3,
                col=1,
            )

            fig.add_hline(
                y=vol_threshold * 100,
                line_dash="dash",
                line_color="red",
                annotation_text=f"門檻 ({vol_threshold*100:.1f}%)",
                row=3,
                col=1,
            )

            # 標示趨勢盤與盤整盤區間（顏色定義維持一致：淺藍=趨勢盤、淺橘=盤整盤）
            df["Regime_Group"] = (
                df["Is_Trend"] != df["Is_Trend"].shift()
            ).cumsum()
            for _, group in df.groupby("Regime_Group"):
                is_trend = group["Is_Trend"].iloc[0]
                x0 = group.index[0]
                x1 = group.index[-1]
                fill_color = (
                    "rgba(100, 181, 246, 0.18)"
                    if is_trend
                    else "rgba(255, 167, 38, 0.18)"
                )

                fig.add_vrect(
                    x0=x0,
                    x1=x1,
                    fillcolor=fill_color,
                    layer="below",
                    line_width=0,
                    row=1,
                    col=1,
                )

            # 圖表版面配置
            fig.update_layout(
                height=800,
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                ),
                margin=dict(l=20, r=20, t=60, b=20),
            )
            fig.update_yaxes(title_text="淨值 (倍數)", row=1, col=1)
            fig.update_yaxes(title_text="回撤 (%)", row=2, col=1)
            fig.update_yaxes(title_text="波動度 (%)", row=3, col=1)

            st.plotly_chart(fig, use_container_width=True)

            # 6. 盤勢與倉位狀態統計摘要
            trend_pct = df["Is_Trend"].mean() * 100
            long_pct = (df["Position"] == 1).mean() * 100
            flat_pct = (df["Position"] == 0).mean() * 100
            short_pct = (df["Position"] == -1).mean() * 100

            st.info(
                f"💡 **區間與倉位統計**：回測期間內**趨勢盤**佔比 `{trend_pct:.1f}%`、**盤整盤**佔比 `{100-trend_pct:.1f}%`｜"
                f"倉位時間分佈：**做多** `{long_pct:.1f}%`、**空手** `{flat_pct:.1f}%`、**做空** `{short_pct:.1f}%`。"
            )
