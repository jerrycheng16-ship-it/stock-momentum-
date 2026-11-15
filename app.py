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

# 內建常用隱含波動度指數清單
IV_OPTIONS = {
    "^VIX (S&P 500 恐慌指數 - 美股大盤)": "^VIX",
    "^VXN (Nasdaq 100 波動率 - 科技/半導體)": "^VXN",
    "^RVX (Russell 2000 波動率 - 中小型股)": "^RVX",
    "^VXD (道瓊工業波動率 - 藍籌傳產)": "^VXD",
    "^VVIX (VIX 的波動率 - 尾部風險預警)": "^VVIX",
    "^VXAPL (Apple 蘋果隱含波動率)": "^VXAPL",
    "^VXAZN (Amazon 亞馬遜隱含波動率)": "^VXAZN",
    "^VXGOG (Google 隱含波動率)": "^VXGOG",
    "^OVX (原油波動率指數 - 能源/通膨)": "^OVX",
    "^GVZ (黃金波動率指數 - 避險貴金屬)": "^GVZ",
    "自訂輸入代號...": "CUSTOM",
}

# --- 側邊欄參數設定 ---
with st.sidebar:
    st.header("⚙️ 策略參數設定")
    ticker = st.text_input(
        "輸入交易標的代號 (股票或 ETF)",
        value="SPY",
        help="台股請加 .TW 或 .TWO (例如: 0050.TW, 2330.TW)，美股直接輸入 (例如: SPY, QQQ, SMH)",
    )

    vol_source = st.radio(
        "波動度資料來源",
        options=[
            "標的自身歷史波動度 (HV)",
            "外部隱含波動度指數 (IV Index)",
        ],
        index=1,
        help="HV：依據標的自身過去漲跌幅計算；IV Index：抓取市場恐慌/隱含波動率指數作為領先情緒濾網。",
    )
    use_external_iv = vol_source == "外部隱含波動度指數 (IV Index)"

    iv_ticker = "^VIX"
    if use_external_iv:
        selected_iv_label = st.selectbox(
            "選擇隱含波動度指數",
            options=list(IV_OPTIONS.keys()),
            index=0,
            help="直接從下拉選單選擇對應市場的 30 天期隱含波動率指數，無需手動輸入。",
        )
        if IV_OPTIONS[selected_iv_label] == "CUSTOM":
            iv_ticker = st.text_input("請輸入自訂波動率指數代號", value="^VIX")
        else:
            iv_ticker = IV_OPTIONS[selected_iv_label]

    timeframe = st.radio(
        "資料頻率（觀察週期）",
        options=["日線 (Daily)", "月線 (Monthly)"],
        index=1,
        horizontal=True,
        help="日線：每日收盤計算指標，次日執行；月線：每月最後交易日收盤計算指標，次月執行。",
    )
    is_monthly = timeframe == "月線 (Monthly)"
    periods_per_year = 12 if is_monthly else 252
    unit_label = "月" if is_monthly else "交易日"

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("開始日期", value=datetime(2006, 1, 1))
    with col2:
        end_date = st.date_input("結束日期", value=datetime.today())

    st.markdown("---")
    st.subheader(f"📐 指標週期設定（單位：{unit_label}）")

    default_vol_win = 12 if is_monthly else 252
    default_ma_win = 6 if is_monthly else 126

    if not use_external_iv:
        vol_window = st.number_input(
            f"歷史波動度計算週期 ({unit_label})",
            min_value=2 if is_monthly else 5,
            max_value=120 if is_monthly else 1000,
            value=default_vol_win,
            step=1,
            help=f"目前為{timeframe}模式：預設 {default_vol_win} {unit_label}（約 12 個月）。",
        )
    else:
        vol_window = 0
        st.info(
            f"ℹ️ 目前直接採用 `{iv_ticker.upper()}` 的收盤數值作為當期年化隱含波動度 (%)。"
        )

    ma_window = st.number_input(
        f"價格均線週期 ({unit_label})",
        min_value=2,
        max_value=60 if is_monthly else 500,
        value=default_ma_win,
        step=1,
        help=f"目前為{timeframe}模式：預設 {default_ma_win} {unit_label}（半年線）。",
    )

    st.markdown("---")
    st.subheader("🌊 波動度狀態（趨勢盤）判斷模式")

    vol_mode = st.selectbox(
        "選擇波動度判斷方法",
        options=[
            "1. 原始波動度 (%) vs 固定門檻",
            "2. 標準化波動度 (Z-Score) vs 門檻",
            "3. 波動度與其均線交叉 (波動度均線)",
            "4. 標準化波動度 + 均線交叉雙重過濾 (高低檔位觸發)",
        ],
        index=3,
        help="模式 4 結合標準化波動度高低檔位與均線突破/跌破，採用狀態記憶機制避免頻繁切換。",
    )

    vol_label_prefix = (
        f"{iv_ticker.upper()} 隱含波動度" if use_external_iv else "歷史波動度"
    )
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
        regime_desc = f"{vol_label_prefix} {op_symbol} {vol_threshold*100:.1f}%"

    elif vol_mode == "2. 標準化波動度 (Z-Score) vs 門檻":
        z_window = st.number_input(
            f"標準化滾動基準週期 ({unit_label})",
            min_value=5 if is_monthly else 20,
            max_value=120 if is_monthly else 1000,
            value=12 if is_monthly else 252,
            step=1,
            help="用來計算波動度平均值與標準差的滾動窗口。",
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
        )
        op_symbol = "<" if is_low_vol_trend else ">"
        regime_desc = f"{vol_label_prefix} Z-Score {op_symbol} {z_threshold:.1f}σ"

    elif vol_mode == "3. 波動度與其均線交叉 (波動度均線)":
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
        )
        is_low_vol_trend = (
            trend_condition == "波動度「下破 (< 均線)」為趨勢盤"
        )
        op_symbol = "<" if is_low_vol_trend else ">"
        regime_desc = (
            f"{vol_label_prefix} {op_symbol} {vol_ma_window}{unit_label}均線"
        )

    else:  # 4. 標準化波動度 + 均線交叉雙重過濾
        z_window = st.number_input(
            f"標準化滾動基準週期 ({unit_label})",
            min_value=5 if is_monthly else 20,
            max_value=120 if is_monthly else 1000,
            value=12 if is_monthly else 252,
            step=1,
        )
        z_ma_window = st.number_input(
            f"標準化波動度均線週期 ({unit_label})",
            min_value=2,
            max_value=60 if is_monthly else 500,
            value=3 if is_monthly else 63,
            step=1,
        )
        extra_lookback = z_window + z_ma_window

        z_low_thresh = st.slider(
            "低檔水準門檻 (Z-Score 下界 σ)",
            min_value=-2.5,
            max_value=1.0,
            value=-0.5,
            step=0.1,
        )
        z_high_thresh = st.slider(
            "高檔水準門檻 (Z-Score 上界 σ)",
            min_value=-1.0,
            max_value=2.5,
            value=0.5,
            step=0.1,
        )

        dual_direction = st.radio(
            "狀態切換方向設定",
            options=[
                "低檔上穿均線 → 趨勢盤｜高檔下破均線 → 盤整盤",
                "高檔下破均線 → 趨勢盤｜低檔上穿均線 → 盤整盤",
            ],
            index=1,
        )
        is_low_cross_up_trend = (
            dual_direction
            == "低檔上穿均線 → 趨勢盤｜高檔下破均線 → 盤整盤"
        )
        regime_desc = (
            f"Z<{z_low_thresh:.1f}σ上穿進趨勢 / Z>{z_high_thresh:.1f}σ下破進盤整"
            if is_low_cross_up_trend
            else f"Z>{z_high_thresh:.1f}σ下破進趨勢 / Z<{z_low_thresh:.1f}σ上穿進盤整"
        )

    st.markdown("---")
    st.subheader("🎯 盤勢動作設定")

    breakdown_action = st.radio(
        "趨勢盤跌破價格均線時的動作",
        options=[
            "平倉空手 (部位 = 0)",
            "反向做空 (部位 = -1)",
        ],
        index=0,
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
    fetch_end = end + timedelta(days=1)
    df = yf.download(symbol, start=fetch_start, end=fetch_end, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if monthly_mode:
        df = df.dropna(subset=["Close"])
        df = df.groupby(df.index.to_period("M")).tail(1)

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


def format_pos_label(pos_val):
    if pos_val == 1:
        return "買入做多 (+1)"
    elif pos_val == -1:
        return "反向做空 (-1)"
    else:
        return "平倉空手 (0)"


# --- 主程式執行 ---
if ticker:
    total_lookback = max(vol_window + extra_lookback, ma_window)
    raw_df = load_data(ticker, start_date, end_date, total_lookback, is_monthly)

    if raw_df.empty:
        st.error(f"找不到交易標的 `{ticker}` 的資料，請確認代號與日期範圍是否正確。")
    else:
        df = raw_df[["Close"]].copy()
        df["Ret"] = df["Close"].pct_change()
        df["MA"] = df["Close"].rolling(window=int(ma_window)).mean()

        # 1. 取得波動度序列（標的自身 HV 或外部 IV 指數）
        iv_load_failed = False
        if use_external_iv:
            iv_raw = load_data(
                iv_ticker, start_date, end_date, total_lookback, is_monthly
            )
            if iv_raw.empty:
                st.error(
                    f"找不到隱含波動度指數 `{iv_ticker}` 的資料，請確認代號是否正確（例如 ^VIX, ^VXN）。"
                )
                iv_load_failed = True
            else:
                if is_monthly:
                    # 修正：使用 pd.Series 包裝 map 結果後再呼叫 ffill()，相容 Pandas 2.0+
                    iv_monthly_map = (
                        (iv_raw["Close"] / 100.0)
                        .groupby(iv_raw.index.to_period("M"))
                        .last()
                    )
                    mapped_iv = pd.Series(
                        df.index.to_period("M").map(iv_monthly_map),
                        index=df.index,
                        dtype=float,
                    )
                    df["Vol"] = mapped_iv.ffill()
                else:
                    iv_series = (
                        (iv_raw["Close"] / 100.0).reindex(df.index).ffill()
                    )
                    df["Vol"] = iv_series
        else:
            df["Vol"] = df["Ret"].rolling(window=int(vol_window)).std() * np.sqrt(
                periods_per_year
            )

        if not iv_load_failed:
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

            elif vol_mode == "3. 波動度與其均線交叉 (波動度均線)":
                df["Vol_MA"] = df["Vol"].rolling(window=int(vol_ma_window)).mean()
                if is_low_vol_trend:
                    df["Is_Trend"] = df["Vol"] < df["Vol_MA"]
                else:
                    df["Is_Trend"] = df["Vol"] > df["Vol_MA"]

            else:  # 4. 標準化波動度 + 均線交叉雙重過濾（狀態機）
                vol_mean = df["Vol"].rolling(window=int(z_window)).mean()
                vol_std = df["Vol"].rolling(window=int(z_window)).std()
                df["Vol_Z"] = (df["Vol"] - vol_mean) / vol_std.replace(0, np.nan)
                df["Vol_Z_MA"] = (
                    df["Vol_Z"].rolling(window=int(z_ma_window)).mean()
                )

                prev_z = df["Vol_Z"].shift(1)
                prev_z_ma = df["Vol_Z_MA"].shift(1)

                cross_up = (prev_z <= prev_z_ma) & (df["Vol_Z"] > df["Vol_Z_MA"])
                cross_down = (prev_z >= prev_z_ma) & (df["Vol_Z"] < df["Vol_Z_MA"])

                cond_low_cross_up = (df["Vol_Z"] < z_low_thresh) & cross_up
                cond_high_cross_down = (df["Vol_Z"] > z_high_thresh) & cross_down

                regime_event = pd.Series(np.nan, index=df.index, dtype=float)
                if is_low_cross_up_trend:
                    regime_event[cond_low_cross_up] = 1.0
                    regime_event[cond_high_cross_down] = 0.0
                else:
                    regime_event[cond_high_cross_down] = 1.0
                    regime_event[cond_low_cross_up] = 0.0

                df["Is_Trend"] = regime_event.ffill().fillna(1.0).astype(bool)

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
                df.iloc[0, df.columns.get_loc("Pos_Change")] = df[
                    "Position"
                ].iloc[0]

                buy_signals = df[(df["Pos_Change"] != 0) & (df["Position"] == 1)]
                flat_signals = df[(df["Pos_Change"] != 0) & (df["Position"] == 0)]
                short_signals = df[
                    (df["Pos_Change"] != 0) & (df["Position"] == -1)
                ]

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

                # ==========================================
                # 最新即時訊號看板（訊號計算日 & 應用日）
                # ==========================================
                latest_row = df.iloc[-1]
                calc_date = df.index[-1]
                calc_date_str = calc_date.strftime("%Y-%m-%d")

                if is_monthly:
                    next_period = calc_date.to_period("M") + 1
                    apply_date_str = f"{next_period.year} 年 {next_period.month:02d} 月（次月首個交易日起）"
                else:
                    next_bday = calc_date + pd.tseries.offsets.BDay(1)
                    apply_date_str = f"{next_bday.strftime('%Y-%m-%d')}（次一交易日）"

                latest_sig_val = int(latest_row["Signal"])
                latest_sig_label = format_pos_label(latest_sig_val)
                latest_regime = (
                    "趨勢盤 (均線濾網啟動)"
                    if latest_row["Is_Trend"]
                    else "盤整盤 (執行盤整規則)"
                )

                st.subheader(f"🚦 {ticker.upper()} 最新即時交易訊號")
                sig_col1, sig_col2, sig_col3, sig_col4 = st.columns(4)

                with sig_col1:
                    st.metric(
                        label="📅 訊號計算日（最新收盤）",
                        value=calc_date_str,
                        delta=f"收盤價: {latest_row['Close']:.2f} | {ma_window}{unit_label}線: {latest_row['MA']:.2f}",
                        delta_color="off",
                    )
                with sig_col2:
                    st.metric(
                        label="⏭️ 訊號應用日（生效時間）",
                        value=apply_date_str.split("（")[0],
                        delta=apply_date_str.split("（")[1].rstrip("）"),
                        delta_color="off",
                    )
                with sig_col3:
                    st.metric(
                        label="🌊 最新判定盤勢",
                        value="趨勢盤" if latest_row["Is_Trend"] else "盤整盤",
                        delta=latest_regime,
                        delta_color="off",
                    )
                with sig_col4:
                    st.metric(
                        label="🎯 最新目標部位（應用日執行）",
                        value=latest_sig_label,
                        delta=f"前期實際持有: {format_pos_label(int(latest_row['Position']))}",
                        delta_color="normal"
                        if latest_sig_val == 1
                        else ("inverse" if latest_sig_val == -1 else "off"),
                    )

                st.markdown("---")

                # 5. 計算績效統計表
                strat_metrics = calculate_metrics(
                    df["Strat_Ret"], df["Position"], periods_per_year
                )
                bh_metrics = calculate_metrics(
                    df["BH_Ret"], None, periods_per_year
                )

                st.subheader(
                    f"📊 {ticker.upper()} 績效指標比較【{timeframe}】 ({regime_desc})"
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

                # 6. 繪製 4 層高對比度互動式圖表
                st.subheader(
                    f"📈 {timeframe}價格均線訊號、淨值走勢、回撤與盤勢分析"
                )

                vol_chart_name = (
                    f"{iv_ticker.upper()} 隱含波動度 (%)"
                    if use_external_iv
                    else f"{vol_window}{unit_label}年化波動度 (%)"
                )

                if vol_mode == "1. 原始波動度 (%) vs 固定門檻":
                    vol_subplot_title = (
                        f"{vol_chart_name} vs 門檻 ({vol_threshold*100:.1f}%)"
                    )
                    vol_y_title = "波動度 (%)"
                elif vol_mode == "2. 標準化波動度 (Z-Score) vs 門檻":
                    vol_subplot_title = f"{vol_label_prefix}標準化 (Z-Score, {z_window}{unit_label}基準) vs 門檻 ({z_threshold:.1f}σ)"
                    vol_y_title = "標準差 (σ)"
                elif vol_mode == "3. 波動度與其均線交叉 (波動度均線)":
                    vol_subplot_title = f"{vol_chart_name} vs 波動度 {vol_ma_window}{unit_label}均線"
                    vol_y_title = "波動度 (%)"
                else:
                    vol_subplot_title = (
                        f"{vol_label_prefix} Z-Score vs {z_ma_window}{unit_label}均線 "
                        f"｜高檔 {z_high_thresh:.1f}σ / 低檔 {z_low_thresh:.1f}σ"
                    )
                    vol_y_title = "標準差 (σ)"

                fig = make_subplots(
                    rows=4,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.06,
                    row_heights=[0.32, 0.32, 0.18, 0.18],
                    subplot_titles=(
                        f"{ticker.upper()} {timeframe}收盤價、{ma_window}{unit_label}均線與買賣訊號 (背景：青藍=趨勢盤 / 暖橘=盤整盤)",
                        "累積淨值走勢對比",
                        "回撤走勢圖 (Drawdown %)",
                        vol_subplot_title,
                    ),
                )

                # --- Row 1: 高亮度收盤價、黃色均線與高對比買賣訊號 ---
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df["Close"],
                        name=f"{'月底' if is_monthly else ''}收盤價",
                        line=dict(color="#F8FAFC", width=2.2),
                        mode="lines+markers" if is_monthly else "lines",
                        marker=dict(size=5, color="#F8FAFC"),
                    ),
                    row=1,
                    col=1,
                )

                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df["MA"],
                        name=f"價格{ma_window}{unit_label}均線",
                        line=dict(color="#FACC15", width=2.0, dash="dash"),
                    ),
                    row=1,
                    col=1,
                )

                if not buy_signals.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=buy_signals.index,
                            y=buy_signals["Close"],
                            mode="markers",
                            name="買入做多 (+1)",
                            marker=dict(
                                symbol="triangle-up",
                                size=12,
                                color="#22C55E",
                                line=dict(width=1.2, color="#0F172A"),
                            ),
                        ),
                        row=1,
                        col=1,
                    )

                if not flat_signals.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=flat_signals.index,
                            y=flat_signals["Close"],
                            mode="markers",
                            name="平倉空手 (0)",
                            marker=dict(
                                symbol="circle",
                                size=11,
                                color="#FB923C",
                                line=dict(width=1.2, color="#0F172A"),
                            ),
                        ),
                        row=1,
                        col=1,
                    )

                if not short_signals.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=short_signals.index,
                            y=short_signals["Close"],
                            mode="markers",
                            name="反向做空 (-1)",
                            marker=dict(
                                symbol="triangle-down",
                                size=12,
                                color="#F43F5E",
                                line=dict(width=1.2, color="#0F172A"),
                            ),
                        ),
                        row=1,
                        col=1,
                    )

                # --- Row 2: 高對比淨值曲線 ---
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df["Strat_Equity"],
                        name="波動度濾網策略",
                        line=dict(color="#38BDF8", width=2.6),
                    ),
                    row=2,
                    col=1,
                )

                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df["BH_Equity"],
                        name="Buy & Hold",
                        line=dict(color="#E2E8F0", width=2.0, dash="dot"),
                    ),
                    row=2,
                    col=1,
                )

                # --- Row 3: Drawdown 曲線 ---
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df["Strat_DD"] * 100,
                        name="策略 Drawdown",
                        fill="tozeroy",
                        line=dict(color="#38BDF8", width=1.6),
                        fillcolor="rgba(56, 189, 248, 0.25)",
                    ),
                    row=3,
                    col=1,
                )

                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df["BH_DD"] * 100,
                        name="Buy & Hold Drawdown",
                        line=dict(color="#F43F5E", width=1.6),
                    ),
                    row=3,
                    col=1,
                )

                # --- Row 4: 依據模式繪製對應的波動度指標 ---
                if vol_mode == "1. 原始波動度 (%) vs 固定門檻":
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df["Vol"] * 100,
                            name=vol_chart_name,
                            line=dict(color="#C084FC", width=2.0),
                        ),
                        row=4,
                        col=1,
                    )
                    fig.add_hline(
                        y=vol_threshold * 100,
                        line_dash="dash",
                        line_color="#F43F5E",
                        annotation_text=f"門檻 ({vol_threshold*100:.1f}%)",
                        row=4,
                        col=1,
                    )

                elif vol_mode == "2. 標準化波動度 (Z-Score) vs 門檻":
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df["Vol_Z"],
                            name="標準化波動度 (Z-Score)",
                            line=dict(color="#C084FC", width=2.0),
                        ),
                        row=4,
                        col=1,
                    )
                    fig.add_hline(
                        y=z_threshold,
                        line_dash="dash",
                        line_color="#F43F5E",
                        annotation_text=f"門檻 ({z_threshold:.1f}σ)",
                        row=4,
                        col=1,
                    )
                    fig.add_hline(
                        y=0,
                        line_dash="dot",
                        line_color="#94A3B8",
                        opacity=0.6,
                        row=4,
                        col=1,
                    )

                elif vol_mode == "3. 波動度與其均線交叉 (波動度均線)":
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df["Vol"] * 100,
                            name=vol_chart_name,
                            line=dict(color="#C084FC", width=2.0),
                        ),
                        row=4,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df["Vol_MA"] * 100,
                            name=f"波動度{vol_ma_window}{unit_label}均線",
                            line=dict(color="#FACC15", width=1.8, dash="dash"),
                        ),
                        row=4,
                        col=1,
                    )

                else:  # 4. 標準化波動度 + 均線交叉雙重過濾
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df["Vol_Z"],
                            name="標準化波動度 (Z-Score)",
                            line=dict(color="#C084FC", width=2.0),
                        ),
                        row=4,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df["Vol_Z_MA"],
                            name=f"標準化波動度{z_ma_window}{unit_label}均線",
                            line=dict(color="#FACC15", width=1.8, dash="dash"),
                        ),
                        row=4,
                        col=1,
                    )
                    fig.add_hline(
                        y=z_high_thresh,
                        line_dash="dot",
                        line_color="#F43F5E",
                        annotation_text=f"高檔 ({z_high_thresh:.1f}σ)",
                        row=4,
                        col=1,
                    )
                    fig.add_hline(
                        y=z_low_thresh,
                        line_dash="dot",
                        line_color="#22C55E",
                        annotation_text=f"低檔 ({z_low_thresh:.1f}σ)",
                        row=4,
                        col=1,
                    )

                # 標示趨勢盤與盤整盤區間（無縫銜接）
                df["Regime_Group"] = (
                    df["Is_Trend"] != df["Is_Trend"].shift()
                ).cumsum()
                groups = list(df.groupby("Regime_Group"))

                for i, (_, group) in enumerate(groups):
                    is_trend = group["Is_Trend"].iloc[0]
                    x0 = group.index[0]
                    x1 = (
                        groups[i + 1][1].index[0]
                        if i + 1 < len(groups)
                        else group.index[-1]
                    )

                    fill_color = (
                        "rgba(56, 189, 248, 0.16)"
                        if is_trend
                        else "rgba(251, 146, 60, 0.14)"
                    )

                    for r in [1, 2]:
                        fig.add_vrect(
                            x0=x0,
                            x1=x1,
                            fillcolor=fill_color,
                            layer="below",
                            line_width=0,
                            row=r,
                            col=1,
                        )

                # 圖表版面配置
                fig.update_layout(
                    height=1120,
                    hovermode="x unified",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.04,
                        xanchor="left",
                        x=0,
                    ),
                    margin=dict(l=20, r=20, t=120, b=20),
                )
                fig.update_yaxes(title_text="價格", row=1, col=1)
                fig.update_yaxes(title_text="淨值 (倍數)", row=2, col=1)
                fig.update_yaxes(title_text="回撤 (%)", row=3, col=1)
                fig.update_yaxes(title_text=vol_y_title, row=4, col=1)

                st.plotly_chart(fig, use_container_width=True)

                # 7. 盤勢與倉位狀態統計摘要
                trend_pct = df["Is_Trend"].mean() * 100
                long_pct = (df["Position"] == 1).mean() * 100
                flat_pct = (df["Position"] == 0).mean() * 100
                short_pct = (df["Position"] == -1).mean() * 100

                st.info(
                    f"💡 **區間與倉位統計（{timeframe}）**：回測期間內**趨勢盤**佔比 `{trend_pct:.1f}%`、**盤整盤**佔比 `{100-trend_pct:.1f}%`｜"
                    f"倉位時間分佈：**做多** `{long_pct:.1f}%`、**空手** `{flat_pct:.1f}%`、**做空** `{short_pct:.1f}%`。"
                )

                # ==========================================
                # 月線逐月計算過程明細表（僅月線模式顯示）
                # ==========================================
                if is_monthly:
                    st.markdown("---")
                    st.subheader("📋 月線逐月計算過程明細表（由新到舊排列）")
                    st.caption(
                        "說明：每月最後一個交易日收盤計算指標與盤勢，產生「當月收盤訊號」後，於「訊號應用月份」正式持有該部位並計算報酬。"
                    )

                    apply_months = (df.index.to_period("M") + 1).astype(str)

                    table_data = {
                        "訊號計算日": df.index.strftime("%Y-%m-%d"),
                        "訊號應用月份": apply_months,
                        "月底收盤價": df["Close"].map(lambda x: f"{x:.2f}"),
                        f"價格{ma_window}月均線": df["MA"].map(
                            lambda x: f"{x:.2f}"
                        ),
                        "均線狀態": np.where(
                            df["Close"] > df["MA"], "站上均線", "跌破均線"
                        ),
                        "波動度 (%)": df["Vol"].map(
                            lambda x: f"{x*100:.2f}%"
                        ),
                    }

                    if "Vol_Z" in df.columns:
                        table_data["標準化波動度 (Z-Score)"] = df["Vol_Z"].map(
                            lambda x: f"{x:.2f}σ"
                        )
                    if "Vol_Z_MA" in df.columns:
                        table_data[f"Z-Score {z_ma_window}月均線"] = df[
                            "Vol_Z_MA"
                        ].map(lambda x: f"{x:.2f}σ")
                    if "Vol_MA" in df.columns:
                        table_data[f"波動度{vol_ma_window}月均線 (%)"] = df[
                            "Vol_MA"
                        ].map(lambda x: f"{x*100:.2f}%")

                    table_data.update(
                        {
                            "判定盤勢": np.where(
                                df["Is_Trend"], "趨勢盤", "盤整盤"
                            ),
                            "收盤產生訊號 (下月執行)": df["Signal"].map(
                                lambda x: format_pos_label(int(x))
                            ),
                            "當月實際持有部位": df["Position"].map(
                                lambda x: format_pos_label(int(x))
                            ),
                            "標的月報酬": df["BH_Ret"].map(
                                lambda x: f"{x*100:+.2f}%"
                            ),
                            "策略月報酬": df["Strat_Ret"].map(
                                lambda x: f"{x*100:+.2f}%"
                            ),
                            "策略累積淨值": df["Strat_Equity"].map(
                                lambda x: f"{x:.3f}"
                            ),
                        }
                    )

                    monthly_detail_df = pd.DataFrame(table_data)
                    monthly_detail_df = monthly_detail_df.iloc[::-1].reset_index(
                        drop=True
                    )

                    st.dataframe(
                        monthly_detail_df,
                        use_container_width=True,
                        height=500,
                    )
