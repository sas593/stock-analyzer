import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# =========================================================
# MULTIBAGGER LAB
# =========================================================

st.set_page_config(
    page_title="Multibagger Lab",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Multibagger Lab")
st.caption(
    "NSE stock scoring • 3× / 5× / 10× historical study • "
    "drawdown • CAGR • portfolio mathematics"
)

# =========================================================
# DATA FUNCTIONS
# =========================================================

@st.cache_data(ttl=900)
def load_price_data(symbol):
    return yf.download(
        symbol,
        period="max",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False
    )


def get_close_series(df):
    if df.empty:
        return pd.Series(dtype=float)

    close = df["Close"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    return pd.to_numeric(close, errors="coerce").dropna()


def calculate_max_drawdown(series):
    if series.empty:
        return 0

    peak = series.cummax()
    drawdown = (series / peak) - 1

    return float(drawdown.min() * 100)


def forward_multiple(close, entry_date, years):
    target_date = entry_date + pd.DateOffset(years=years)

    future = close.loc[close.index >= target_date]

    if future.empty:
        return np.nan

    entry_price = float(close.loc[entry_date])
    future_price = float(future.iloc[0])

    return future_price / entry_price


def calculate_signal_backtest(close):

    df = pd.DataFrame({"Close": close})

    df["MA50"] = close.rolling(50).mean()
    df["MA200"] = close.rolling(200).mean()
    df["Momentum6M"] = close.pct_change(126)

    # Historical price-based signal
    df["Signal"] = (
        (df["Close"] > df["MA200"]) &
        (df["MA50"] > df["MA200"]) &
        (df["Momentum6M"] > 0)
    )

    signal_dates = df.index[df["Signal"].fillna(False)]

    selected_dates = []
    last_signal = None

    # Avoid counting every signal day as a new trade
    for date in signal_dates:

        if (
            last_signal is None
            or (date - last_signal).days >= 90
        ):
            selected_dates.append(date)
            last_signal = date

    results = []

    for entry_date in selected_dates:

        entry_price = float(close.loc[entry_date])

        multiple_3y = forward_multiple(
            close,
            entry_date,
            3
        )

        multiple_5y = forward_multiple(
            close,
            entry_date,
            5
        )

        multiple_7y = forward_multiple(
            close,
            entry_date,
            7
        )

        # Maximum drawdown after entry during first 5 years
        future = close.loc[close.index > entry_date]

        end_date = entry_date + pd.DateOffset(years=5)

        window = future.loc[future.index <= end_date]

        path = pd.concat(
            [
                pd.Series(
                    [entry_price],
                    index=[entry_date]
                ),
                window
            ]
        )

        drawdown_5y = calculate_max_drawdown(path)

        results.append(
            {
                "Entry Date": entry_date.date(),
                "Entry Price": entry_price,
                "3Y Multiple": multiple_3y,
                "5Y Multiple": multiple_5y,
                "7Y Multiple": multiple_7y,
                "5Y Max DD": drawdown_5y,

                "3×": (
                    "YES"
                    if pd.notna(multiple_3y)
                    and multiple_3y >= 3
                    else "NO"
                ),

                "5×": (
                    "YES"
                    if pd.notna(multiple_5y)
                    and multiple_5y >= 5
                    else "NO"
                ),

                "10×": (
                    "YES"
                    if pd.notna(multiple_5y)
                    and multiple_5y >= 10
                    else "NO"
                )
            }
        )

    return pd.DataFrame(results)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("🔎 Stock Search")

ticker = st.sidebar.text_input(
    "Enter NSE ticker",
    value="HDFCBANK"
).strip().upper()

assumed_cagr = st.sidebar.slider(
    "Assumed 5Y EPS CAGR (%)",
    min_value=10.0,
    max_value=40.0,
    value=25.0,
    step=1.0
)

symbol = (
    ticker
    if ticker.endswith(".NS")
    else f"{ticker}.NS"
)


# =========================================================
# LOAD DATA
# =========================================================

try:

    with st.spinner(f"Loading {ticker} data..."):

        stock = yf.Ticker(symbol)

        info = stock.info

        hist = load_price_data(symbol)

        close = get_close_series(hist)


    if close.empty:

        st.error(
            "No historical data found. "
            "Please check the NSE ticker symbol."
        )

        st.stop()


    # =====================================================
    # FUNDAMENTALS
    # =====================================================

    price = (
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or float(close.iloc[-1])
    )

    eps = info.get("trailingEps") or 0

    pe = info.get("trailingPE") or 0

    roe = (
        info.get("returnOnEquity") or 0
    ) * 100

    debt_equity = info.get("debtToEquity")

    debt_equity = (
        float(debt_equity)
        if debt_equity is not None
        else np.nan
    )

    peg = info.get("pegRatio") or 0

    market_cap = (
        info.get("marketCap") or 0
    ) / 1e7

    company_name = (
        info.get("longName")
        or ticker
    )


    # =====================================================
    # 1. HARD GATEKEEPERS
    # =====================================================

    pass_debt = (
        np.isnan(debt_equity)
        or debt_equity < 100
    )

    pass_roe = roe > 12

    pass_pe = (
        0 < pe < 75
    )

    gatekeepers_passed = (
        pass_debt
        and pass_roe
        and pass_pe
    )


    # =====================================================
    # 2. 100 POINT SCORE
    # =====================================================

    score_earnings = (
        22
        if assumed_cagr >= 25
        else 15
        if assumed_cagr >= 15
        else 8
    )

    score_quality = (
        18
        if roe > 20
        and (
            np.isnan(debt_equity)
            or debt_equity < 30
        )
        else 12
        if roe > 12
        else 6
    )

    score_opportunity = (
        14
        if market_cap < 50000
        else 10
        if market_cap < 200000
        else 6
    )

    score_valuation = (
        13
        if 0 < peg <= 1.5
        else 8
        if 0 < pe < 40
        else 4
    )

    score_management = (
        9
        if pass_debt and roe > 15
        else 6
    )

    ma50 = close.rolling(50).mean().iloc[-1]

    score_technical = (
        8
        if price > ma50
        else 4
    )

    score_catalysts = 4

    total_score = (
        score_earnings
        + score_quality
        + score_opportunity
        + score_valuation
        + score_management
        + score_technical
        + score_catalysts
    )


    # =====================================================
    # 3. 10X MATHEMATICAL MODEL
    # =====================================================

    exit_pe = (
        35
        if pe <= 0
        else min(pe, 40)
    )

    future_eps = (
        eps
        * (
            1
            + assumed_cagr / 100
        ) ** 5
        if eps > 0
        else 0
    )

    target_price = (
        future_eps * exit_pe
    )

    potential_multiple = (
        target_price / price
        if price > 0
        else 0
    )


    # =====================================================
    # 4. VERDICT
    # =====================================================

    if total_score >= 80 and gatekeepers_passed:

        verdict = "BUY ZONE 🟢"
        portfolio_role = (
            "Top Conviction Multibagger"
        )

    elif total_score >= 65:

        verdict = "ACCUMULATE 🟡"
        portfolio_role = (
            "Core Quality Compounder"
        )

    else:

        verdict = "WATCH / AVOID 🔴"
        portfolio_role = (
            "High Risk / Low Probability"
        )


    # =====================================================
    # HEADER
    # =====================================================

    st.header(
        f"{company_name} ({ticker})"
    )

    st.write(
        f"**Market Cap:** ₹{market_cap:,.0f} Cr "
        f"| **Current Price:** ₹{price:,.2f}"
    )


    # =====================================================
    # KEY METRICS
    # =====================================================

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Current Price",
        f"₹{price:,.2f}"
    )

    c2.metric(
        "Overall Score",
        f"{total_score}/100"
    )

    c3.metric(
        "5Y Model Multiple",
        f"{potential_multiple:.1f}×"
    )

    c4.metric(
        "Decision",
        verdict
    )


    # =====================================================
    # RISK + SCORE
    # =====================================================

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader(
            "1. Hard Gatekeeper Check"
        )

        if np.isnan(debt_equity):

            st.write(
                "Debt/Equity: PASS / unavailable"
            )

        else:

            st.write(
                f"Debt/Equity: "
                f"{'PASS ✅' if pass_de else 'FAIL ❌'} "
                f"({debt_equity:.1f}%)"
            )

        st.write(
            f"ROE > 12%: "
            f"{'PASS ✅' if pass_roe else 'FAIL ❌'} "
            f"({roe:.1f}%)"
        )

        st.write(
            f"P/E < 75: "
            f"{'PASS ✅' if pass_pe else 'FAIL ❌'} "
            f"({pe:.1f}×)"
        )

        if gatekeepers_passed:

            st.success(
                "All mandatory gatekeepers passed."
            )

        else:

            st.error(
                "One or more gatekeepers failed."
            )


    with right:

        st.subheader(
            "2. 100-Point Factor Score"
        )

        score_table = pd.DataFrame(
            {
                "Factor": [
                    "Earnings Growth",
                    "Business Quality",
                    "Future Opportunity",
                    "Valuation",
                    "Management",
                    "Technical Setup",
                    "Catalysts"
                ],

                "Score": [
                    score_earnings,
                    score_quality,
                    score_opportunity,
                    score_valuation,
                    score_management,
                    score_technical,
                    score_catalysts
                ],

                "Maximum": [
                    25,
                    20,
                    15,
                    15,
                    10,
                    10,
                    5
                ]
            }
        )

        st.dataframe(
            score_table,
            use_container_width=True,
            hide_index=True
        )


    # =====================================================
    # 10X MATH + PRICE HISTORY
    # =====================================================

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader(
            "3. 10× Mathematical Model"
        )

        st.write(
            f"Trailing EPS: ₹{eps:.2f}"
        )

        st.write(
            f"Assumed EPS CAGR: "
            f"{assumed_cagr:.0f}%"
        )

        st.write(
            f"Projected 5Y EPS: "
            f"₹{future_eps:.2f}"
        )

        st.write(
            f"Modeled Exit P/E: "
            f"{exit_pe:.1f}×"
        )

        st.write(
            f"5Y Target Price: "
            f"₹{target_price:,.2f}"
        )

        if potential_multiple >= 10:

            st.success(
                "🔥 10× mathematical candidate"
            )

        elif potential_multiple >= 5:

            st.success(
                "🟢 5×+ mathematical candidate"
            )

        elif potential_multiple >= 3:

            st.info(
                "🟡 3× mathematical candidate"
            )

        else:

            st.warning(
                "Below 3× mathematical model"
            )


    with right:

        st.subheader(
            "4. Historical Price Behaviour"
        )

        st.metric(
            "5Y Maximum Drawdown",
            f"{calculate_max_drawdown(close):.1f}%"
        )

        st.line_chart(
            close.tail(1250)
        )


    # =====================================================
    # HISTORICAL BACKTEST
    # =====================================================

    st.divider()

    st.header(
        "🔬 Historical Multibagger Backtest"
    )

    st.info(
        "This backtest uses historical price behaviour "
        "only. It does NOT use today's fundamentals to "
        "score historical dates. Therefore it avoids the "
        "most obvious hindsight problem. A full "
        "point-in-time fundamental backtest requires "
        "archived historical financial data."
    )


    backtest = calculate_signal_backtest(
        close
    )


    if backtest.empty:

        st.warning(
            "No qualifying historical signals found."
        )

    else:

        # ---------------------------------------------
        # SUMMARY
        # ---------------------------------------------

        total_signals = len(backtest)

        hits_3x = int(
            (
                backtest["3×"] == "YES"
            ).sum()
        )

        hits_5x = int(
            (
                backtest["5×"] == "YES"
            ).sum()
        )

        hits_10x = int(
            (
                backtest["10×"] == "YES"
            ).sum()
        )

        q1, q2, q3, q4 = st.columns(4)

        q1.metric(
            "Historical Signals",
            total_signals
        )

        q2.metric(
            "3× Hits",
            hits_3x
        )

        q3.metric(
            "5× Hits",
            hits_5x
        )

        q4.metric(
            "10× Hits",
            hits_10x
        )


        st.subheader(
            "Signal-by-Signal Results"
        )

        display_bt = backtest.copy()

        for col in [
            "3Y Multiple",
            "5Y Multiple",
            "7Y Multiple",
            "5Y Max DD"
        ]:

            display_bt[col] = (
                display_bt[col]
                .round(2)
            )

        st.dataframe(
            display_bt.sort_values(
                "Entry Date",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )


        # ---------------------------------------------
        # BEST VS FAILED
        # ---------------------------------------------

        valid = backtest.dropna(
            subset=["5Y Multiple"]
        )

        if not valid.empty:

            st.subheader(
                "🏆 Best vs Failed Signals"
            )

            best_col, failed_col = st.columns(2)

            with best_col:

                st.write(
                    "🏆 Best Historical Signals"
                )

                st.dataframe(
                    valid
                    .sort_values(
                        "5Y Multiple",
                        ascending=False
                    )
                    .head(10),
                    use_container_width=True,
                    hide_index=True
                )


            with failed_col:

                st.write(
                    "⚠️ Weak / Failed Signals"
                )

                failed = valid[
                    valid["5Y Multiple"] < 2
                ].sort_values(
                    "5Y Multiple"
                )

                st.dataframe(
                    failed.head(10),
                    use_container_width=True,
                    hide_index=True
                )


            # -----------------------------------------
            # PERFORMANCE METRICS
            # -----------------------------------------

            win_rate = (
                valid["5Y Multiple"] >= 3
            ).mean() * 100

            worst_drawdown = (
                valid["5Y Max DD"].min()
            )

            average_multiple = (
                valid["5Y Multiple"].mean()
            )

            st.subheader(
                "Backtest Statistics"
            )

            b1, b2, b3 = st.columns(3)

            b1.metric(
                "3× Win Rate",
                f"{win_rate:.1f}%"
            )

            b2.metric(
                "Average 5Y Multiple",
                f"{average_multiple:.2f}×"
            )

            b3.metric(
                "Worst 5Y Drawdown",
                f"{worst_drawdown:.1f}%"
            )


        # ---------------------------------------------
        # EXPORT
        # ---------------------------------------------

        csv = backtest.to_csv(
            index=False
        )

        st.download_button(
            "⬇️ Download Backtest CSV",
            csv,
            file_name=(
                f"{ticker}_multibagger_backtest.csv"
            ),
            mime="text/csv"
        )


    # =====================================================
    # PORTFOLIO MATH
    # =====================================================

    st.divider()

    st.header(
        "💰 ₹10 Lakh → ₹1 Crore Portfolio Math"
    )

    p1, p2, p3 = st.columns(3)

    with p1:

        starting_capital = st.number_input(
            "Starting Capital",
            min_value=100000,
            max_value=100000000,
            value=1000000,
            step=100000
        )

    with p2:

        target_capital = st.number_input(
            "Target Capital",
            min_value=1000000,
            max_value=1000000000,
            value=10000000,
            step=1000000
        )

    with p3:

        investment_years = st.slider(
            "Investment Period",
            min_value=3,
            max_value=15,
            value=5
        )


    required_cagr = (
        (
            target_capital
            / starting_capital
        ) ** (
            1 / investment_years
        )
        - 1
    ) * 100


    st.metric(
        "Required CAGR",
        f"{required_cagr:.2f}%"
    )


    portfolio_values = [
        starting_capital
        * (
            1 + required_cagr / 100
        ) ** year
        for year in range(
            investment_years + 1
        )
    ]


    portfolio_df = pd.DataFrame(
        {
            "Portfolio Value": portfolio_values
        },
        index=range(
            investment_years + 1
        )
    )


    st.line_chart(
        portfolio_df
    )


    # =====================================================
    # EXECUTION LEVELS
    # =====================================================

    st.divider()

    st.header(
        "🎯 Execution Price Levels"
    )

    e1, e2, e3, e4 = st.columns(4)

    e1.metric(
        "15% Down",
        f"₹{price * 0.85:,.2f}"
    )

    e2.metric(
        "2×",
        f"₹{price * 2:,.2f}"
    )

    e3.metric(
        "3×",
        f"₹{price * 3:,.2f}"
    )

    e4.metric(
        "5×",
        f"₹{price * 5:,.2f}"
    )


    # =====================================================
    # FINAL THESIS
    # =====================================================

    st.divider()

    st.header(
        "🧠 Investment Thesis Snapshot"
    )

    thesis_col1, thesis_col2 = st.columns(2)

    with thesis_col1:

        st.write(
            f"**Company:** {company_name}"
        )

        st.write(
            f"**Current Price:** ₹{price:,.2f}"
        )

        st.write(
            f"**Score:** {total_score}/100"
        )

        st.write(
            f"**Decision:** {verdict}"
        )

        st.write(
            f"**Portfolio Role:** "
            f"{portfolio_role}"
        )


    with thesis_col2:

        st.write(
            f"**5Y EPS CAGR Assumption:** "
            f"{assumed_cagr:.0f}%"
        )

        st.write(
            f"**5Y Model Multiple:** "
            f"{potential_multiple:.1f}×"
        )

        st.write(
            f"**Historical Maximum Drawdown:** "
            f"{calculate_max_drawdown(close):.1f}%"
        )

        st.write(
            f"**Historical 3× Signals:** "
            f"{hits_3x if not backtest.empty else 0}"
        )

        st.write(
            f"**Historical 10× Signals:** "
            f"{hits_10x if not backtest.empty else 0}"
        )


    # =====================================================
    # DATA DISCLAIMER
    # =====================================================

    st.divider()

    st.caption(
        "Data source: Yahoo Finance via yfinance. "
        "Historical price results are downloaded market data. "
        "Fundamentals are current/latest available data and "
        "are NOT reconstructed point-in-time. "
        "This tool is a research engine, not a guarantee of future returns."
    )


# =========================================================
# ERROR HANDLING
# =========================================================

except Exception as error:

    st.error(
        f"Unable to analyse {ticker}. "
        "Check the NSE ticker or Yahoo Finance availability."
    )

    st.exception(error)
