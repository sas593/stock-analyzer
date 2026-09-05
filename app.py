import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# ============================================================
# APP CONFIG
# ============================================================
st.set_page_config(
    page_title="Multibagger Lab",
    layout="wide"
)


# ============================================================
# GENERAL HELPERS
# ============================================================
def clean_num(x):
    try:
        if x is None or pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def normalize_symbol(x):
    x = str(x).strip().upper()

    if not x:
        return ""

    return x if x.endswith(".NS") else f"{x}.NS"


def display_ticker(symbol):
    return str(symbol).replace(".NS", "")


def fmt_pct(x):
    if pd.isna(x):
        return "Unavailable"

    return f"{x:.1f}%"


def fmt_x(x):
    if pd.isna(x):
        return "Unavailable"

    return f"{x:.1f}×"


# ============================================================
# MARKET DATA
# ============================================================
@st.cache_data(
    ttl=900,
    show_spinner=False
)
def load_price(symbol, period="max"):
    try:
        return yf.download(
            symbol,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            group_by="column"
        )
    except Exception:
        return pd.DataFrame()


@st.cache_data(
    ttl=900,
    show_spinner=False
)
def load_fundamentals(symbol):
    try:
        t = yf.Ticker(symbol)

        return {
            "info": t.info,
            "income": t.income_stmt,
            "balance": t.balance_sheet,
            "cashflow": t.cashflow,
        }

    except Exception:
        return {
            "info": {},
            "income": pd.DataFrame(),
            "balance": pd.DataFrame(),
            "cashflow": pd.DataFrame(),
        }


def close_series(df):
    if (
        df is None
        or df.empty
        or "Close" not in df
    ):
        return pd.Series(dtype=float)

    x = df["Close"]

    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]

    x = pd.to_numeric(
        x,
        errors="coerce"
    ).dropna()

    try:
        x.index = pd.to_datetime(
            x.index
        ).tz_localize(None)
    except Exception:
        x.index = pd.to_datetime(
            x.index
        )

    return x.sort_index()


def row_value(df, names):
    if (
        df is None
        or df.empty
    ):
        return np.nan

    for name in names:

        if name in df.index:

            s = pd.to_numeric(
                df.loc[name],
                errors="coerce"
            ).dropna()

            if not s.empty:
                return float(
                    s.iloc[0]
                )

    return np.nan


def annual_series(df, names):
    if (
        df is None
        or df.empty
    ):
        return pd.Series(
            dtype=float
        )

    for name in names:

        if name in df.index:

            s = pd.to_numeric(
                df.loc[name],
                errors="coerce"
            ).dropna()

            if len(s) >= 2:

                s.index = pd.to_datetime(
                    s.index
                )

                return s.sort_index()

    return pd.Series(
        dtype=float
    )


# ============================================================
# FINANCIAL CALCULATIONS
# ============================================================
def cagr_from_series(s, years=3):

    if len(s) < 2:
        return np.nan

    s = s.dropna()

    if len(s) < 2:
        return np.nan

    # CAGR is not valid through zero/negative earnings.
    if (s <= 0).any():
        return np.nan

    end = float(s.iloc[-1])

    target_date = (
        s.index[-1]
        - pd.DateOffset(
            years=years
        )
    )

    prior = s.loc[
        s.index <= target_date
    ]

    if prior.empty:
        if len(s) < 2:
            return np.nan

        start = float(
            s.iloc[0]
        )

        actual_years = (
            s.index[-1] -
            s.index[0]
        ).days / 365.25

    else:

        start = float(
            prior.iloc[-1]
        )

        actual_years = (
            s.index[-1] -
            prior.index[-1]
        ).days / 365.25

    if (
        start <= 0
        or end <= 0
        or actual_years <= 0
    ):
        return np.nan

    return (
        (end / start)
        ** (1 / actual_years)
        - 1
    )


def max_dd(s):

    if len(s) == 0:
        return np.nan

    peak = s.cummax()

    return (
        (s / peak) - 1
    ).min() * 100


def forward_multiple(
    close,
    entry,
    years
):

    target_date = (
        entry +
        pd.DateOffset(
            years=years
        )
    )

    future = close.loc[
        close.index >= target_date
    ]

    if future.empty:
        return np.nan

    return float(
        future.iloc[0] /
        close.loc[entry]
    )


def safe_growth_score(
    earnings_cagr,
    revenue_cagr
):

    if not pd.isna(
        earnings_cagr
    ):

        if earnings_cagr >= .25:
            return 25

        if earnings_cagr >= .15:
            return 19

        if earnings_cagr >= .08:
            return 12

        return 5

    if not pd.isna(
        revenue_cagr
    ):

        if revenue_cagr >= .25:
            return 22

        if revenue_cagr >= .15:
            return 16

        if revenue_cagr >= .08:
            return 9

        return 4

    return 0


# ============================================================
# STOCK ANALYSIS ENGINE
# ============================================================
def analyse_symbol(
    symbol,
    need_history=True
):

    symbol = normalize_symbol(
        symbol
    )

    if not symbol:
        return None

    price_df = load_price(
        symbol,
        "max"
        if need_history
        else "1y"
    )

    close = close_series(
        price_df
    )

    if close.empty:
        return None

    price = float(
        close.iloc[-1]
    )

    latest_date = (
        close.index[-1]
    )

    fund = load_fundamentals(
        symbol
    )

    info = fund["info"]
    income = fund["income"]
    balance = fund["balance"]
    cashflow = fund["cashflow"]

    name = (
        info.get("longName")
        or display_ticker(symbol)
    )

    # --------------------------------------------------------
    # EPS
    # --------------------------------------------------------
    eps = clean_num(
        info.get(
            "trailingEps"
        )
    )

    diluted_eps_s = annual_series(
        income,
        [
            "Diluted EPS",
            "Basic EPS"
        ]
    )

    if (
        pd.isna(eps)
        and not diluted_eps_s.empty
    ):
        eps = float(
            diluted_eps_s.iloc[-1]
        )

    # --------------------------------------------------------
    # P/E
    # --------------------------------------------------------
    pe = clean_num(
        info.get(
            "trailingPE"
        )
    )

    if (
        pd.isna(pe)
        and not pd.isna(eps)
        and eps > 0
    ):
        pe = price / eps

    # --------------------------------------------------------
    # Growth
    # --------------------------------------------------------
    net_income_s = annual_series(
        income,
        [
            "Net Income",
            "Net Income Common Stockholders"
        ]
    )

    revenue_s = annual_series(
        income,
        [
            "Total Revenue",
            "Operating Revenue"
        ]
    )

    earnings_cagr = cagr_from_series(
        diluted_eps_s,
        3
    )

    if pd.isna(
        earnings_cagr
    ):
        earnings_cagr = cagr_from_series(
            net_income_s,
            3
        )

    revenue_cagr = cagr_from_series(
        revenue_s,
        3
    )

    # --------------------------------------------------------
    # ROE
    # --------------------------------------------------------
    equity = row_value(
        balance,
        [
            "Stockholders Equity",
            "Total Equity Gross Minority Interest"
        ]
    )

    ni = row_value(
        income,
        [
            "Net Income",
            "Net Income Common Stockholders"
        ]
    )

    roe = clean_num(
        info.get(
            "returnOnEquity"
        )
    )

    if (
        pd.isna(roe)
        and not pd.isna(ni)
        and not pd.isna(equity)
        and equity != 0
    ):
        roe = ni / equity

    roe_pct = (
        roe * 100
        if not pd.isna(roe)
        else np.nan
    )

    # --------------------------------------------------------
    # Debt
    # --------------------------------------------------------
    debt = row_value(
        balance,
        ["Total Debt"]
    )

    de = clean_num(
        info.get(
            "debtToEquity"
        )
    )

    # Yahoo debtToEquity is normally expressed
    # as a percentage number.
    if (
        pd.isna(de)
        and not pd.isna(debt)
        and not pd.isna(equity)
        and equity != 0
    ):
        de = (
            debt /
            equity *
            100
        )

    # --------------------------------------------------------
    # Market Cap
    # --------------------------------------------------------
    market_cap = clean_num(
        info.get(
            "marketCap"
        )
    )

    if pd.isna(
        market_cap
    ):

        shares = row_value(
            balance,
            [
                "Ordinary Shares Number",
                "Share Issued"
            ]
        )

        if not pd.isna(shares):
            market_cap = (
                price * shares
            )

    # --------------------------------------------------------
    # Cash Flow
    # --------------------------------------------------------
    peg = clean_num(
        info.get(
            "pegRatio"
        )
    )

    ocf = row_value(
        cashflow,
        [
            "Operating Cash Flow",
            "Total Cash From Operating Activities"
        ]
    )

    fcf = row_value(
        cashflow,
        [
            "Free Cash Flow"
        ]
    )

    if pd.isna(
        fcf
    ):

        capex = row_value(
            cashflow,
            [
                "Capital Expenditure",
                "Capital Expenditures"
            ]
        )

        if (
            not pd.isna(ocf)
            and not pd.isna(capex)
        ):
            fcf = (
                ocf + capex
            )

    # --------------------------------------------------------
    # Ownership
    # --------------------------------------------------------
    promoter = clean_num(
        info.get(
            "heldPercentInsiders"
        )
    )

    # Do NOT confuse insider ownership
    # with promoter pledge.
    pledge = np.nan

    # --------------------------------------------------------
    # Technicals
    # --------------------------------------------------------
    ma50 = (
        close.rolling(50)
        .mean()
        .iloc[-1]
        if len(close) >= 50
        else np.nan
    )

    ma200 = (
        close.rolling(200)
        .mean()
        .iloc[-1]
        if len(close) >= 200
        else np.nan
    )

    momentum_6m = (
        close.iloc[-1] /
        close.iloc[-127] -
        1
        if len(close) >= 127
        else np.nan
    )

    entry_conditions = {

        "Price > 200 DMA":
            price > ma200
            if not pd.isna(ma200)
            else False,

        "50 DMA > 200 DMA":
            ma50 > ma200
            if (
                not pd.isna(ma50)
                and not pd.isna(ma200)
            )
            else False,

        "6M Momentum > 0":
            momentum_6m > 0
            if not pd.isna(
                momentum_6m
            )
            else False,
    }

    entry_trigger = all(
        entry_conditions.values()
    )

    exit_conditions = {

        "Price < 200 DMA":
            price < ma200
            if not pd.isna(ma200)
            else False,

        "50 DMA < 200 DMA":
            ma50 < ma200
            if (
                not pd.isna(ma50)
                and not pd.isna(ma200)
            )
            else False,
    }

    exit_trigger = all(
        exit_conditions.values()
    )

    # --------------------------------------------------------
    # Hard Gates
    # --------------------------------------------------------
    de_pass = (
        True
        if pd.isna(de)
        else de < 100
    )

    roe_pass = (
        True
        if pd.isna(roe_pct)
        else roe_pct > 12
    )

    pe_pass = (
        True
        if pd.isna(pe)
        else 0 < pe < 75
    )

    gates = (
        de_pass
        and roe_pass
        and pe_pass
    )

    # --------------------------------------------------------
    # 100 POINT SCORE
    # --------------------------------------------------------
    s1 = safe_growth_score(
        earnings_cagr,
        revenue_cagr
    )

    if not pd.isna(
        roe_pct
    ):

        if (
            roe_pct > 20
            and (
                pd.isna(de)
                or de < 30
            )
        ):
            s2 = 18

        elif roe_pct > 15:
            s2 = 14

        elif roe_pct > 12:
            s2 = 10

        else:
            s2 = 5

    else:
        s2 = 0

    if not pd.isna(
        market_cap
    ):

        mcap_cr = (
            market_cap /
            1e7
        )

        if mcap_cr < 50000:
            s3 = 15

        elif mcap_cr < 200000:
            s3 = 11

        else:
            s3 = 7

    else:
        s3 = 0

    if (
        not pd.isna(peg)
        and peg > 0
    ):

        if peg <= 1:
            s4 = 15

        elif peg <= 1.5:
            s4 = 11

        elif peg <= 2:
            s4 = 7

        else:
            s4 = 3

    elif (
        not pd.isna(pe)
        and pe > 0
    ):

        if pe <= 20:
            s4 = 15

        elif pe <= 30:
            s4 = 11

        elif pe <= 40:
            s4 = 7

        else:
            s4 = 3

    else:
        s4 = 0

    # Management/governance placeholder
    # until verified company-filing data is connected.
    s5 = 6

    if entry_trigger:
        s6 = 10

    elif (
        not pd.isna(ma200)
        and price > ma200
        and ma50 > ma200
    ):
        s6 = 8

    elif (
        not pd.isna(ma200)
        and price > ma200
    ):
        s6 = 6

    else:
        s6 = 3

    # Catalysts remain zero until a
    # validated catalyst data source is connected.
    s7 = 0

    total = int(
        s1 +
        s2 +
        s3 +
        s4 +
        s5 +
        s6 +
        s7
    )

    # --------------------------------------------------------
    # 5 YEAR MODEL
    # --------------------------------------------------------
    base_cagr = (
        earnings_cagr
        if not pd.isna(
            earnings_cagr
        )
        else revenue_cagr
    )

    # IMPORTANT:
    # Do not pretend missing growth is actual data.
    # 15% is only a modelling assumption.
    base_cagr_is_assumption = False

    if pd.isna(
        base_cagr
    ):
        base_cagr = 0.15
        base_cagr_is_assumption = True

    # Avoid absurd growth assumptions.
    base_cagr = min(
        max(base_cagr, -0.50),
        0.50
    )

    exit_pe = (
        pe
        if (
            not pd.isna(pe)
            and pe > 0
        )
        else 25
    )

    exit_pe = min(
        exit_pe,
        40
    )

    future_eps = (
        eps *
        (
            (1 + base_cagr)
            ** 5
        )
        if (
            not pd.isna(eps)
            and eps > 0
        )
        else np.nan
    )

    target = (
        future_eps *
        exit_pe
        if not pd.isna(
            future_eps
        )
        else np.nan
    )

    multiple = (
        target /
        price
        if (
            not pd.isna(target)
            and price > 0
        )
        else np.nan
    )

    # --------------------------------------------------------
    # VERDICT
    # --------------------------------------------------------
    if exit_trigger:
        verdict = (
            "WATCH / AVOID 🔴"
        )

    elif (
        total >= 80
        and gates
        and entry_trigger
    ):
        verdict = (
            "BUY / ENTRY 🟢"
        )

    elif (
        total >= 65
        and gates
    ):
        verdict = (
            "ACCUMULATE 🟡"
        )

    else:
        verdict = (
            "WATCH / AVOID 🔴"
        )

    return {
        "symbol": symbol,
        "ticker": display_ticker(symbol),
        "name": name,
        "close": close,
        "price": price,
        "latest_date": latest_date,
        "eps": eps,
        "pe": pe,
        "roe_pct": roe_pct,
        "de": de,
        "market_cap": market_cap,
        "peg": peg,
        "ocf": ocf,
        "fcf": fcf,
        "promoter": promoter,
        "pledge": pledge,
        "earnings_cagr": earnings_cagr,
        "revenue_cagr": revenue_cagr,
        "ma50": ma50,
        "ma200": ma200,
        "momentum_6m": momentum_6m,
        "entry_conditions": entry_conditions,
        "entry_trigger": entry_trigger,
        "exit_conditions": exit_conditions,
        "exit_trigger": exit_trigger,
        "de_pass": de_pass,
        "roe_pass": roe_pass,
        "pe_pass": pe_pass,
        "gates": gates,
        "s1": s1,
        "s2": s2,
        "s3": s3,
        "s4": s4,
        "s5": s5,
        "s6": s6,
        "s7": s7,
        "score": total,
        "base_cagr": base_cagr,
        "base_cagr_is_assumption":
            base_cagr_is_assumption,
        "exit_pe": exit_pe,
        "future_eps": future_eps,
        "target": target,
        "multiple": multiple,
        "verdict": verdict,
    }


# ============================================================
# STOCK ANALYSIS PAGE
# ============================================================
def render_stock_page():

    st.title(
        "📈 Multibagger Lab"
    )

    st.caption(
        "NSE stock analysis • fundamentals • "
        "technicals • multibagger model • backtest"
    )

    ticker = st.sidebar.text_input(
        "NSE ticker",
        "DODLA"
    ).strip().upper()

    if not ticker:
        return

    with st.spinner(
        f"Loading {ticker}..."
    ):
        a = analyse_symbol(
            ticker,
            need_history=True
        )

    if a is None:
        st.error(
            f"No usable market data found for {ticker}."
        )
        return

    st.header(
        f"{a['name']} ({a['ticker']})"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Latest Price",
        f"₹{a['price']:,.2f}",
        f"As of {a['latest_date'].date()}"
    )

    c2.metric(
        "Score",
        f"{a['score']}/100"
    )

    c3.metric(
        "5Y Model",
        fmt_x(a['multiple'])
    )

    c4.metric(
        "Decision",
        a['verdict']
    )

    st.divider()

    # --------------------------------------------------------
    # ENTRY / EXIT
    # --------------------------------------------------------
    st.subheader(
        "🎯 Entry / Exit Timing"
    )

    e1, e2 = st.columns(2)

    with e1:

        if a["entry_trigger"]:
            st.success(
                "🟢 ENTRY TRIGGER: YES"
            )
        else:
            st.error(
                "🔴 ENTRY TRIGGER: NO"
            )

        for k, v in (
            a["entry_conditions"]
            .items()
        ):
            st.write(
                f"{'✅' if v else '❌'} {k}"
            )

        st.caption(
            f"Price ₹{a['price']:,.2f} | "
            f"50 DMA ₹{a['ma50']:,.2f} | "
            f"200 DMA ₹{a['ma200']:,.2f} | "
            f"6M momentum {fmt_pct(a['momentum_6m']*100)}"
        )

    with e2:

        if a["exit_trigger"]:
            st.error(
                "🔴 EXIT TRIGGER: YES"
            )
        else:
            st.success(
                "🟢 EXIT TRIGGER: NO"
            )

        for k, v in (
            a["exit_conditions"]
            .items()
        ):
            st.write(
                f"{'⚠️' if v else '✅'} {k}"
            )

        st.caption(
            "Exit = Price below 200 DMA "
            "AND 50 DMA below 200 DMA."
        )

    st.divider()

    # --------------------------------------------------------
    # GATES + SCORE
    # --------------------------------------------------------
    left, right = st.columns(2)

    with left:

        st.subheader(
            "1. Gatekeeper Checks"
        )

        def gate_line(
            label,
            value,
            passed,
            unit=""
        ):

            if pd.isna(value):

                st.write(
                    f"⚪ {label}: DATA UNAVAILABLE"
                )

            else:

                st.write(
                    f"{'✅' if passed else '❌'} "
                    f"{label}: {value:.1f}{unit}"
                )

        gate_line(
            "Debt/Equity",
            a["de"],
            a["de_pass"],
            "%"
        )

        gate_line(
            "ROE",
            a["roe_pct"],
            a["roe_pass"],
            "%"
        )

        gate_line(
            "P/E",
            a["pe"],
            a["pe_pass"],
            "×"
        )

        if a["gates"]:
            st.success(
                "Available gatekeeper checks pass."
            )
        else:
            st.error(
                "One or more gatekeeper checks fail."
            )

    with right:

        st.subheader(
            "2. Automatic 100-Point Score"
        )

        score_df = pd.DataFrame({
            "Factor": [
                "Earnings Growth",
                "Business Quality",
                "Future Opportunity",
                "Valuation",
                "Management & Governance",
                "Technical Setup",
                "Catalysts"
            ],
            "Score": [
                a["s1"],
                a["s2"],
                a["s3"],
                a["s4"],
                a["s5"],
                a["s6"],
                a["s7"]
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
        })

        st.dataframe(
            score_df,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    # --------------------------------------------------------
    # FUNDAMENTALS
    # --------------------------------------------------------
    st.subheader(
        "3. Fundamental Snapshot"
    )

    f = st.columns(4)

    f[0].metric(
        "EPS",
        (
            "Unavailable"
            if pd.isna(a["eps"])
            else f"₹{a['eps']:.2f}"
        )
    )

    f[1].metric(
        "3Y Earnings CAGR",
        (
            "Unavailable"
            if pd.isna(
                a["earnings_cagr"]
            )
            else f"{a['earnings_cagr']*100:.1f}%"
        )
    )

    f[2].metric(
        "3Y Revenue CAGR",
        (
            "Unavailable"
            if pd.isna(
                a["revenue_cagr"]
            )
            else f"{a['revenue_cagr']*100:.1f}%"
        )
    )

    f[3].metric(
        "Market Cap",
        (
            "Unavailable"
            if pd.isna(
                a["market_cap"]
            )
            else f"₹{a['market_cap']/1e7:,.0f} Cr"
        )
    )

    g = st.columns(4)

    g[0].metric(
        "ROE",
        fmt_pct(a["roe_pct"])
    )

    g[1].metric(
        "P/E",
        fmt_x(a["pe"])
    )

    g[2].metric(
        "Debt/Equity",
        (
            "Unavailable"
            if pd.isna(a["de"])
            else f"{a['de']:.1f}%"
        )
    )

    g[3].metric(
        "Operating Cash Flow",
        (
            "Unavailable"
            if pd.isna(a["ocf"])
            else f"₹{a['ocf']/1e7:,.0f} Cr"
        )
    )

    st.caption(
        "Missing data is shown as unavailable and is "
        "not converted to zero."
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------
    st.divider()

    st.subheader(
        "4. 5-Year Multibagger Model"
    )

    m1, m2 = st.columns(2)

    with m1:

        st.write(
            f"Current EPS: "
            f"{'Unavailable' if pd.isna(a['eps']) else f'₹{a['eps']:.2f}'}"
        )

        st.write(
            f"Growth assumption: "
            f"{a['base_cagr']*100:.1f}%"
        )

        if a[
            "base_cagr_is_assumption"
        ]:
            st.warning(
                "Growth assumption is used only because "
                "validated historical earnings growth is unavailable."
            )

        st.write(
            f"Modeled exit P/E: "
            f"{a['exit_pe']:.1f}×"
        )

        st.write(
            f"Projected 5Y EPS: "
            f"{'Unavailable' if pd.isna(a['future_eps']) else f'₹{a['future_eps']:.2f}'}"
        )

        st.write(
            f"Modeled 5Y price: "
            f"{'Unavailable' if pd.isna(a['target']) else f'₹{a['target']:,.2f}'}"
        )

    with m2:

        if pd.isna(
            a["multiple"]
        ):
            st.warning(
                "Insufficient EPS data."
            )

        elif a["multiple"] >= 10:
            st.success(
                f"10×+ model candidate — "
                f"{a['multiple']:.1f}×"
            )

        elif a["multiple"] >= 5:
            st.success(
                f"5×+ model candidate — "
                f"{a['multiple']:.1f}×"
            )

        elif a["multiple"] >= 3:
            st.warning(
                f"3×+ model candidate — "
                f"{a['multiple']:.1f}×"
            )

        else:
            st.info(
                f"Below 3× model — "
                f"{a['multiple']:.1f}×"
            )

    # --------------------------------------------------------
    # CHART
    # --------------------------------------------------------
    st.divider()

    st.subheader(
        "5. Price Trend"
    )

    st.line_chart(
        pd.DataFrame({
            "Price":
                a["close"],
            "50 DMA":
                a["close"].rolling(50).mean(),
            "200 DMA":
                a["close"].rolling(200).mean()
        }).tail(750)
    )

    st.metric(
        "Maximum Drawdown",
        f"{max_dd(a['close']):.1f}%"
    )

    # --------------------------------------------------------
    # BACKTEST
    # --------------------------------------------------------
    st.divider()

    st.header(
        "🔬 Historical Multibagger Backtest"
    )

    st.info(
        "Current backtest is a price-based signal study. "
        "It is NOT yet the final point-in-time fundamental "
        "backtest. The final investment engine will use "
        "point-in-time fundamentals to avoid look-ahead bias."
    )

    bt = pd.DataFrame({
        "Close": a["close"]
    })

    bt["MA50"] = (
        bt["Close"]
        .rolling(50)
        .mean()
    )

    bt["MA200"] = (
        bt["Close"]
        .rolling(200)
        .mean()
    )

    bt["Momentum6M"] = (
        bt["Close"]
        .pct_change(126)
    )

    bt["Entry"] = (
        (bt["Close"] > bt["MA200"])
        &
        (bt["MA50"] > bt["MA200"])
        &
        (bt["Momentum6M"] > 0)
    )

    dates = bt.index[
        bt["Entry"]
        .fillna(False)
    ]

    selected = []
    last = None

    for dt in dates:

        if (
            last is None
            or (
                dt - last
            ).days >= 90
        ):

            selected.append(dt)
            last = dt

    rows = []

    for entry in selected:

        ep = float(
            a["close"].loc[entry]
        )

        m3 = forward_multiple(
            a["close"],
            entry,
            3
        )

        m5 = forward_multiple(
            a["close"],
            entry,
            5
        )

        future = a["close"].loc[
            a["close"].index > entry
        ]

        end = (
            entry +
            pd.DateOffset(
                years=5
            )
        )

        window = future.loc[
            future.index <= end
        ]

        path = pd.concat([
            pd.Series(
                [ep],
                index=[entry]
            ),
            window
        ])

        dd = (
            max_dd(path)
            if len(path) > 1
            else np.nan
        )

        outcome = (
            "TOO EARLY"
            if pd.isna(m5)
            else (
                "10×+"
                if m5 >= 10
                else (
                    "5×–<10×"
                    if m5 >= 5
                    else (
                        "3×–<5×"
                        if m5 >= 3
                        else "<3×"
                    )
                )
            )
        )

        rows.append({
            "Entry Date":
                entry.date(),
            "Entry Price":
                ep,
            "3Y Multiple":
                m3,
            "5Y Multiple":
                m5,
            "5Y Max DD":
                dd,
            "Outcome":
                outcome
        })

    bt_out = pd.DataFrame(
        rows
    )

    if bt_out.empty:

        st.warning(
            "No historical signals found."
        )

    else:

        matured = (
            bt_out
            .dropna(
                subset=[
                    "5Y Multiple"
                ]
            )
        )

        q = st.columns(5)

        q[0].metric(
            "Signals",
            len(bt_out)
        )

        q[1].metric(
            "Matured",
            len(matured)
        )

        q[2].metric(
            "10×+",
            int(
                (
                    matured[
                        "5Y Multiple"
                    ] >= 10
                ).sum()
            )
        )

        q[3].metric(
            "5×–<10×",
            int(
                (
                    (
                        matured[
                            "5Y Multiple"
                        ] >= 5
                    )
                    &
                    (
                        matured[
                            "5Y Multiple"
                        ] < 10
                    )
                ).sum()
            )
        )

        q[4].metric(
            "3×–<5×",
            int(
                (
                    (
                        matured[
                            "5Y Multiple"
                        ] >= 3
                    )
                    &
                    (
                        matured[
                            "5Y Multiple"
                        ] < 5
                    )
                ).sum()
            )
        )

        st.dataframe(
            bt_out.sort_values(
                "Entry Date",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

        if not matured.empty:

            b = st.columns(3)

            b[0].metric(
                "5Y 3×+ Win Rate",
                f"{(
                    matured[
                        '5Y Multiple'
                    ] >= 3
                ).mean()*100:.1f}%"
            )

            b[1].metric(
                "5Y 5×+ Win Rate",
                f"{(
                    matured[
                        '5Y Multiple'
                    ] >= 5
                ).mean()*100:.1f}%"
            )

            b[2].metric(
                "Worst 5Y DD",
                f"{matured['5Y Max DD'].min():.1f}%"
            )


# ============================================================
# PORTFOLIO PRIVACY
# ============================================================
def portfolio_access():

    # Password must be stored in Streamlit Secrets.
    try:
        password = st.secrets[
            "PORTFOLIO_PASSWORD"
        ]
    except Exception:

        st.error(
            "Private Portfolio is not configured yet."
        )

        st.info(
            "Add PORTFOLIO_PASSWORD to "
            "Streamlit Secrets before using this page."
        )

        return False

    if st.session_state.get(
        "portfolio_authenticated",
        False
    ):
        return True

    st.subheader(
        "🔒 Private Portfolio"
    )

    st.caption(
        "Your holdings, transactions, returns and "
        "wealth targets are private."
    )

    entered = st.text_input(
        "Portfolio password",
        type="password"
    )

    if st.button(
        "Unlock Portfolio",
        type="primary"
    ):

        if entered == password:

            st.session_state[
                "portfolio_authenticated"
            ] = True

            st.rerun()

        else:

            st.error(
                "Incorrect password."
            )

    return False


# ============================================================
# XIRR
# ============================================================
def calculate_xirr(
    cashflows
):

    if (
        cashflows is None
        or len(cashflows) < 2
    ):
        return np.nan

    cf = cashflows.copy()

    cf["Date"] = pd.to_datetime(
        cf["Date"],
        errors="coerce"
    )

    cf["Amount"] = pd.to_numeric(
        cf["Amount"],
        errors="coerce"
    )

    cf = (
        cf.dropna(
            subset=[
                "Date",
                "Amount"
            ]
        )
        .sort_values("Date")
    )

    if len(cf) < 2:
        return np.nan

    amounts = cf[
        "Amount"
    ].to_numpy(
        dtype=float
    )

    dates = cf[
        "Date"
    ].to_numpy()

    if not (
        np.any(amounts < 0)
        and np.any(amounts > 0)
    ):
        return np.nan

    days = np.array([
        (
            d - dates[0]
        )
        .astype(
            "timedelta64[D]"
        )
        .astype(int)
        for d in dates
    ]) / 365.25

    def npv(rate):

        if rate <= -0.999999:
            return np.nan

        return np.sum(
            amounts /
            (
                (1 + rate)
                ** days
            )
        )

    # Wide range of candidate rates.
    rates = np.concatenate([
        np.linspace(
            -0.99,
            -0.01,
            100
        ),
        np.linspace(
            0,
            1,
            201
        ),
        np.linspace(
            1.01,
            10,
            180
        )
    ])

    values = []

    for r in rates:

        try:
            values.append(
                npv(r)
            )
        except Exception:
            values.append(np.nan)

    for i in range(
        len(rates) - 1
    ):

        y1 = values[i]
        y2 = values[i + 1]

        if (
            pd.isna(y1)
            or pd.isna(y2)
        ):
            continue

        if y1 == 0:
            return rates[i]

        if y1 * y2 < 0:

            low = rates[i]
            high = rates[i + 1]

            for _ in range(100):

                mid = (
                    low + high
                ) / 2

                ym = npv(mid)

                if pd.isna(ym):
                    return np.nan

                if abs(ym) < 1e-9:
                    return mid

                if y1 * ym <= 0:

                    high = mid
                    y2 = ym

                else:

                    low = mid
                    y1 = ym

            return (
                low + high
            ) / 2

    return np.nan


# ============================================================
# PORTFOLIO ENGINE
# ============================================================
def render_portfolio_page():

    if not portfolio_access():
        return

    st.title(
        "💼 Private Portfolio"
    )

    st.caption(
        "Private capital allocation engine • "
        "Actual transactions • XIRR • Wealth milestones"
    )

    # --------------------------------------------------------
    # SESSION STORAGE
    # --------------------------------------------------------
    if "portfolio_txns" not in st.session_state:

        st.session_state[
            "portfolio_txns"
        ] = pd.DataFrame(
            columns=[
                "Date",
                "Type",
                "Ticker",
                "Quantity",
                "Price",
                "Cash Amount"
            ]
        )

    # --------------------------------------------------------
    # TRANSACTION LEDGER
    # --------------------------------------------------------
    st.subheader(
        "1. Portfolio Transaction Ledger"
    )

    st.caption(
        "BUY / SELL = shares | "
        "DEPOSIT / WITHDRAWAL = external money | "
        "DIVIDEND = dividend received"
    )

    if (
        st.session_state[
            "portfolio_txns"
        ].empty
    ):

        template = pd.DataFrame({
            "Date": [
                pd.Timestamp.today().date()
            ],
            "Type": [
                "BUY"
            ],
            "Ticker": [
                ""
            ],
            "Quantity": [
                0.0
            ],
            "Price": [
                0.0
            ],
            "Cash Amount": [
                0.0
            ]
        })

    else:

        template = (
            st.session_state[
                "portfolio_txns"
            ]
            .copy()
        )

    edited = st.data_editor(
        template,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={

            "Date":
                st.column_config.DateColumn(
                    "Date"
                ),

            "Type":
                st.column_config.SelectboxColumn(
                    "Type",
                    options=[
                        "BUY",
                        "SELL",
                        "DIVIDEND",
                        "DEPOSIT",
                        "WITHDRAWAL"
                    ]
                ),

            "Ticker":
                st.column_config.TextColumn(
                    "Ticker"
                ),

            "Quantity":
                st.column_config.NumberColumn(
                    "Quantity",
                    min_value=0.0,
                    step=1.0
                ),

            "Price":
                st.column_config.NumberColumn(
                    "Price",
                    min_value=0.0,
                    format="₹%.2f"
                ),

            "Cash Amount":
                st.column_config.NumberColumn(
                    "Cash Amount",
                    min_value=0.0,
                    format="₹%.2f"
                ),
        },
        key="portfolio_editor"
    )

    if st.button(
        "Save Transactions",
        type="primary"
    ):

        x = edited.copy()

        x["Date"] = pd.to_datetime(
            x["Date"],
            errors="coerce"
        )

        x["Type"] = (
            x["Type"]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        x["Ticker"] = (
            x["Ticker"]
            .astype(str)
            .str.upper()
            .str.replace(
                ".NS",
                "",
                regex=False
            )
            .str.strip()
        )

        for col in [
            "Quantity",
            "Price",
            "Cash Amount"
        ]:

            x[col] = pd.to_numeric(
                x[col],
                errors="coerce"
            )

        valid_types = {
            "BUY",
            "SELL",
            "DIVIDEND",
            "DEPOSIT",
            "WITHDRAWAL"
        }

        x = x[
            x["Date"].notna()
            &
            x["Type"].isin(
                valid_types
            )
        ].copy()

        st.session_state[
            "portfolio_txns"
        ] = x.reset_index(
            drop=True
        )

        st.success(
            f"Saved {len(x)} transaction(s)."
        )

    txns = (
        st.session_state[
            "portfolio_txns"
        ].copy()
    )

    if txns.empty:

        st.info(
            "Enter your actual portfolio transactions "
            "to activate the private portfolio engine."
        )

        return

    # --------------------------------------------------------
    # HOLDINGS + COST BASIS
    # --------------------------------------------------------
    holdings = {}

    cash_balance = 0.0
    realized_pnl = 0.0
    total_dividends = 0.0

    for _, r in (
        txns.sort_values(
            "Date"
        ).iterrows()
    ):

        ticker = (
            str(r["Ticker"])
            .strip()
            .upper()
        )

        txn_type = (
            str(r["Type"])
            .strip()
            .upper()
        )

        qty = clean_num(
            r["Quantity"]
        )

        price = clean_num(
            r["Price"]
        )

        cash_amount = clean_num(
            r["Cash Amount"]
        )

        # ----------------------------------------------------
        # DEPOSIT
        # ----------------------------------------------------
        if txn_type == "DEPOSIT":

            if (
                not pd.isna(
                    cash_amount
                )
                and cash_amount > 0
            ):
                cash_balance += (
                    cash_amount
                )

            continue

        # ----------------------------------------------------
        # WITHDRAWAL
        # ----------------------------------------------------
        if txn_type == "WITHDRAWAL":

            if (
                not pd.isna(
                    cash_amount
                )
                and cash_amount > 0
            ):
                cash_balance -= (
                    cash_amount
                )

            continue

        # ----------------------------------------------------
        # DIVIDEND
        # ----------------------------------------------------
        if txn_type == "DIVIDEND":

            amount = cash_amount

            if (
                not pd.isna(amount)
                and amount > 0
            ):

                cash_balance += amount
                total_dividends += amount

            continue

        if not ticker:
            continue

        if ticker not in holdings:

            holdings[ticker] = {
                "Quantity": 0.0,
                "Cost": 0.0
            }

        # ----------------------------------------------------
        # BUY
        # ----------------------------------------------------
        if txn_type == "BUY":

            if (
                pd.isna(qty)
                or qty <= 0
                or pd.isna(price)
                or price <= 0
            ):
                continue

            cost = (
                qty * price
            )

            holdings[ticker][
                "Quantity"
            ] += qty

            holdings[ticker][
                "Cost"
            ] += cost

            cash_balance -= cost

        # ----------------------------------------------------
        # SELL
        # ----------------------------------------------------
        elif txn_type == "SELL":

            if (
                pd.isna(qty)
                or qty <= 0
                or pd.isna(price)
                or price <= 0
            ):
                continue

            current_qty = holdings[
                ticker
            ]["Quantity"]

            current_cost = holdings[
                ticker
            ]["Cost"]

            if current_qty <= 0:
                continue

            sell_qty = min(
                qty,
                current_qty
            )

            average_cost = (
                current_cost /
                current_qty
            )

            cost_removed = (
                sell_qty *
                average_cost
            )

            proceeds = (
                sell_qty *
                price
            )

            realized_pnl += (
                proceeds -
                cost_removed
            )

            holdings[ticker][
                "Quantity"
            ] -= sell_qty

            holdings[ticker][
                "Cost"
            ] -= cost_removed

            cash_balance += proceeds

    # Remove zero holdings
    holdings = {
        k: v
        for k, v in holdings.items()
        if v["Quantity"] > 0.000001
    }

    # --------------------------------------------------------
    # MARKET VALUE
    # --------------------------------------------------------
    rows = []

    for ticker, h in holdings.items():

        qty = h["Quantity"]
        cost = h["Cost"]

        with st.spinner(
            f"Updating {ticker}..."
        ):

            a = analyse_symbol(
                ticker,
                need_history=False
            )

        if a is None:

            rows.append({
                "Ticker":
                    ticker,
                "Quantity":
                    qty,
                "Invested":
                    cost,
                "Current Value":
                    np.nan,
                "P&L":
                    np.nan,
                "P&L %":
                    np.nan,
                "Score":
                    np.nan,
                "Decision":
                    "DATA UNAVAILABLE"
            })

            continue

        current_price = (
            a["price"]
        )

        current_value = (
            qty *
            current_price
        )

        pnl = (
            current_value -
            cost
        )

        rows.append({

            "Ticker":
                ticker,

            "Quantity":
                qty,

            "Avg Buy":
                cost / qty
                if qty > 0
                else np.nan,

            "Price":
                current_price,

            "Invested":
                cost,

            "Current Value":
                current_value,

            "P&L":
                pnl,

            "P&L %":
                (
                    pnl /
                    cost *
                    100
                    if cost > 0
                    else np.nan
                ),

            "Allocation %":
                np.nan,

            "Score":
                a["score"],

            "5Y Model":
                a["multiple"],

            "Entry":
                (
                    "YES"
                    if a[
                        "entry_trigger"
                    ]
                    else "NO"
                ),

            "Exit":
                (
                    "YES"
                    if a[
                        "exit_trigger"
                    ]
                    else "NO"
                ),

            "Decision":
                a["verdict"],

            "ROE":
                a["roe_pct"],

            "Debt/Equity":
                a["de"]
        })

    out = pd.DataFrame(
        rows
    )

    if out.empty:

        st.warning(
            "No current holdings found."
        )

        return

    valid = out[
        out[
            "Current Value"
        ].notna()
    ].copy()

    if valid.empty:

        st.warning(
            "Current holdings could not be priced."
        )

        return

    # --------------------------------------------------------
    # PORTFOLIO TOTALS
    # --------------------------------------------------------
    total_stock_value = (
        valid[
            "Current Value"
        ].sum()
    )

    total_invested = (
        valid[
            "Invested"
        ].sum()
    )

    total_pnl = (
        valid[
            "P&L"
        ].sum()
    )

    total_portfolio_value = (
        total_stock_value +
        cash_balance
    )

    if (
        total_stock_value > 0
    ):

        valid[
            "Allocation %"
        ] = (
            valid[
                "Current Value"
            ]
            /
            total_stock_value
            *
            100
        )

    # --------------------------------------------------------
    # PRIVATE DASHBOARD
    # --------------------------------------------------------
    st.subheader(
        "2. Where Is My Money Now?"
    )

    c = st.columns(6)

    c[0].metric(
        "Invested",
        f"₹{total_invested:,.0f}"
    )

    c[1].metric(
        "Stocks Value",
        f"₹{total_stock_value:,.0f}"
    )

    c[2].metric(
        "Cash",
        f"₹{cash_balance:,.0f}"
    )

    c[3].metric(
        "Portfolio Value",
        f"₹{total_portfolio_value:,.0f}"
    )

    c[4].metric(
        "Unrealised P&L",
        f"₹{total_pnl:,.0f}"
    )

    c[5].metric(
        "Holdings",
        len(valid)
    )

    # --------------------------------------------------------
    # XIRR
    # --------------------------------------------------------
    st.subheader(
        "3. Actual Portfolio Return — XIRR"
    )

    xirr_flows = []

    for _, r in txns.iterrows():

        txn_date = pd.to_datetime(
            r["Date"],
            errors="coerce"
        )

        if pd.isna(
            txn_date
        ):
            continue

        txn_type = str(
            r["Type"]
        ).upper().strip()

        amount = clean_num(
            r["Cash Amount"]
        )

        if txn_type == "DEPOSIT":

            if (
                not pd.isna(amount)
                and amount > 0
            ):

                xirr_flows.append({
                    "Date":
                        txn_date,
                    "Amount":
                        -amount
                })

        elif txn_type == "WITHDRAWAL":

            if (
                not pd.isna(amount)
                and amount > 0
            ):

                xirr_flows.append({
                    "Date":
                        txn_date,
                    "Amount":
                        amount
                })

        elif txn_type == "DIVIDEND":

            if (
                not pd.isna(amount)
                and amount > 0
            ):

                xirr_flows.append({
                    "Date":
                        txn_date,
                    "Amount":
                        amount
                })

    # Current total wealth is the terminal value.
    xirr_flows.append({
        "Date":
            pd.Timestamp.today(),
        "Amount":
            total_portfolio_value
    })

    xirr_df = pd.DataFrame(
        xirr_flows
    )

    portfolio_xirr = calculate_xirr(
        xirr_df
    )

    x1, x2, x3 = st.columns(3)

    x1.metric(
        "Portfolio XIRR",
        (
            "Unavailable"
            if pd.isna(
                portfolio_xirr
            )
            else f"{portfolio_xirr*100:.2f}%"
        )
    )

    x2.metric(
        "Realised P&L",
        f"₹{realized_pnl:,.0f}"
    )

    x3.metric(
        "Dividends",
        f"₹{total_dividends:,.0f}"
    )

    st.caption(
        "XIRR uses external deposits/withdrawals, "
        "dividends and current total portfolio wealth. "
        "BUY/SELL transactions are internal portfolio movements "
        "and are not double-counted as external cash flows."
    )

    # --------------------------------------------------------
    # WEALTH MILESTONES
    # --------------------------------------------------------
    st.divider()

    st.subheader(
        "4. 🎯 Dynamic Wealth Milestones"
    )

    st.caption(
        "Your current portfolio value is the starting point. "
        "The primary target automatically moves to the next "
        "milestone after each milestone is crossed."
    )

    milestones = [
        2500000,
        5000000,
        10000000
    ]

    achieved = []

    for m in milestones:

        achieved.append(
            total_portfolio_value >= m
        )

    next_target = None

    for m, done in zip(
        milestones,
        achieved
    ):

        if not done:
            next_target = m
            break

    if next_target is None:

        st.success(
            "🏆 ₹1 Crore milestone achieved."
        )

        next_target = 10000000

    # --------------------------------------------------------
    # Milestone table
    # --------------------------------------------------------
    milestone_rows = []

    for m, done in zip(
        milestones,
        achieved
    ):

        gap = max(
            0,
            m -
            total_portfolio_value
        )

        progress = min(
            100,
            total_portfolio_value /
            m *
            100
        )

        if done:

            status = "ACHIEVED"

        else:

            status = (
                "PRIMARY TARGET"
                if m == next_target
                else "NEXT"
            )

        milestone_rows.append({
            "Milestone":
                f"₹{m/100000:.0f}L",
            "Current Value":
                total_portfolio_value,
            "Gap":
                gap,
            "Progress %":
                progress,
            "Status":
                status
        })

    milestone_df = pd.DataFrame(
        milestone_rows
    )

    st.dataframe(
        milestone_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Current Value":
                st.column_config.NumberColumn(
                    format="₹%,.0f"
                ),
            "Gap":
                st.column_config.NumberColumn(
                    format="₹%,.0f"
                ),
            "Progress %":
                st.column_config.NumberColumn(
                    format="%.1f%%"
                )
        }
    )

    # --------------------------------------------------------
    # NEXT TARGET MATH
    # --------------------------------------------------------
    st.subheader(
        f"Primary Target: ₹{next_target/100000:.0f}L"
    )

    gap = max(
        0,
        next_target -
        total_portfolio_value
    )

    progress = (
        min(
            100,
            total_portfolio_value /
            next_target *
            100
        )
        if next_target > 0
        else 100
    )

    st.progress(
        progress / 100
    )

    mc = st.columns(4)

    mc[0].metric(
        "Current",
        f"₹{total_portfolio_value:,.0f}"
    )

    mc[1].metric(
        "Target",
        f"₹{next_target:,.0f}"
    )

    mc[2].metric(
        "Gap",
        f"₹{gap:,.0f}"
    )

    mc[3].metric(
        "Progress",
        f"{progress:.1f}%"
    )

    # --------------------------------------------------------
    # FUTURE RETURN SCENARIOS
    # --------------------------------------------------------
    st.subheader(
        "5. Time to Next Milestone"
    )

    st.caption(
        "These are planning scenarios, not promises or forecasts."
    )

    scenario_rates = [
        (
            "Conservative",
            0.10
        ),
        (
            "Base",
            0.15
        ),
        (
            "High Growth",
            0.25
        )
    ]

    scenario_rows = []

    for label, rate in (
        scenario_rates
    ):

        if (
            total_portfolio_value > 0
            and next_target >
                total_portfolio_value
        ):

            years_required = (
                np.log(
                    next_target /
                    total_portfolio_value
                )
                /
                np.log(
                    1 + rate
                )
            )

            months_required = (
                years_required *
                12
            )

            milestone_date = (
                pd.Timestamp.today()
                +
                pd.DateOffset(
                    months=int(
                        np.ceil(
                            months_required
                        )
                    )
                )
            )

        else:

            years_required = 0
            months_required = 0
            milestone_date = (
                pd.Timestamp.today()
            )

        scenario_rows.append({
            "Scenario":
                label,
            "Assumed CAGR":
                rate * 100,
            "Years":
                years_required,
            "Months":
                months_required,
            "Estimated Date":
                milestone_date.date()
        })

    scenario_df = pd.DataFrame(
        scenario_rows
    )

    st.dataframe(
        scenario_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Assumed CAGR":
                st.column_config.NumberColumn(
                    format="%.1f%%"
                ),
            "Years":
                st.column_config.NumberColumn(
                    format="%.1f"
                ),
            "Months":
                st.column_config.NumberColumn(
                    format="%.0f"
                )
        }
    )

    # --------------------------------------------------------
    # MOVE THE NEEDLE
    # --------------------------------------------------------
    st.subheader(
        "6. 💰 Move the Needle"
    )

    deploy = st.number_input(
        "Additional capital available now",
        min_value=0.0,
        value=100000.0,
        step=25000.0,
        format="%.0f"
    )

    after_deployment = (
        total_portfolio_value +
        deploy
    )

    old_gap = max(
        0,
        next_target -
        total_portfolio_value
    )

    new_gap = max(
        0,
        next_target -
        after_deployment
    )

    gap_reduction = (
        old_gap -
        new_gap
    )

    if old_gap > 0:

        needle = (
            gap_reduction /
            old_gap *
            100
        )

    else:

        needle = 0

    nd = st.columns(4)

    nd[0].metric(
        "Current Gap",
        f"₹{old_gap:,.0f}"
    )

    nd[1].metric(
        "After Deployment",
        f"₹{new_gap:,.0f}"
    )

    nd[2].metric(
        "Gap Reduced",
        f"₹{gap_reduction:,.0f}"
    )

    nd[3].metric(
        "Move the Needle",
        f"{needle:.1f}%"
    )

    st.info(
        f"Deploying ₹{deploy:,.0f} increases the "
        f"portfolio value from ₹{total_portfolio_value:,.0f} "
        f"to ₹{after_deployment:,.0f}, assuming the capital "
        f"is actually invested. The next version will connect "
        f"this directly to the stock-ranking/capital-allocation engine."
    )

    # --------------------------------------------------------
    # HOLDINGS
    # --------------------------------------------------------
    st.divider()

    st.subheader(
        "7. Current Holdings"
    )

    st.dataframe(
        valid,
        use_container_width=True,
        hide_index=True,
        column_config={

            "Avg Buy":
                st.column_config.NumberColumn(
                    format="₹%.2f"
                ),

            "Price":
                st.column_config.NumberColumn(
                    format="₹%.2f"
                ),

            "Invested":
                st.column_config.NumberColumn(
                    format="₹%,.0f"
                ),

            "Current Value":
                st.column_config.NumberColumn(
                    format="₹%,.0f"
                ),

            "P&L":
                st.column_config.NumberColumn(
                    format="₹%,.0f"
                ),

            "P&L %":
                st.column_config.NumberColumn(
                    format="%.1f%%"
                ),

            "Allocation %":
                st.column_config.NumberColumn(
                    format="%.1f%%"
                ),

            "5Y Model":
                st.column_config.NumberColumn(
                    format="%.1f×"
                ),

            "ROE":
                st.column_config.NumberColumn(
                    format="%.1f%%"
                ),

            "Debt/Equity":
                st.column_config.NumberColumn(
                    format="%.1f%%"
                )
        }
    )

    # --------------------------------------------------------
    # PORTFOLIO SIGNALS
    # --------------------------------------------------------
    st.subheader(
        "8. Portfolio Signals"
    )

    s1, s2, s3, s4 = st.columns(4)

    s1.metric(
        "Exit Triggers",
        int(
            (
                valid["Exit"]
                == "YES"
            ).sum()
        )
    )

    s2.metric(
        "Entry Triggers",
        int(
            (
                valid["Entry"]
                == "YES"
            ).sum()
        )
    )

    top_idx = (
        valid[
            "Allocation %"
        ].idxmax()
    )

    s3.metric(
        "Largest Holding",
        f"{valid.loc[top_idx,'Ticker']} "
        f"({valid.loc[top_idx,'Allocation %']:.1f}%)"
    )

    s4.metric(
        "Cash %",
        (
            cash_balance /
            total_portfolio_value *
            100
            if total_portfolio_value > 0
            else 0
        )
    )

    if (
        total_stock_value > 0
    ):

        st.bar_chart(
            valid.set_index(
                "Ticker"
            )[
                "Allocation %"
            ]
        )

    # --------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------
    st.subheader(
        "9. Private Data Export"
    )

    st.download_button(
        "Export Portfolio Analysis",
        valid.to_csv(
            index=False
        ),
        "portfolio_analysis.csv",
        "text/csv"
    )

    st.download_button(
        "Export Transaction Ledger",
        txns.to_csv(
            index=False
        ),
        "portfolio_transactions.csv",
        "text/csv"
    )

    # --------------------------------------------------------
    # LOGOUT
    # --------------------------------------------------------
    if st.button(
        "Lock Portfolio"
    ):

        st.session_state[
            "portfolio_authenticated"
        ] = False

        st.rerun()


# ============================================================
# NSE UNIVERSE
# ============================================================
@st.cache_data(
    ttl=86400,
    show_spinner=False
)
def get_nse_universe():

    urls = [
        "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
        "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    ]

    for url in urls:

        try:

            import requests

            r = requests.get(
                url,
                headers={
                    "User-Agent":
                        "Mozilla/5.0"
                },
                timeout=20
            )

            r.raise_for_status()

            df = pd.read_csv(
                io.BytesIO(
                    r.content
                )
            )

            if "SYMBOL" in df.columns:

                cols = [
                    c
                    for c in [
                        "SYMBOL",
                        "NAME OF COMPANY",
                        "ISIN NUMBER"
                    ]
                    if c in df.columns
                ]

                return (
                    df[cols]
                    .drop_duplicates(
                        "SYMBOL"
                    )
                )

        except Exception:
            pass

    return pd.DataFrame()


# ============================================================
# ATH SCANNER
# ============================================================
def render_ath_page():

    st.title(
        "🚀 ATH Breakouts"
    )

    st.caption(
        "New all-time highs using adjusted daily prices. "
        "The production version will rank the universe by "
        "market capitalisation and liquidity before scanning."
    )

    universe = get_nse_universe()

    uploaded = st.file_uploader(
        "Optional NSE security master CSV",
        type=["csv"],
        key="ath_universe"
    )

    if uploaded is not None:

        try:

            u = pd.read_csv(
                uploaded
            )

            symcol = next(
                (
                    c
                    for c in u.columns
                    if c.upper().strip()
                    in [
                        "SYMBOL",
                        "TICKER",
                        "NSE SYMBOL"
                    ]
                ),
                None
            )

            if symcol:

                universe = pd.DataFrame({
                    "SYMBOL":
                        u[symcol]
                        .astype(str)
                        .str.upper()
                        .str.replace(
                            ".NS",
                            "",
                            regex=False
                        )
                })

        except Exception as e:

            st.error(
                f"Upload failed: {e}"
            )

    if universe.empty:

        st.warning(
            "NSE universe unavailable. "
            "Upload the security master CSV."
        )

        return

    universe = (
        universe[
            universe["SYMBOL"].notna()
        ]
        .copy()
    )

    universe["SYMBOL"] = (
        universe["SYMBOL"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    exclude = {
        "NIFTY",
        "NIFTYBEES"
    }

    universe = universe[
        ~universe[
            "SYMBOL"
        ].isin(exclude)
    ]

    st.write(
        f"Universe: **{len(universe):,} symbols**"
    )

    c1, c2, c3 = st.columns(3)

    lookback = c1.selectbox(
        "Recent ATH window",
        [1, 3, 6, 12, 24],
        index=4
    )

    max_symbols = c2.number_input(
        "Max symbols",
        100,
        3000,
        min(
            1000,
            len(universe)
        ),
        100
    )

    min_score = c3.slider(
        "Minimum score",
        0,
        100,
        0,
        5
    )

    if not st.button(
        "Scan ATH Universe",
        type="primary"
    ):
        return

    end = (
        pd.Timestamp.today()
        .normalize()
    )

    start = (
        end -
        pd.DateOffset(
            months=lookback
        )
    )

    symbols = (
        universe[
            "SYMBOL"
        ]
        .head(
            int(max_symbols)
        )
        .tolist()
    )

    results = []

    progress = st.progress(
        0
    )

    status = st.empty()

    for i, ticker in enumerate(
        symbols
    ):

        status.write(
            f"Scanning {ticker} "
            f"({i+1}/{len(symbols)})"
        )

        try:

            px = load_price(
                normalize_symbol(
                    ticker
                ),
                "max"
            )

            close = close_series(
                px
            )

            if len(close) < 200:
                continue

            hist = close[
                close.index <= end
            ]

            # IMPORTANT:
            # Compare today's price with the
            # previous cumulative high.
            # This identifies a NEW ATH rather
            # than repeated equal-price observations.
            prior_max = (
                hist.shift(1)
                .cummax()
            )

            new_ath = (
                hist > prior_max
            )

            recent = new_ath[
                (
                    new_ath.index >= start
                )
                &
                (
                    new_ath.index <= end
                )
            ]

            if recent.empty:
                continue

            ath_date = (
                recent.index[-1]
            )

            ath_price = float(
                hist.loc[
                    ath_date
                ]
            )

            current = float(
                hist.iloc[-1]
            )

            from_ath = (
                current /
                ath_price -
                1
            ) * 100

            a = analyse_symbol(
                ticker,
                need_history=False
            )

            if (
                a is not None
                and a["score"] < min_score
            ):
                continue

            results.append({

                "Ticker":
                    ticker,

                "Company":
                    (
                        a["name"]
                        if a
                        else ticker
                    ),

                "New ATH Date":
                    ath_date.date(),

                "ATH Price":
                    ath_price,

                "Current":
                    current,

                "From ATH %":
                    from_ath,

                "Score":
                    (
                        a["score"]
                        if a
                        else np.nan
                    ),

                "Entry":
                    (
                        "YES"
                        if (
                            a
                            and
                            a[
                                "entry_trigger"
                            ]
                        )
                        else "NO"
                    ),

                "5Y Model":
                    (
                        a["multiple"]
                        if a
                        else np.nan
                    )
            })

        except Exception:
            pass

        progress.progress(
            (i + 1) /
            len(symbols)
        )

    status.empty()
    progress.empty()

    if not results:

        st.warning(
            "No recent new ATH candidates found."
        )

        return

    out = pd.DataFrame(
        results
    ).sort_values(
        [
            "New ATH Date",
            "Score"
        ],
        ascending=[
            False,
            False
        ]
    )

    st.success(
        f"Found {len(out)} recent new ATH candidate(s)."
    )

    st.dataframe(
        out,
        use_container_width=True,
        hide_index=True,
        column_config={

            "ATH Price":
                st.column_config.NumberColumn(
                    format="₹%.2f"
                ),

            "Current":
                st.column_config.NumberColumn(
                    format="₹%.2f"
                ),

            "From ATH %":
                st.column_config.NumberColumn(
                    format="%.1f%%"
                ),

            "5Y Model":
                st.column_config.NumberColumn(
                    format="%.1f×"
                )
        }
    )

    st.download_button(
        "Export ATH Scan",
        out.to_csv(
            index=False
        ),
        "ath_scan.csv",
        "text/csv"
    )


# ============================================================
# MAIN
# ============================================================
def main():

    st.sidebar.title(
        "Multibagger Lab"
    )

    page = st.sidebar.radio(
        "Page",
        [
            "Stock Analysis",
            "Portfolio 🔒",
            "ATH Scanner"
        ]
    )

    if page == "Stock Analysis":

        render_stock_page()

    elif page == "Portfolio 🔒":

        render_portfolio_page()

    else:

        render_ath_page()

    st.sidebar.divider()

    st.sidebar.caption(
        "Public market research is separate from "
        "private portfolio data."
    )

    st.sidebar.caption(
        "Prototype market data: Yahoo Finance/yfinance EOD."
    )


main()
