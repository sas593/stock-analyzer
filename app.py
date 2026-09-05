import io
import zipfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Multibagger Lab", layout="wide")

# ============================================================
# Helpers
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


@st.cache_data(ttl=900, show_spinner=False)
def load_price(symbol, period="max"):
    return yf.download(
        symbol, period=period, interval="1d", auto_adjust=True,
        progress=False, threads=False, group_by="column"
    )


@st.cache_data(ttl=900, show_spinner=False)
def load_fundamentals(symbol):
    t = yf.Ticker(symbol)
    return {
        "info": t.info,
        "income": t.income_stmt,
        "balance": t.balance_sheet,
        "cashflow": t.cashflow,
    }


def close_series(df):
    if df is None or df.empty or "Close" not in df:
        return pd.Series(dtype=float)
    x = df["Close"]
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    x = pd.to_numeric(x, errors="coerce").dropna()
    x.index = pd.to_datetime(x.index).tz_localize(None)
    return x.sort_index()


def row_value(df, names):
    if df is None or df.empty:
        return np.nan
    for name in names:
        if name in df.index:
            s = pd.to_numeric(df.loc[name], errors="coerce").dropna()
            if not s.empty:
                return float(s.iloc[0])
    return np.nan


def annual_series(df, names):
    if df is None or df.empty:
        return pd.Series(dtype=float)
    for name in names:
        if name in df.index:
            s = pd.to_numeric(df.loc[name], errors="coerce").dropna()
            if len(s) >= 2:
                s.index = pd.to_datetime(s.index)
                return s.sort_index()
    return pd.Series(dtype=float)


def cagr_from_series(s, years=3):
    if len(s) < 2:
        return np.nan
    s = s[s > 0]
    if len(s) < 2:
        return np.nan
    start_i = max(0, len(s) - years - 1)
    start = float(s.iloc[start_i])
    end = float(s.iloc[-1])
    actual_years = max(1, len(s.iloc[start_i:]) - 1)
    if start <= 0 or end <= 0:
        return np.nan
    return (end / start) ** (1 / actual_years) - 1


def max_dd(s):
    if len(s) == 0:
        return np.nan
    peak = s.cummax()
    return ((s / peak) - 1).min() * 100


def forward_multiple(close, entry, years):
    target_date = entry + pd.DateOffset(years=years)
    future = close.loc[close.index >= target_date]
    if future.empty:
        return np.nan
    return float(future.iloc[0] / close.loc[entry])


def fmt_pct(x):
    return "Unavailable" if pd.isna(x) else f"{x:.1f}%"


def fmt_x(x):
    return "Unavailable" if pd.isna(x) else f"{x:.1f}×"


def safe_growth_score(earnings_cagr, revenue_cagr):
    if not pd.isna(earnings_cagr):
        return 25 if earnings_cagr >= .25 else (19 if earnings_cagr >= .15 else (12 if earnings_cagr >= .08 else 5))
    if not pd.isna(revenue_cagr):
        return 22 if revenue_cagr >= .25 else (16 if revenue_cagr >= .15 else (9 if revenue_cagr >= .08 else 4))
    return 0


def analyse_symbol(symbol, need_history=True):
    symbol = normalize_symbol(symbol)
    if not symbol:
        return None
    price_df = load_price(symbol, "max" if need_history else "1y")
    close = close_series(price_df)
    if close.empty:
        return None

    price = float(close.iloc[-1])
    latest_date = close.index[-1]
    fund = load_fundamentals(symbol)
    info, income, balance, cashflow = fund["info"], fund["income"], fund["balance"], fund["cashflow"]
    name = info.get("longName") or display_ticker(symbol)

    eps = clean_num(info.get("trailingEps"))
    diluted_eps_s = annual_series(income, ["Diluted EPS", "Basic EPS"])
    if pd.isna(eps) and not diluted_eps_s.empty:
        eps = float(diluted_eps_s.iloc[-1])

    pe = clean_num(info.get("trailingPE"))
    if pd.isna(pe) and not pd.isna(eps) and eps > 0:
        pe = price / eps

    net_income_s = annual_series(income, ["Net Income", "Net Income Common Stockholders"])
    revenue_s = annual_series(income, ["Total Revenue", "Operating Revenue"])
    earnings_cagr = cagr_from_series(diluted_eps_s, 3)
    if pd.isna(earnings_cagr):
        earnings_cagr = cagr_from_series(net_income_s, 3)
    revenue_cagr = cagr_from_series(revenue_s, 3)

    equity = row_value(balance, ["Stockholders Equity", "Total Equity Gross Minority Interest"])
    ni = row_value(income, ["Net Income", "Net Income Common Stockholders"])
    roe = clean_num(info.get("returnOnEquity"))
    if pd.isna(roe) and not pd.isna(ni) and not pd.isna(equity) and equity != 0:
        roe = ni / equity
    roe_pct = roe * 100 if not pd.isna(roe) else np.nan

    debt = row_value(balance, ["Total Debt"])
    de = clean_num(info.get("debtToEquity"))
    if pd.isna(de) and not pd.isna(debt) and not pd.isna(equity) and equity != 0:
        de = debt / equity * 100

    market_cap = clean_num(info.get("marketCap"))
    if pd.isna(market_cap):
        shares = row_value(balance, ["Ordinary Shares Number", "Share Issued"])
        if not pd.isna(shares):
            market_cap = price * shares

    peg = clean_num(info.get("pegRatio"))
    ocf = row_value(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"])
    fcf = row_value(cashflow, ["Free Cash Flow"])
    if pd.isna(fcf):
        capex = row_value(cashflow, ["Capital Expenditure", "Capital Expenditures"])
        if not pd.isna(ocf) and not pd.isna(capex):
            fcf = ocf + capex

    promoter = clean_num(info.get("heldPercentInsiders"))
    pledge = np.nan  # never confuse insider ownership with promoter pledge

    ma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else np.nan
    ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan
    momentum_6m = close.iloc[-1] / close.iloc[-127] - 1 if len(close) >= 127 else np.nan
    entry_conditions = {
        "Price > 200 DMA": price > ma200 if not pd.isna(ma200) else False,
        "50 DMA > 200 DMA": ma50 > ma200 if not pd.isna(ma50) and not pd.isna(ma200) else False,
        "6M Momentum > 0": momentum_6m > 0 if not pd.isna(momentum_6m) else False,
    }
    entry_trigger = all(entry_conditions.values())
    exit_conditions = {
        "Price < 200 DMA": price < ma200 if not pd.isna(ma200) else False,
        "50 DMA < 200 DMA": ma50 < ma200 if not pd.isna(ma50) and not pd.isna(ma200) else False,
    }
    exit_trigger = all(exit_conditions.values())

    de_pass = True if pd.isna(de) else de < 100
    roe_pass = True if pd.isna(roe_pct) else roe_pct > 12
    pe_pass = True if pd.isna(pe) else 0 < pe < 75
    gates = de_pass and roe_pass and pe_pass

    s1 = safe_growth_score(earnings_cagr, revenue_cagr)
    if not pd.isna(roe_pct):
        s2 = 18 if roe_pct > 20 and (pd.isna(de) or de < 30) else (14 if roe_pct > 15 else (10 if roe_pct > 12 else 5))
    else:
        s2 = 0
    if not pd.isna(market_cap):
        mcap_cr = market_cap / 1e7
        s3 = 15 if mcap_cr < 50000 else (11 if mcap_cr < 200000 else 7)
    else:
        s3 = 0
    if not pd.isna(peg) and peg > 0:
        s4 = 15 if peg <= 1 else (11 if peg <= 1.5 else (7 if peg <= 2 else 3))
    elif not pd.isna(pe) and pe > 0:
        s4 = 15 if pe <= 20 else (11 if pe <= 30 else (7 if pe <= 40 else 3))
    else:
        s4 = 0
    s5 = 6
    if entry_trigger:
        s6 = 10
    elif not pd.isna(ma200) and price > ma200 and ma50 > ma200:
        s6 = 8
    elif not pd.isna(ma200) and price > ma200:
        s6 = 6
    else:
        s6 = 3
    s7 = 0
    total = int(s1 + s2 + s3 + s4 + s5 + s6 + s7)

    base_cagr = earnings_cagr if not pd.isna(earnings_cagr) else revenue_cagr
    if pd.isna(base_cagr):
        base_cagr = .15
    exit_pe = pe if not pd.isna(pe) and pe > 0 else 25
    exit_pe = min(exit_pe, 40)
    future_eps = eps * ((1 + base_cagr) ** 5) if not pd.isna(eps) and eps > 0 else np.nan
    target = future_eps * exit_pe if not pd.isna(future_eps) else np.nan
    multiple = target / price if not pd.isna(target) and price > 0 else np.nan

    if exit_trigger:
        verdict = "WATCH / AVOID 🔴"
    elif total >= 80 and gates and entry_trigger:
        verdict = "BUY / ENTRY 🟢"
    elif total >= 65 and gates:
        verdict = "ACCUMULATE 🟡"
    else:
        verdict = "WATCH / AVOID 🔴"

    return {
        "symbol": symbol, "ticker": display_ticker(symbol), "name": name,
        "close": close, "price": price, "latest_date": latest_date,
        "eps": eps, "pe": pe, "roe_pct": roe_pct, "de": de,
        "market_cap": market_cap, "peg": peg, "ocf": ocf, "fcf": fcf,
        "promoter": promoter, "pledge": pledge,
        "earnings_cagr": earnings_cagr, "revenue_cagr": revenue_cagr,
        "ma50": ma50, "ma200": ma200, "momentum_6m": momentum_6m,
        "entry_conditions": entry_conditions, "entry_trigger": entry_trigger,
        "exit_conditions": exit_conditions, "exit_trigger": exit_trigger,
        "de_pass": de_pass, "roe_pass": roe_pass, "pe_pass": pe_pass, "gates": gates,
        "s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5, "s6": s6, "s7": s7,
        "score": total, "base_cagr": base_cagr, "exit_pe": exit_pe,
        "future_eps": future_eps, "target": target, "multiple": multiple,
        "verdict": verdict,
    }


def render_stock_page():
    st.title("📈 Multibagger Lab")
    st.caption("NSE stock analysis • fundamentals • technicals • multibagger model • historical signal study")
    ticker = st.sidebar.text_input("NSE ticker", "DODLA").strip().upper()
    if not ticker:
        return
    with st.spinner(f"Loading latest available data for {ticker}..."):
        a = analyse_symbol(ticker, need_history=True)
    if a is None:
        st.error(f"No usable historical data found for {ticker}.")
        return

    st.header(f"{a['name']} ({a['ticker']})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest Price", f"₹{a['price']:,.2f}", f"As of {a['latest_date'].date()}")
    c2.metric("Score", f"{a['score']}/100")
    c3.metric("5Y Model", fmt_x(a['multiple']))
    c4.metric("Decision", a['verdict'])

    st.divider(); st.subheader("🎯 Live Entry / Exit Trigger")
    e1, e2 = st.columns(2)
    with e1:
        st.success("🟢 ENTRY TRIGGER: YES" if a['entry_trigger'] else "🔴 ENTRY TRIGGER: NO")
        for k, v in a['entry_conditions'].items(): st.write(f"{'✅' if v else '❌'} {k}")
        st.caption(f"Price ₹{a['price']:,.2f} | 50 DMA ₹{a['ma50']:,.2f} | 200 DMA ₹{a['ma200']:,.2f} | 6M momentum {fmt_pct(a['momentum_6m']*100)}")
    with e2:
        st.error("🔴 EXIT TRIGGER: YES" if a['exit_trigger'] else "🟢 EXIT TRIGGER: NO")
        for k, v in a['exit_conditions'].items(): st.write(f"{'⚠️' if v else '✅'} {k}")
        st.caption("Exit rule: closing price below 200 DMA AND 50 DMA below 200 DMA.")

    st.divider(); left, right = st.columns(2)
    with left:
        st.subheader("1. Hard Gatekeeper Check")
        def gate_line(label, value, passed, unit=""):
            if pd.isna(value): st.write(f"⚪ {label}: DATA UNAVAILABLE")
            else: st.write(f"{'✅' if passed else '❌'} {label}: {value:.1f}{unit}")
        gate_line("Debt/Equity", a['de'], a['de_pass'])
        gate_line("ROE", a['roe_pct'], a['roe_pass'], "%")
        gate_line("P/E", a['pe'], a['pe_pass'], "×")
        if a['gates']:
    st.success("All available gatekeeper checks pass.")
else:
    st.error("One or more available gatekeeper checks fail.")
    with right:
        st.subheader("2. Automatic 100-Point Score")
        st.dataframe(pd.DataFrame({"Factor":["Earnings Growth","Business Quality","Future Opportunity","Valuation","Management & Governance","Technical Setup","Catalysts"],"Score":[a['s1'],a['s2'],a['s3'],a['s4'],a['s5'],a['s6'],a['s7']],"Maximum":[25,20,15,15,10,10,5]}), use_container_width=True, hide_index=True)

    st.divider(); st.subheader("3. Latest Fundamental Data")
    f = st.columns(4)
    f[0].metric("EPS", "Unavailable" if pd.isna(a['eps']) else f"₹{a['eps']:.2f}")
    f[1].metric("Earnings CAGR", fmt_pct(a['earnings_cagr']*100) if not pd.isna(a['earnings_cagr']) else "Unavailable")
    f[2].metric("Revenue CAGR", fmt_pct(a['revenue_cagr']*100) if not pd.isna(a['revenue_cagr']) else "Unavailable")
    f[3].metric("Market Cap", "Unavailable" if pd.isna(a['market_cap']) else f"₹{a['market_cap']/1e7:,.0f} Cr")
    g = st.columns(4)
    g[0].metric("ROE", fmt_pct(a['roe_pct']))
    g[1].metric("P/E", fmt_x(a['pe']))
    g[2].metric("Debt/Equity", "Unavailable" if pd.isna(a['de']) else f"{a['de']:.1f}%")
    g[3].metric("Operating Cash Flow", "Unavailable" if pd.isna(a['ocf']) else f"₹{a['ocf']/1e7:,.0f} Cr")
    st.caption("Unavailable is not zero and is not treated as a financial failure. Promoter pledge is not inferred from insider ownership.")

    st.divider(); st.subheader("4. 5-Year Multibagger Model")
    m1, m2 = st.columns(2)
    with m1:
        st.write(f"Current EPS: {'Unavailable' if pd.isna(a['eps']) else f'₹{a['eps']:.2f}'}")
        st.write(f"Automatic earnings/revenue growth assumption: {a['base_cagr']*100:.1f}%")
        st.write(f"Modeled exit P/E: {a['exit_pe']:.1f}×")
        st.write(f"Projected 5Y EPS: {'Unavailable' if pd.isna(a['future_eps']) else f'₹{a['future_eps']:.2f}'}")
        st.write(f"Modeled 5Y price: {'Unavailable' if pd.isna(a['target']) else f'₹{a['target']:,.2f}'}")
    with m2:
        if pd.isna(a['multiple']): st.warning("Insufficient EPS data for a 5-year multiple model.")
        elif a['multiple'] >= 10: st.success(f"10×+ model candidate — {a['multiple']:.1f}×")
        elif a['multiple'] >= 5: st.success(f"5×+ model candidate — {a['multiple']:.1f}×")
        elif a['multiple'] >= 3: st.warning(f"3× model candidate — {a['multiple']:.1f}×")
        else: st.info(f"Below 3× model — {a['multiple']:.1f}×")

    st.divider(); st.subheader("5. Price Trend")
    st.line_chart(pd.DataFrame({"Price":a['close'],"50 DMA":a['close'].rolling(50).mean(),"200 DMA":a['close'].rolling(200).mean()}).tail(750))
    st.metric("Maximum Drawdown (full history)", f"{max_dd(a['close']):.1f}%")

    st.divider(); st.header("🔬 Historical Multibagger Backtest")
    st.info("Price-based signal study only. Entry = Price > 200 DMA + 50 DMA > 200 DMA + 6M momentum > 0. It is not a point-in-time fundamental backtest.")
    bt = pd.DataFrame({"Close":a['close']})
    bt["MA50"] = bt["Close"].rolling(50).mean(); bt["MA200"] = bt["Close"].rolling(200).mean(); bt["Momentum6M"] = bt["Close"].pct_change(126)
    bt["Entry"] = (bt["Close"] > bt["MA200"]) & (bt["MA50"] > bt["MA200"]) & (bt["Momentum6M"] > 0)
    dates = bt.index[bt["Entry"].fillna(False)]
    selected=[]; last=None
    for dt in dates:
        if last is None or (dt-last).days >= 90: selected.append(dt); last=dt
    rows=[]
    for entry in selected:
        ep=float(a['close'].loc[entry]); m3=forward_multiple(a['close'],entry,3); m5=forward_multiple(a['close'],entry,5)
        future=a['close'].loc[a['close'].index>entry]; end=entry+pd.DateOffset(years=5); window=future.loc[future.index<=end]; path=pd.concat([pd.Series([ep],index=[entry]),window]); dd=max_dd(path) if len(path)>1 else np.nan
        outcome="TOO EARLY" if pd.isna(m5) else ("10×+" if m5>=10 else ("5×–<10×" if m5>=5 else ("3×–<5×" if m5>=3 else "<3×")))
        rows.append({"Entry Date":entry.date(),"Entry Price":ep,"3Y Multiple":m3,"5Y Multiple":m5,"5Y Max DD":dd,"Outcome":outcome})
    bt_out=pd.DataFrame(rows)
    if bt_out.empty: st.warning("No historical entry signals found.")
    else:
        matured=bt_out.dropna(subset=["5Y Multiple"]); too_early=bt_out["5Y Multiple"].isna().sum()
        q=st.columns(5); q[0].metric("Signals",len(bt_out)); q[1].metric("Matured",len(matured)); q[2].metric("10×+",int((matured['5Y Multiple']>=10).sum())); q[3].metric("5×–<10×",int(((matured['5Y Multiple']>=5)&(matured['5Y Multiple']<10)).sum())); q[4].metric("3×–<5×",int(((matured['5Y Multiple']>=3)&(matured['5Y Multiple']<5)).sum()))
        st.dataframe(bt_out.sort_values("Entry Date",ascending=False),use_container_width=True,hide_index=True)
        if not matured.empty:
            b=st.columns(3); b[0].metric("5Y 3×+ Win Rate",f"{(matured['5Y Multiple']>=3).mean()*100:.1f}%"); b[1].metric("5Y 5×+ Win Rate",f"{(matured['5Y Multiple']>=5).mean()*100:.1f}%"); b[2].metric("Worst Signal 5Y DD",f"{matured['5Y Max DD'].min():.1f}%")
        if too_early: st.caption(f"{too_early} recent signal(s) are TOO EARLY to judge over 5 years.")

    st.divider(); st.header("💰 ₹10L → ₹1Cr Math")
    p=st.columns(3); start=p[0].number_input("Starting capital",100000,100000000,1000000,100000); target_cap=p[1].number_input("Target capital",1000000,1000000000,10000000,1000000); years=p[2].slider("Years",3,15,5)
    req=((target_cap/start)**(1/years)-1)*100; st.metric("Required CAGR",f"{req:.2f}%")
    st.caption("Prototype data source: Yahoo Finance/yfinance EOD. Production PMS use should move to licensed market/fundamental data. NSE's official reports distinguish corporate-action-adjusted 52-week data from raw bhavcopy prices.")


def render_portfolio_page():
    st.title("💼 Portfolio Dashboard")
    st.caption("Manual holdings + Excel/CSV upload use the same portfolio engine.")
    if "portfolio_df" not in st.session_state:
        st.session_state.portfolio_df = pd.DataFrame(columns=["Ticker","Quantity","Average Buy Price"])

    upload = st.file_uploader("Upload Portfolio Excel / CSV", type=["xlsx","xls","csv"])
    if upload is not None and st.button("Load Uploaded Portfolio", type="primary"):
        try:
            if upload.name.lower().endswith(".csv"):
                df=pd.read_csv(upload)
            else:
                df=pd.read_excel(upload)
            cols={c.lower().strip():c for c in df.columns}
            def find_col(options):
                for o in options:
                    if o in cols: return cols[o]
                return None
            tc=find_col(["ticker","symbol","stock","nse symbol"]); qc=find_col(["quantity","qty","shares"]); pc=find_col(["average buy price","avg buy price","buy price","average price","avg price"])
            if not all([tc,qc,pc]): raise ValueError("File must contain Ticker/Symbol, Quantity, and Average Buy Price columns.")
            out=pd.DataFrame({"Ticker":df[tc].map(display_ticker),"Quantity":pd.to_numeric(df[qc],errors="coerce"),"Average Buy Price":pd.to_numeric(df[pc],errors="coerce")}).dropna()
            st.session_state.portfolio_df=out.reset_index(drop=True)
            st.success(f"Loaded {len(out)} holdings.")
        except Exception as e: st.error(str(e))

    st.subheader("1. Holdings")
    edited=st.data_editor(st.session_state.portfolio_df, num_rows="dynamic", use_container_width=True, hide_index=True, column_config={"Quantity":st.column_config.NumberColumn(min_value=0,step=1),"Average Buy Price":st.column_config.NumberColumn(min_value=0,format="₹%.2f")})
    if st.button("Save Manual Holdings"):
        x=edited.copy(); x["Ticker"]=x["Ticker"].astype(str).str.upper().str.replace(".NS","",regex=False); x["Quantity"]=pd.to_numeric(x["Quantity"],errors="coerce"); x["Average Buy Price"]=pd.to_numeric(x["Average Buy Price"],errors="coerce"); st.session_state.portfolio_df=x.dropna().reset_index(drop=True); st.success("Portfolio saved for this app session.")
    if st.session_state.portfolio_df.empty:
        st.info("Add holdings manually or upload an Excel/CSV file.")
        st.download_button("Download Portfolio Template", pd.DataFrame(columns=["Ticker","Quantity","Average Buy Price"]).to_csv(index=False), "portfolio_template.csv", "text/csv")
        return

    rows=[]
    for _,r in st.session_state.portfolio_df.iterrows():
        ticker=str(r["Ticker"]).strip().upper(); qty=clean_num(r["Quantity"]); avg=clean_num(r["Average Buy Price"])
        if not ticker or pd.isna(qty) or pd.isna(avg) or qty<=0 or avg<0: continue
        with st.spinner(f"Updating {ticker}..."):
            a=analyse_symbol(ticker,need_history=False)
        if a is None: rows.append({"Ticker":ticker,"Status":"No market data"}); continue
        invested=qty*avg; value=qty*a['price']; pnl=value-invested
        rows.append({"Ticker":ticker,"Quantity":qty,"Avg Buy":avg,"Price":a['price'],"Invested":invested,"Current Value":value,"P&L":pnl,"P&L %":pnl/invested*100 if invested else np.nan,"Allocation %":np.nan,"Score":a['score'],"5Y Model":a['multiple'],"Entry": "YES" if a['entry_trigger'] else "NO","Exit": "YES" if a['exit_trigger'] else "NO","Decision":a['verdict'],"ROE":a['roe_pct'],"Debt/Equity":a['de']})
    out=pd.DataFrame(rows)
    valid=out[out["Current Value"].notna()].copy() if "Current Value" in out else pd.DataFrame()
    if not valid.empty:
        total_value=valid["Current Value"].sum(); total_invested=valid["Invested"].sum(); valid["Allocation %"]=valid["Current Value"]/total_value*100
        c=st.columns(5); c[0].metric("Invested",f"₹{total_invested:,.0f}"); c[1].metric("Current Value",f"₹{total_value:,.0f}"); c[2].metric("Total P&L",f"₹{total_value-total_invested:,.0f}"); c[3].metric("Return",f"{(total_value/total_invested-1)*100:.1f}%" if total_invested else "-"); c[4].metric("Holdings",len(valid))
        st.dataframe(valid,use_container_width=True,hide_index=True,column_config={"P&L %":st.column_config.NumberColumn(format="%.1f%%"),"Allocation %":st.column_config.NumberColumn(format="%.1f%%"),"5Y Model":st.column_config.NumberColumn(format="%.1f×"),"ROE":st.column_config.NumberColumn(format="%.1f%%")})
        st.subheader("Portfolio Signals")
        s1,s2,s3=st.columns(3); s1.metric("Exit Trigger Holdings",int((valid["Exit"]=="YES").sum())); s2.metric("Entry Trigger Holdings",int((valid["Entry"]=="YES").sum())); s3.metric("Top Holding",f"{valid.loc[valid['Allocation %'].idxmax(),'Ticker']} ({valid['Allocation %'].max():.1f}%)")
        st.bar_chart(valid.set_index("Ticker")["Allocation %"])
        st.download_button("Export Portfolio Analysis",valid.to_csv(index=False),"portfolio_analysis.csv","text/csv")
    else: st.warning("No valid holdings could be priced.")


def load_nse_universe():
    # Official NSE security master. If NSE blocks the request, user can upload the security list.
    urls=["https://archives.nseindia.com/content/equities/EQUITY_L.csv","https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"]
    for url in urls:
        try:
            import requests
            r=requests.get(url,headers={"User-Agent":"Mozilla/5.0"},timeout=20)
            r.raise_for_status(); df=pd.read_csv(io.BytesIO(r.content));
            if "SYMBOL" in df.columns:
                return df[[c for c in ["SYMBOL","NAME OF COMPANY","ISIN NUMBER"] if c in df.columns]].drop_duplicates("SYMBOL")
        except Exception:
            pass
    return pd.DataFrame()


@st.cache_data(ttl=86400, show_spinner=False)
def get_nse_universe():
    return load_nse_universe()


def render_ath_page():
    st.title("🚀 ATH Breakouts — Last 6 Months")
    st.caption("Stocks whose corporate-action-adjusted daily price reached a new all-time high at least once during the last six months.")
    st.info("ATH is calculated from the full available adjusted daily history, not from the 52-week-high field. yfinance auto-adjusts historical OHLC for corporate actions; NSE also documents corporate-action adjustments for bonus, splits, rights and related events. citeturn0search0turn0search1")

    universe=get_nse_universe()
    uploaded=st.file_uploader("Optional: upload NSE security master CSV to replace the universe",type=["csv"],key="ath_universe")
    if uploaded is not None:
        try:
            u=pd.read_csv(uploaded); symcol=next((c for c in u.columns if c.upper().strip() in ["SYMBOL","TICKER","NSE SYMBOL"]),None)
            if symcol: universe=pd.DataFrame({"SYMBOL":u[symcol].astype(str).str.upper().str.replace(".NS","",regex=False)})
        except Exception as e: st.error(f"Universe upload failed: {e}")
    if universe.empty:
        st.warning("NSE security master could not be downloaded in this runtime. Upload the NSE security master CSV to scan the full universe.")
        return

    universe=universe[universe["SYMBOL"].notna()].copy(); universe["SYMBOL"]=universe["SYMBOL"].astype(str).str.upper().str.strip()
    exclude={"NIFTY","NIFTYBEES"}; universe=universe[~universe["SYMBOL"].isin(exclude)]
    st.write(f"Universe available: **{len(universe):,} NSE symbols**")
    col1,col2,col3=st.columns(3)
    window_months=col1.selectbox("ATH lookback",[1,3,6],index=2)
    max_symbols=col2.number_input("Max symbols per scan",100,3000,min(1000,len(universe)),100)
    min_score=col3.slider("Minimum Multibagger Lab score",0,100,0,5)
    st.caption("For a full-market scan, use 3,000 or the full available universe. Large scans can take time because historical price data is fetched per security.")

    if not st.button("Scan ATH Universe",type="primary"):
        return
    end=pd.Timestamp.today().normalize(); start=end-pd.DateOffset(months=window_months)
    syms=universe["SYMBOL"].head(int(max_symbols)).tolist()
    results=[]; progress=st.progress(0); status=st.empty()
    for i,ticker in enumerate(syms):
        status.write(f"Scanning {ticker} ({i+1}/{len(syms)})")
        try:
            px=load_price(normalize_symbol(ticker),"max"); close=close_series(px)
            if len(close)<200: continue
            hist=close[close.index<=end]
            running_max=hist.cummax(); is_ath=hist.eq(running_max)
            recent=is_ath[(is_ath.index>=start)&(is_ath.index<=end)]
            if recent.empty: continue
            ath_date=recent.index[-1]; ath_price=float(hist.loc[ath_date]); current=float(hist.iloc[-1]); dist=(current/ath_price-1)*100
            r1=current/float(hist.loc[hist.index>=end-pd.DateOffset(months=1)].iloc[0])-1 if (hist.index>=end-pd.DateOffset(months=1)).any() else np.nan
            r3=current/float(hist.loc[hist.index>=end-pd.DateOffset(months=3)].iloc[0])-1 if (hist.index>=end-pd.DateOffset(months=3)).any() else np.nan
            r6=current/float(hist.loc[hist.index>=end-pd.DateOffset(months=6)].iloc[0])-1 if (hist.index>=end-pd.DateOffset(months=6)).any() else np.nan
            # Avoid loading fundamentals for every stock until a candidate is found; fetch only after ATH test.
            a=analyse_symbol(ticker,need_history=False)
            if a is not None and a["score"]<min_score: continue
            results.append({"Ticker":ticker,"Company":a["name"] if a else ticker,"ATH Date":ath_date.date(),"ATH Price":ath_price,"Current":current,"From ATH %":dist,"1M %":r1*100,"3M %":r3*100,"6M %":r6*100,"Score":a["score"] if a else np.nan,"Entry": "YES" if a and a["entry_trigger"] else "NO","5Y Model":a["multiple"] if a else np.nan})
        except Exception:
            pass
        progress.progress((i+1)/len(syms))
    status.empty(); progress.empty()
    if not results:
        st.warning("No ATH candidates found in the selected window/universe.")
        return
    out=pd.DataFrame(results).sort_values(["ATH Date","Score"],ascending=[False,False])
    st.success(f"Found **{len(out)}** stocks that made a new all-time high in the last {window_months} month(s).")
    st.dataframe(out,use_container_width=True,hide_index=True,column_config={"From ATH %":st.column_config.NumberColumn(format="%.1f%%"),"1M %":st.column_config.NumberColumn(format="%.1f%%"),"3M %":st.column_config.NumberColumn(format="%.1f%%"),"6M %":st.column_config.NumberColumn(format="%.1f%%"),"5Y Model":st.column_config.NumberColumn(format="%.1f×")})
    st.download_button("Export ATH Scan",out.to_csv(index=False),"ath_last_6_months.csv","text/csv")
    st.subheader("ATH → What Happened Next")
    st.caption("For each ATH candidate, the next version of the engine can store the ATH event and calculate +5 trading days, +1M, +3M, +6M and +12M forward returns. This is deliberately kept separate from today's live ATH list so the event date is not lost.")


def main():
    st.sidebar.title("Multibagger Lab")
    page=st.sidebar.radio("Page",["Stock Analysis","Portfolio","ATH — Last 6 Months"])
    if page=="Stock Analysis": render_stock_page()
    elif page=="Portfolio": render_portfolio_page()
    else: render_ath_page()
    st.sidebar.divider(); st.sidebar.caption("Data status: Yahoo Finance/yfinance EOD prototype. For production PMS, replace with licensed NSE/market-data feeds and retain source + as-of timestamps for every field.")

main()
