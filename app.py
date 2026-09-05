import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Page Layout Setup
st.set_page_config(page_title="Stock Analyzer & 10x Math Engine", layout="wide")

st.title("📊 Institutional Equity Scoring & 10x Math Engine")
st.caption("Live Market Feed | 100-Point Factor Matrix & Hard Gatekeeper Filters")

# Sidebar Controls
st.sidebar.header("Search Stock")
ticker_input = st.sidebar.text_input("Enter NSE Ticker Symbol:", value="POLYCAB").strip().upper()

symbol = f"{ticker_input}.NS" if not (ticker_input.endswith(".NS") or ticker_input.endswith(".BO")) else ticker_input

if st.sidebar.button("Fetch Live Analysis", type="primary"):
    with st.spinner(f"Connecting to live market feed for {ticker_input}..."):
        try:
            stock = yf.Ticker(symbol)
            info = stock.info

            price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
            eps_trailing = info.get('trailingEps', 0.0)
            pe_ratio = info.get('trailingPE', 0.0)
            roe = (info.get('returnOnEquity', 0.0) or 0.0) * 100
            debt_to_equity = info.get('debtToEquity', 0.0)
            company_name = info.get('longName', ticker_input)

            # Hard Gatekeeper Filters Evaluation
            pass_debt = debt_to_equity < 100.0 if debt_to_equity is not None else True
            pass_roe = roe > 12.0
            
            # Factor Scoring (100 Points Scale)
            earnings_score = 20 if eps_trailing > 0 else 5
            quality_score = 18 if roe > 18 else (12 if roe > 10 else 6)
            opportunity_score = 12
            valuation_score = 12 if pe_ratio > 0 and pe_ratio < 40 else 6
            management_score = 8
            technical_score = 8
            catalyst_score = 4
            
            total_score = (earnings_score + quality_score + opportunity_score + 
                           valuation_score + management_score + technical_score + catalyst_score)

            # Crucial 10x Math Projections
            assumed_growth_rate = 25.0
            exit_pe = 30.0 if pe_ratio <= 0 else min(pe_ratio, 45.0)
            
            future_eps_5y = eps_trailing * ((1 + (assumed_growth_rate / 100)) ** 5) if eps_trailing > 0 else 0
            target_price_5y = future_eps_5y * exit_pe
            potential_multiple = (target_price_5y / price) if price and price > 0 else 0

            # Verdict Logic
            if total_score >= 80 and pass_debt:
                verdict = "BUY ZONE 🟢"
            elif total_score >= 65:
                verdict = "ACCUMULATE 🟡"
            else:
                verdict = "AVOID / WAIT 🔴"

            # Display Output Metrics
            st.header(f"{company_name} ({ticker_input})")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Live Market Price", f"₹{price:,.2f}" if price else "N/A")
            col2.metric("Overall Score", f"{total_score} / 100")
            col3.metric("5-Year Multiple", f"{potential_multiple:.1f}x")
            col4.metric("Verdict", verdict)
            
            st.divider()

            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("1. Hard Gatekeeper Check")
                st.write(f"• **Debt to Equity:** {'PASS ✅' if pass_debt else 'FAIL ❌'} ({debt_to_equity:.1f}%)")
                st.write(f"• **Return on Equity (ROE):** {'PASS ✅' if pass_roe else 'WARNING ⚠️'} ({roe:.1f}%)")
                st.write(f"• **Trailing P/E Ratio:** {pe_ratio:.1f}x" if pe_ratio else "• **Trailing P/E Ratio:** N/A")

            with col_b:
                st.subheader("2. Crucial 10x Math Model")
                st.write(f"• **Trailing EPS:** ₹{eps_trailing:.2f}")
                st.write(f"• **Modeled Growth:** {assumed_growth_rate}% CAGR")
                st.write(f"• **Projected 5Y EPS:** ₹{future_eps_5y:.2f}")
                st.write(f"• **5Y Target Price:** ₹{target_price_5y:,.2f}")

            st.divider()
            st.subheader("3. Execution Price Milestones")
            t_col1, t_col2, t_col3, t_col4 = st.columns(4)
            t_col1.metric("Stop Loss (15%)", f"₹{price * 0.85:,.2f}" if price else "N/A")
            t_col2.metric("2x Target", f"₹{price * 2.0:,.2f}" if price else "N/A")
            t_col3.metric("3x Target", f"₹{price * 3.0:,.2f}" if price else "N/A")
            t_col4.metric("5x Target", f"₹{price * 5.0:,.2f}" if price else "N/A")

        except Exception as e:
            st.error(f"Could not load live data for {ticker_input}. Ensure the ticker symbol is valid on NSE.")
