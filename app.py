import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(page_title="Institutional Stock Analyzer & Backtest", layout="wide")

st.title("📊 Institutional Stock Scoring, Backtest & 10x Math Engine")
st.caption("Live Yahoo Finance API | 6-Point Analysis Engine & Historical Backtest Matrix")

# Sidebar Controls
st.sidebar.header("Search & Parameters")
ticker_input = st.sidebar.text_input("Enter NSE Ticker Symbol:", value="POLYCAB").strip().upper()
assumed_cagr = st.sidebar.slider("Assumed 5Y Earnings CAGR (%)", min_value=10.0, max_value=40.0, value=25.0, step=1.0)

symbol = f"{ticker_input}.NS" if not (ticker_input.endswith(".NS") or ticker_input.endswith(".BO")) else ticker_input

if st.sidebar.button("Fetch Complete Analysis", type="primary"):
    with st.spinner(f"Fetching complete 6-point analysis for {ticker_input}..."):
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            hist = stock.history(period="5y")

            # Fundamental Data Extraction
            price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
            eps_trailing = info.get('trailingEps', 0.0)
            pe_ratio = info.get('trailingPE', 0.0)
            roe = (info.get('returnOnEquity', 0.0) or 0.0) * 100
            roce = (info.get('returnOnAssets', 0.0) or 0.0) * 100 * 1.5  # Approximate ROCE proxy
            debt_to_equity = info.get('debtToEquity', 0.0)
            peg_ratio = info.get('pegRatio', 0.0)
            company_name = info.get('longName', ticker_input)
            market_cap = info.get('marketCap', 0) / 1e7  # Convert to Crores

            # --- MODULE 1: HARD GATEKEEPER FILTERS ---
            pass_debt = debt_to_equity < 100.0 if debt_to_equity is not None else True
            pass_roe = roe > 12.0
            pass_pe = pe_ratio > 0 and pe_ratio < 75.0
            gatekeepers_passed = all([pass_debt, pass_roe, pass_pe])

            # --- MODULE 2: DETAILED 100-POINT FACTOR SCORING ---
            score_earnings = 22 if assumed_cagr >= 25 else (15 if assumed_cagr >= 15 else 8)
            score_quality = 18 if roe > 20 and debt_to_equity < 30 else (12 if roe > 12 else 6)
            score_opportunity = 14 if market_cap < 50000 else (10 if market_cap < 200000 else 6)
            score_valuation = 13 if peg_ratio > 0 and peg_ratio <= 1.5 else (8 if pe_ratio < 40 else 4)
            score_management = 9 if pass_debt and roe > 15 else 6
            score_technical = 8 if not hist.empty and price > hist['Close'].tail(50).mean() else 4
            score_catalysts = 4

            total_score = (score_earnings + score_quality + score_opportunity + 
                           score_valuation + score_management + score_technical + score_catalysts)

            # --- MODULE 3: 10X MATH COMPOUNDING MODEL ---
            exit_pe = 35.0 if pe_ratio <= 0 else min(pe_ratio, 40.0)
            future_eps_5y = eps_trailing * ((1 + (assumed_cagr / 100)) ** 5) if eps_trailing > 0 else 0
            target_price_5y = future_eps_5y * exit_pe
            potential_multiple = (target_price_5y / price) if price and price > 0 else 0

            # --- MODULE 4: HISTORICAL 5-YEAR BACKTEST METRICS ---
            if not hist.empty:
                price_5y_ago = hist['Close'].iloc[0]
                cagr_5y_actual = (((price / price_5y_ago) ** (1 / 5)) - 1) * 100
                total_return_5y = ((price - price_5y_ago) / price_5y_ago) * 100
                max_drawdown = (((hist['Close'] - hist['Close'].cummax()) / hist['Close'].cummax()).min()) * 100
            else:
                cagr_5y_actual, total_return_5y, max_drawdown = 0, 0, 0

            # --- MODULE 5: STRATEGY VERDICT & ALLOCATION ---
            if total_score >= 80 and gatekeepers_passed:
                verdict = "BUY ZONE 🟢"
                portfolio_role = "Top Conviction Multibagger (Tier 1)"
            elif total_score >= 65:
                verdict = "ACCUMULATE 🟡"
                portfolio_role = "Core Quality Compounder (Tier 2)"
            else:
                verdict = "AVOID / WAIT 🔴"
                portfolio_role = "High Risk / Low Growth Probability"

            # ================= DISPLAY UI =================
            st.header(f"{company_name} ({ticker_input})")
            st.write(f"**Market Cap:** ₹{market_cap:,.0f} Cr | **Portfolio Role:** {portfolio_role}")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Live Market Price", f"₹{price:,.2f}" if price else "N/A")
            m2.metric("Strategy Score", f"{total_score} / 100")
            m3.metric("5Y Target Multiple", f"{potential_multiple:.1f}x")
            m4.metric("Strategy Verdict", verdict)

            st.divider()

            # Row 1: Hard Gatekeepers & 100-Point Scoring
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("1. Hard Gatekeeper Verification")
                st.write(f"• **Debt-to-Equity (<100%):** {'PASS ✅' if pass_debt else 'FAIL ❌'} ({debt_to_equity:.1f}%)")
                st.write(f"• **Return on Equity (>12%):** {'PASS ✅' if pass_roe else 'FAIL ❌'} ({roe:.1f}%)")
                st.write(f"• **Valuation Safety (P/E < 75):** {'PASS ✅' if pass_pe else 'FAIL ❌'} ({pe_ratio:.1f}x)")
                if gatekeepers_passed:
                    st.success("Passes all mandatory risk filters.")
                else:
                    st.error("Failed gatekeeper filters—exercise high risk caution.")

            with col_b:
                st.subheader("2. 100-Point Factor Score Breakdown")
                score_df = pd.DataFrame({
                    "Factor": ["Earnings Growth", "Business Quality", "TAM Opportunity", "Valuation", "Management", "Technicals", "Catalysts"],
                    "Score": [score_earnings, score_quality, score_opportunity, score_valuation, score_management, score_technical, score_catalysts],
                    "Max": [25, 20, 15, 15, 10, 10, 5]
                })
                st.dataframe(score_df, use_container_width=True, hide_index=True)

            st.divider()

            # Row 2: 10x Math Model & 5-Year Backtest
            col_c, col_d = st.columns(2)
            with col_c:
                st.subheader("3. Crucial 10x Math Model (5Y Projections)")
                st.write(f"• **Trailing EPS:** ₹{eps_trailing:.2f}")
                st.write(f"• **Modeled Growth:** {assumed_cagr}% CAGR")
                st.write(f"• **Projected 5Y EPS:** ₹{future_eps_5y:.2f}")
                st.write(f"• **Modeled Exit P/E:** {exit_pe:.1f}x")
                st.write(f"• **5Y Target Price:** ₹{target_price_5y:,.2f}")
                st.write(f"• **Strategy Fit:** {'QUALIFIED FOR 10X ROADMAP 🟢' if potential_multiple >= 3.0 else 'DEFENSIVE COMPOUNDER ONLY 🟡'}")

            with col_d:
                st.subheader("4. Historical 5-Year Backtest Performance")
                st.write(f"• **Historical 5Y Return:** {total_return_5y:+.1f}%")
                st.write(f"• **Historical Annualized CAGR:** {cagr_5y_actual:+.1f}% p.a.")
                st.write(f"• **Maximum Drawdown (5Y):** {max_drawdown:.1f}%")
                if cagr_5y_actual > 20:
                    st.info("Historical backtest verifies high compounding capability.")
                else:
                    st.warning("Historical CAGR is under 20%—relies heavily on future acceleration.")

            st.divider()

            # Row 3: Target Milestones & Execution Plan
            st.subheader("5. Execution Target Milestones")
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Stop Loss (-15%)", f"₹{price * 0.85:,.2f}")
            t2.metric("2x Milestone", f"₹{price * 2.0:,.2f}")
            t3.metric("3x Milestone", f"₹{price * 3.0:,.2f}")
            t4.metric("5x Milestone", f"₹{price * 5.0:,.2f}")

        except Exception as e:
            st.error(f"Error executing analysis for {ticker_input}. Verify symbol on NSE.")
            st.exception(e)
