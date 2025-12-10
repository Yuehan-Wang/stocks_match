import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.data_loader import fetch_stock_data
from src.processor import create_multifeature_windows
from src.indexer import VectorIndex
from src.macro_loader import MacroLoader
from src.backtester import BacktesterEngine, StrategyConfig
import yfinance as yf

# --- Page Configuration ---
st.set_page_config(page_title="Project Mirror", layout="wide", page_icon="🔮")

# --- Session State Init ---
if 'universe' not in st.session_state:
    # UPDATED: Expanded Universe from Phase 1 Report (65 Tickers)
    st.session_state.universe = list(set([
        # High Vol / Speculative
        "NVDA", "AMD", "TSLA", "MSTR", "COIN", "HOOD", "PLTR", "SMCI", 
        "MARA", "RIOT", "DKNG", "SQ", "ROKU", "NET", "AFRM", "UPST", 
        "CVNA", "GME", "AMC", "RIVN", "LCID", "SOFI", "AI", "U", "PATH",
        # Standard / Mega Cap
        "GOOGL", "MSFT", "AAPL", "AMZN", "META", "NFLX", "ADBE", "CRM", 
        "AVGO", "QCOM", "INTC", "TXN", "ORCL", "IBM", "CSCO", "DIS", 
        "NKE", "SBUX", "TGT", "HD", "LOW",
        # Low Vol / Defensive
        "JPM", "KO", "XOM", "PEP", "JNJ", "MCD", "WMT", "PG", "VZ", "T",
        "CVX", "BAC", "UNH", "LLY", "MRK", "COST", "CL", "MO", "PM", 
        # ETFs
        "SPY", "QQQ", "IWM", "TLT", "GLD", "SLV", "ARKK"
    ]))

# --- Navigation ---
page = st.sidebar.radio("Navigation", ["Live Analysis", "Backtest Lab"])

# ==========================================
# PAGE 1: LIVE ANALYSIS
# ==========================================
if page == "Live Analysis":
    st.title("Live Pattern Matcher")
    
    st.sidebar.header("Live Settings")
    target_ticker = st.sidebar.text_input("Target Ticker", value="NVDA").upper()
    window_size = st.sidebar.slider("Window Size", 10, 60, 30)
    future_days = st.sidebar.slider("Projection Days", 3, 10, 5)
    
    st.sidebar.subheader("Filters")
    use_macro = st.sidebar.checkbox("Macro (Yield) Filter", value=True)
    yield_tol = st.sidebar.slider("Yield Tolerance (+/- %)", 0.5, 5.0, 2.0)
    vol_tol = st.sidebar.slider("Volatility Ratio Max", 1.0, 5.0, 3.0)

    @st.cache_resource
    def get_macro_loader():
        return MacroLoader()

    @st.cache_resource
    def build_live_index(tickers, w_size):
        db = VectorIndex(dimension=w_size * 12)
        store = {}
        try: spy = fetch_stock_data("SPY")
        except: spy = None
        
        status = st.empty()
        prog = st.progress(0)
        
        total = len(tickers)
        for i, t in enumerate(tickers):
            try:
                df = fetch_stock_data(t)
                if len(df) > w_size + 50:
                    store[t] = df['Close'].values
                    feats, dates, idxs = create_multifeature_windows(df, w_size, benchmark_df=spy)
                    db.add_data(feats, dates, t, idxs)
            except: pass
            
            if i % 5 == 0:
                status.text(f"Indexing Market Data... ({i}/{total})")
            prog.progress((i+1)/total)
        
        status.empty()
        prog.empty()
        return db, store, spy

    if st.button("Analyze Live Pattern"):
        macro = get_macro_loader()
        
        with st.spinner("Building AI Brain..."):
            db, store, spy_df = build_live_index(st.session_state.universe, window_size)
        
        try:
            target_df = fetch_stock_data(target_ticker, period="2y")
            feats, _, _ = create_multifeature_windows(target_df, window_size, benchmark_df=spy_df)
            curr_vec = feats[-1]
            curr_price_pat = target_df['Close'].values[-window_size:]
            try: curr_price = yf.Ticker(target_ticker).fast_info['last_price']
            except: curr_price = target_df['Close'].iloc[-1]
        except Exception as e:
            st.error(f"Error fetching {target_ticker}: {e}")
            st.stop()

        curr_macro = macro.get_latest_context()
        st.info(f"Current 10Y Yield: {curr_macro['yield']:.2f}% | VIX: {curr_macro['vix']:.2f}")

        # Search
        raw_matches = db.search(curr_vec, k=300)
        valid_matches = []
        MAX_SAMPLES = 30 
        
        curr_vol = np.std(curr_price_pat)
        if curr_vol == 0: curr_vol = 0.001
        
        for m in raw_matches:
            if m['distance'] < 1e-5: continue
            
            # Macro Filter
            if use_macro:
                hist_macro = macro.get_context_at_date(m['date'])
                if not hist_macro or abs(curr_macro['yield'] - hist_macro['yield']) > yield_tol:
                    continue
                m['yield'] = hist_macro['yield']
            else: m['yield'] = 0
            
            full_series = store.get(m['ticker'])
            if full_series is None: continue
            
            start = m['start_index']
            hist_shape = full_series[start : start+window_size]
            if len(hist_shape) < window_size: continue
            
            # Correlation Filter
            corr = np.corrcoef(curr_price_pat, hist_shape)[0,1]
            if corr < 0.85: continue
            
            # Volatility Filter
            hist_vol = np.std(hist_shape)
            vol_ratio = hist_vol / curr_vol
            if vol_ratio > vol_tol or vol_ratio < (1.0 / vol_tol): continue
            
            end = start + window_size + future_days
            if end > len(full_series): continue
            
            full_seg = full_series[start : end]
            m['full_window'] = full_seg
            m['corr'] = corr
            m['vol_ratio'] = vol_ratio
            
            ret = (full_seg[-1] - full_seg[window_size-1]) / full_seg[window_size-1]
            m['return'] = ret
            valid_matches.append(m)
            if len(valid_matches) >= MAX_SAMPLES: break
            
        if not valid_matches:
            st.warning("No matches found. Try relaxing the Volatility or Yield filters.")
        else:
            valid_matches.sort(key=lambda x: x['corr'], reverse=True)
            corrs = [m['corr'] for m in valid_matches]
            returns = [m['return'] for m in valid_matches]
            weights = np.array(corrs) ** 4
            w_avg_return = np.average(returns, weights=weights)
            
            winners_weight = sum(w for r, w in zip(returns, weights) if r > 0)
            total_weight = sum(weights)
            w_win_rate = winners_weight / total_weight if total_weight > 0 else 0
            
            if w_avg_return >= 0:
                direction_label = "Bullish"
                confidence = w_win_rate
                ret_color = "normal"
            else:
                direction_label = "Bearish"
                confidence = 1.0 - w_win_rate
                ret_color = "inverse"
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Trend Prediction", direction_label)
            c2.metric("Model Confidence", f"{confidence:.1%}") 
            c3.metric(f"{future_days}-Day Avg Return", f"{w_avg_return:.2%}", delta_color=ret_color)
            
            # Plot
            fig = go.Figure()
            for m in valid_matches[:5]:
                hist = m['full_window']
                norm_hist = (hist / hist[window_size-1] - 1) * 100
                color = "green" if m['return'] > 0 else "red"
                label = f"{m['ticker']} (Corr:{m['corr']:.2f})"
                fig.add_trace(go.Scatter(y=norm_hist, mode='lines', name=label, line=dict(color=color, width=1, dash='dot')))
            
            curr_norm = (curr_price_pat / curr_price_pat[-1] - 1) * 100
            x_curr = list(range(window_size))
            fig.add_trace(go.Scatter(y=curr_norm, x=x_curr, mode='lines', name="Current", line=dict(color='black', width=3)))
            
            fig.add_vline(x=window_size-1, line_dash="dash")
            fig.update_layout(title="Pattern Projection (Normalized % Change)", xaxis_title="Days", yaxis_title="Change %", height=500)
            st.plotly_chart(fig, width='stretch')
            
            st.subheader(f"Top {len(valid_matches)} Matches Detail")
            match_data = []
            for m in valid_matches:
                match_data.append({
                    "Ticker": m['ticker'],
                    "Date": m['date'].strftime('%Y-%m-%d'),
                    "Correlation": f"{m['corr']:.4f}",
                    "Vol Ratio": f"{m['vol_ratio']:.2f}x",
                    "Hist Yield": f"{m.get('yield', 0):.2f}%",
                    "5d Return": f"{m['return']:.2%}"
                })
            st.dataframe(pd.DataFrame(match_data), width='stretch')

# ==========================================
# PAGE 2: BACKTEST LAB
# ==========================================
elif page == "Backtest Lab":
    st.title("Strategy Backtest Lab")
    
    with st.expander(" Global Configuration (Universe & Index)", expanded=True):
        c1, c2 = st.columns([3, 1])
        with c1:
            st.info("Universe: The library of historical patterns the AI learns from.")
            universe_selected = st.multiselect(
                "Active Universe List",
                options=st.session_state.universe,
                default=st.session_state.universe
            )
        with c2:
            window_size = st.number_input("Pattern Window", value=30)
            proj_days = st.selectbox("Forecast Horizon", [1, 3, 5, 10], index=2)
            # NEW: Velocity Filter Control
            min_vol_req = st.slider(
                "Min Daily Volatility % (Velocity Filter)", 
                0.0, 5.0, 1.5, 0.1, 
                help="Ignore stocks that move less than this % per day. (e.g. 1.5% for high-beta only)"
            )

    @st.cache_resource
    def get_engine(univ, w_size):
        eng = BacktesterEngine(univ, w_size)
        return eng

    engine = get_engine(universe_selected, window_size)
    
    if st.button("♻️ Rebuild/Update Database"):
        engine.is_initialized = False 
        placeholder = st.empty()
        def update_prog(p):
            placeholder.progress(p, text="Indexing Universe...")
        engine.initialize(progress_callback=update_prog)
        placeholder.success("Database Ready!")
    
    st.divider()
    st.subheader("🎯 Test Targets & Strategy")
    st.markdown("Enter the stocks you want to BACKTEST.")
    
    col_sel, col_add = st.columns([2, 1])
    with col_sel:
        preset_targets = st.multiselect("Select from Universe List", options=st.session_state.universe, default=["NVDA", "TSLA"])
    with col_add:
        custom_input = st.text_input("Or Type ANY Ticker", placeholder="e.g. GME, PLTR")
    
    custom_targets = [t.strip().upper() for t in custom_input.split(',') if t.strip()]
    final_targets = list(set(preset_targets + custom_targets))
    
    strategies = {}
    
    if final_targets:
        cols = st.columns(3)
        for i, ticker in enumerate(final_targets):
            col = cols[i % 3]
            defaults = StrategyConfig.get_default_params(ticker)
            
            with col:
                with st.expander(f" {ticker} Strategy", expanded=False):
                    tp = st.number_input(f"TP ({ticker})", 0.01, 1.00, defaults['tp'], 0.01, key=f"tp_{ticker}")
                    sl = st.number_input(f"SL ({ticker})", -0.50, -0.01, defaults['sl'], 0.01, key=f"sl_{ticker}")
                    tr_trig = st.number_input(f"Trail Trigger ({ticker})", 0.01, 0.50, defaults['trail_trig'], key=f"tt_{ticker}")
                    tr_dist = st.number_input(f"Trail Dist ({ticker})", 0.01, 0.20, defaults['trail_dist'], key=f"td_{ticker}")
                    
                    strategies[ticker] = {
                        'tp': tp, 'sl': sl, 'trail_trig': tr_trig, 'trail_dist': tr_dist
                    }

    if st.button("Run Backtest Simulation", type="primary"):
        if not engine.is_initialized:
            st.warning("Please click 'Rebuild/Update Database' first!")
        else:
            results_data = []
            equity_curves = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, ticker in enumerate(final_targets):
                status_text.text(f"Testing {ticker}...")
                strat = strategies[ticker]
                
                # UPDATED CALL: Passing min_vol_req from Slider
                df_res, equity = engine.run_strategy(
                    ticker, strat, 
                    projection_days=proj_days, 
                    backtest_days=250,
                    min_volatility=min_vol_req / 100.0 # Convert 1.5 to 0.015
                )
                
                if df_res is not None and not df_res.empty:
                    final_eq = equity[-1]
                    trades = df_res[df_res['position']=='LONG']
                    win_rate = len(trades[trades['actual_return']>0]) / len(trades) if len(trades)>0 else 0
                    
                    results_data.append({
                        "Ticker": ticker,
                        "Win Rate": f"{win_rate:.1%}",
                        "Final Equity": f"${final_eq:.2f}",
                        "Trades": len(trades),
                        "Strategy": f"TP:{strat['tp']} SL:{strat['sl']}"
                    })
                    
                    equity_curves[ticker] = [(x - 100) for x in equity]
                else:
                    st.toast(f"Skipping {ticker}: No trades found (Check Volatility Filter).", icon="⚠️")
                
                progress_bar.progress((i+1)/len(final_targets))
            
            status_text.empty()
            progress_bar.empty()
            
            st.divider()
            
            st.subheader("Performance Report")
            if results_data:
                st.dataframe(pd.DataFrame(results_data), width='stretch')
            else:
                st.warning("No trades generated. Try lowering the Min Volatility Filter or choosing wilder stocks.")
            
            st.subheader("Cumulative Profit (%)")
            if equity_curves:
                fig = go.Figure()
                for ticker, curve in equity_curves.items():
                    fig.add_trace(go.Scatter(y=curve, mode='lines', name=ticker))
                fig.update_layout(xaxis_title="Trading Days", yaxis_title="Net Profit %", height=500, hovermode="x unified")
                st.plotly_chart(fig, width='stretch')