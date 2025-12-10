import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. SETUP PATHS BEFORE IMPORTS
# We must add the parent directory to sys.path so Python can find the 'src' folder
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 2. NOW IMPORT ENGINE
from src.backtester import BacktesterEngine

# ==============================================================================
# SCIENTIFIC BENCHMARK CONFIGURATION
# ==============================================================================

STRATEGY_BENCHMARKS = {
    'HIGH_VOLATILITY': {
        'tp': 0.20, 'sl': -0.08, 'trail_trig': 0.10, 'trail_dist': 0.04   
    },
    'STANDARD_GROWTH': {
        'tp': 0.12, 'sl': -0.05, 'trail_trig': 0.06, 'trail_dist': 0.02
    },
    'LOW_VOLATILITY': {
        'tp': 0.05, 'sl': -0.025, 'trail_trig': 0.03, 'trail_dist': 0.01
    }
}

def assign_benchmark_profile(ticker):
    high_beta = [
        "NVDA", "AMD", "TSLA", "MSTR", "COIN", "HOOD", 
        "PLTR", "SMCI", "MARA", "RIOT", "DKNG", "SQ", "ROKU", "NET"
    ]
    low_beta = [
        "KO", "PEP", "JNJ", "MCD", "WMT", "PG", "VZ", "T",
        "XOM", "CVX", "JPM", "BAC", "BRK-B", "UNH"
    ]
    
    if ticker in high_beta:
        return STRATEGY_BENCHMARKS['HIGH_VOLATILITY'], "High Vol"
    elif ticker in low_beta:
        return STRATEGY_BENCHMARKS['LOW_VOLATILITY'], "Low Vol"
    else:
        return STRATEGY_BENCHMARKS['STANDARD_GROWTH'], "Standard"

if __name__ == "__main__":
    
    # 1. THE "REPORT CARD" LIST
    test_targets = [
        "NVDA", "AMD", "TSLA", "MSTR", "COIN", "HOOD",  
        "GOOGL", "MSFT", "AAPL", "AMZN",                
        "JPM", "KO", "XOM"                              
    ]
    
    # 2. THE "BRAIN" UNIVERSE
    universe_db = list(set([
        "NVDA", "AMD", "TSLA", "MSTR", "COIN", "HOOD", "PLTR", "SMCI", 
        "MARA", "RIOT", "DKNG", "SQ", "ROKU", "NET", "AFRM", "UPST", 
        "CVNA", "GME", "AMC", "RIVN", "LCID", "SOFI", "AI", "U", "PATH",
        "GOOGL", "MSFT", "AAPL", "AMZN", "META", "NFLX", "ADBE", "CRM", 
        "AVGO", "QCOM", "INTC", "TXN", "ORCL", "IBM", "CSCO", "DIS", 
        "NKE", "SBUX", "TGT", "HD", "LOW",
        "JPM", "KO", "XOM", "PEP", "JNJ", "MCD", "WMT", "PG", "VZ", "T",
        "CVX", "BAC", "UNH", "LLY", "MRK", "COST", "CL", "MO", "PM", 
        "SPY", "QQQ", "IWM", "TLT", "GLD", "SLV", "ARKK"
    ]))
    
    print(f"\n Initializing Model Benchmark with {len(universe_db)} tickers...")
    engine = BacktesterEngine(universe_db, window_size=30)
    engine.initialize(progress_callback=lambda x: None)
    
    SLIPPAGE_BPS = 0 
    MIN_VOL_FILTER = 0.00 
    CONFIDENCE_REQ = 0.80 
    
    print(f"\n MODEL PERFORMANCE REPORT (Last 250 Days)")
    print(f"   Config: 0bps Slippage | Long/Short Enabled | {CONFIDENCE_REQ*100}% Confidence Req")
    print("=" * 115)
    print(f"{'Ticker':<6} | {'Profile':<10} | {'Win Rate':<8} | {'Avg Win':<8} | {'Avg Loss':<8} | {'Trades':<6} | {'Final Eq'} | {'Expectancy'}")
    print("-" * 115)
    
    summary_stats = []
    
    for ticker in test_targets:
        params, profile_name = assign_benchmark_profile(ticker)
        
        # Execute Engine (Updated arguments)
        df_res, equity = engine.run_strategy(
            target_ticker=ticker, 
            params=params, 
            projection_days=5, 
            backtest_days=250,
            top_k=5,
            min_volatility=MIN_VOL_FILTER,
            min_confidence=CONFIDENCE_REQ
        )
        
        win_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        num_trades = 0
        final_eq = 100.0
        expectancy = 0.0
        
        if df_res is not None and not df_res.empty:
            wins = df_res[df_res['actual_return'] > 0]
            losses = df_res[df_res['actual_return'] <= 0]
            num_trades = len(df_res)
            
            if num_trades > 0:
                win_rate = len(wins) / num_trades
                avg_win = wins['actual_return'].mean() if len(wins) > 0 else 0
                avg_loss = losses['actual_return'].mean() if len(losses) > 0 else 0
                
                loss_rate = 1.0 - win_rate
                expectancy = (win_rate * avg_win) - (loss_rate * abs(avg_loss))
                
            final_eq = equity[-1]
            exp_str = f"{expectancy:.4f}"
            
            print(f"{ticker:<6} | {profile_name:<10} | {win_rate:.1%}    | {avg_win:6.2%}   | {avg_loss:6.2%}   | {num_trades:<6} | ${final_eq:.2f}    | {exp_str}")
            summary_stats.append({'Ticker': ticker, 'Final Equity': final_eq, 'Type': profile_name})
            
        else:
             print(f"{ticker:<6} | {profile_name:<10} | N/A      | N/A      | N/A      | 0      | $100.00    | 0.0000")

    print("=" * 115)
    
    if summary_stats:
        df_stats = pd.DataFrame(summary_stats)
        print("\n SECTOR PERFORMANCE SUMMARY:")
        print(df_stats.groupby('Type')['Final Equity'].mean().apply(lambda x: f"${x:.2f}"))
        
        try:
            tickers = [s['Ticker'] for s in summary_stats]
            values = [s['Final Equity'] - 100 for s in summary_stats]
            
            plt.figure(figsize=(14, 6))
            colors = []
            for t in tickers:
                _, prof = assign_benchmark_profile(t)
                if prof == "High Vol": colors.append('#e74c3c')
                elif prof == "Low Vol": colors.append('#3498db')
                else: colors.append('#f1c40f')
                
            bars = plt.bar(tickers, values, color=colors)
            plt.title(f"Model Performance (Long/Short, {CONFIDENCE_REQ*100}% Confidence)")
            plt.ylabel("Net Profit (%)")
            plt.axhline(0, color='black', linewidth=0.8)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.show()
        except: pass

