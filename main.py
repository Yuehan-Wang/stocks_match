import numpy as np
import pandas as pd
from src.data_loader import fetch_stock_data
from src.processor import create_multifeature_windows
from src.indexer import VectorIndex
from src.macro_loader import MacroLoader
from src.visualizer import plot_matches
from tqdm import tqdm

# Configuration
TICKERS_DB = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA", "SPY", "QQQ"]
TARGET_TICKER = "NVDA" 
WINDOW_SIZE = 30
FUTURE_DAYS = 5
TOP_K = 5

# Threshold: How different can the interest rate be?
# 1.5 means if today is 4.5%, we accept history between 3.0% and 6.0%
MACRO_YIELD_TOLERANCE = 1.5 

GLOBAL_DATA_STORE = {}

def main():
    # 1. Initialize Engines
    macro_engine = MacroLoader()
    
    feature_dim = WINDOW_SIZE * 2 # Price + Volume
    print(f"\nInitializing Multi-Feature Database (Dim={feature_dim})...")
    db = VectorIndex(dimension=feature_dim) 
    
    # 2. Ingest Data
    print("Ingesting Price + Volume data...")
    for ticker in tqdm(TICKERS_DB):
        try:
            df = fetch_stock_data(ticker)
            if 'Volume' not in df.columns: continue
            
            # Store raw price for projection
            GLOBAL_DATA_STORE[ticker] = df['Close'].values
            
            # Create Features
            features, dates, indices = create_multifeature_windows(df, WINDOW_SIZE)
            db.add_data(features, dates, ticker, indices)
            
        except Exception as e:
            pass # Skip errors for cleaner output

    # 3. Get Target State
    print(f"\nFetching Target: {TARGET_TICKER}...")
    target_df = fetch_stock_data(TARGET_TICKER, period="1y") 
    target_features, _, _ = create_multifeature_windows(target_df, WINDOW_SIZE)
    
    current_vector = target_features[-1]
    current_date = target_df.index[-1]
    
    # --- MACRO CHECK ---
    current_macro = macro_engine.get_latest_context()
    print(f"\n🌍 Current Macro Regime:")
    print(f"   10-Year Yield: {current_macro['yield']:.2f}%")
    print(f"   VIX Level:     {current_macro['vix']:.2f}")
    print(f"   Filtering for history where Yield is within +/- {MACRO_YIELD_TOLERANCE}%...")

    # 4. Search
    # Fetch 3x more candidates because the Macro Filter will delete many of them
    matches = db.search(current_vector, k=TOP_K * 4)
    
    final_matches = []
    
    # 5. Filter Matches
    print(f"\nFiltering {len(matches)} raw matches by Macro Context...")
    
    for m in matches:
        if m['distance'] < 1e-5: continue 
        
        # Check Macro on that historical date
        hist_macro = macro_engine.get_context_at_date(m['date'])
        
        if hist_macro is None:
            continue
            
        # FILTER 1: Interest Rate Check
        yield_diff = abs(current_macro['yield'] - hist_macro['yield'])
        if yield_diff > MACRO_YIELD_TOLERANCE:
            # Skip this match, it's from a different economic era
            continue
            
        # Hydrate with future data
        ticker = m['ticker']
        start_idx = m['start_index']
        full_series = GLOBAL_DATA_STORE[ticker]
        end_idx = start_idx + WINDOW_SIZE + FUTURE_DAYS
        
        if end_idx > len(full_series): continue
            
        m['full_window'] = full_series[start_idx : end_idx]
        m['macro_yield'] = hist_macro['yield'] # Store for display
        final_matches.append(m)
        
        if len(final_matches) >= TOP_K:
            break
            
    # 6. Output
    print(f"\nTop {len(final_matches)} Macro-Aligned Matches:")
    print("-" * 65)
    print(f"{'Ticker':<8} | {'Date':<12} | {'Yield':<8} | {'Dist':<8}")
    print("-" * 65)
    
    for m in final_matches:
        print(f"{m['ticker']:<8} | {str(m['date'].date()):<12} | {m['macro_yield']:.2f}%    | {m['distance']:.4f}")
        
    # Plot (using only price part of current vector for viz)
    current_price_plot = target_df['Close'].values[-WINDOW_SIZE:]
    plot_matches(current_price_plot, final_matches, future_length=FUTURE_DAYS)

if __name__ == "__main__":
    main()