import pandas as pd
import numpy as np
from tqdm import tqdm
from src.data_loader import fetch_stock_data
from src.processor import create_multifeature_windows
from src.indexer import VectorIndex
from src.macro_loader import MacroLoader

class BacktesterEngine:
    def __init__(self, universe, window_size=30):
        self.universe = universe
        self.window_size = window_size
        self.macro_loader = MacroLoader()
        self.db = None
        self.global_store = {}
        self.benchmark_data = None
        self.is_initialized = False
        
    def initialize(self, progress_callback=None):
        if self.is_initialized: return
        self.benchmark_data = fetch_stock_data("SPY")
        
        total = len(self.universe)
        for i, ticker in enumerate(self.universe):
            try:
                df = fetch_stock_data(ticker)
                if len(df) > self.window_size + 100:
                    self.global_store[ticker] = df
            except: pass
            if progress_callback: progress_callback(i / total * 0.5)
                
        processed_data = []
        sample_feats = None
        
        print("Processing Vectors...")
        for ticker, df in self.global_store.items():
            try:
                features, dates, indices = create_multifeature_windows(
                    df, self.window_size, benchmark_df=self.benchmark_data
                )
                if len(features) > 0:
                    processed_data.append((ticker, features, dates, indices))
                    if sample_feats is None: sample_feats = features[0]
            except: continue
            
        if sample_feats is not None:
            self.db = VectorIndex(dimension=len(sample_feats))
            for p in processed_data:
                self.db.add_data(p[1], p[2], p[0], p[3])
                
        self.is_initialized = True
        if progress_callback: progress_callback(1.0)

    def run_strategy(self, target_ticker, params, projection_days=5, backtest_days=252, top_k=5, min_volatility=0.015, min_confidence=0.8):
        """
        Flexible Execution Engine (Long/Short Capable).
        Args:
            min_confidence (float): 0.0 to 1.0. Required consistency of matches to trigger a trade.
        """
        if target_ticker not in self.global_store:
            try: self.global_store[target_ticker] = fetch_stock_data(target_ticker)
            except: return None, None

        target_data = self.global_store[target_ticker]
        
        try:
            target_features, target_dates, _ = create_multifeature_windows(
                target_data, self.window_size, benchmark_df=self.benchmark_data
            )
        except: return None, None
        
        available_len = len(target_features)
        if available_len < backtest_days: backtest_days = available_len - 1
        if backtest_days < 10: return None, None

        start_idx = available_len - backtest_days
        results = []
        equity_curve = [100.0] 
        
        holding = False
        position_type = "NONE" # 'LONG' or 'SHORT'
        entry_price = 0.0
        max_price = 0.0 
        min_price = 0.0 
        days_held = 0
        
        # Volatility Calc
        high = target_data['High']
        low = target_data['Low']
        close = target_data['Close']
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr_pct_series = (tr.rolling(14).mean() / close).fillna(0)
        
        for i in range(start_idx, available_len - 1):
            current_date = target_dates[i]
            try:
                decision_idx = target_data.index.get_loc(current_date) + self.window_size - 1
            except KeyError: continue
            
            if decision_idx >= len(target_data) - 1: break
            
            decision_date = target_data.index[decision_idx]
            current_price = target_data['Close'].iloc[decision_idx]
            
            # --- EXIT LOGIC ---
            if holding:
                days_held += 1
                should_sell = False
                exit_reason = ""
                pct_change = 0.0
                
                if position_type == "LONG":
                    if current_price > max_price: max_price = current_price
                    pct_change = (current_price - entry_price) / entry_price
                    drawdown = (current_price - max_price) / max_price
                    
                    if pct_change >= params['tp']: should_sell = True; exit_reason = "TP"
                    elif pct_change > params['trail_trig'] and drawdown < -params['trail_dist']: should_sell = True; exit_reason = "Trail"
                    elif pct_change > 0.02 and current_price < entry_price * 0.995: should_sell = True; exit_reason = "BE"
                    elif pct_change <= params['sl']: should_sell = True; exit_reason = "SL"
                    elif days_held >= projection_days: should_sell = True; exit_reason = "Time"
                    
                elif position_type == "SHORT":
                    if current_price < min_price: min_price = current_price
                    # Short Return: (Entry - Current) / Entry
                    pct_change = (entry_price - current_price) / entry_price
                    # Drawback for short: (Current - Min) / Min
                    drawback = (current_price - min_price) / min_price 
                    
                    if pct_change >= params['tp']: should_sell = True; exit_reason = "TP"
                    elif pct_change > params['trail_trig'] and drawback > params['trail_dist']: should_sell = True; exit_reason = "Trail"
                    elif pct_change <= params['sl']: should_sell = True; exit_reason = "SL"
                    elif days_held >= projection_days: should_sell = True; exit_reason = "Time"

                if should_sell:
                    equity_curve.append(equity_curve[-1] * (1 + pct_change))
                    results.append({
                        'date': decision_date, 
                        'position': position_type, 
                        'actual_return': pct_change, 
                        'type': exit_reason
                    })
                    holding = False
                    position_type = "NONE"
                    continue
                
                equity_curve.append(equity_curve[-1])
                continue

            # --- ENTRY LOGIC ---
            
            current_atr_pct = atr_pct_series.iloc[decision_idx]
            if current_atr_pct < min_volatility: 
                equity_curve.append(equity_curve[-1])
                continue

            current_vector = target_features[i]
            current_price_shape = current_vector[:self.window_size]
            current_macro = self.macro_loader.get_context_at_date(decision_date)
            if current_macro is None: equity_curve.append(equity_curve[-1]); continue

            raw_price_seq = target_data['Close'].iloc[decision_idx - self.window_size + 1 : decision_idx + 1].values
            curr_vol = np.std(raw_price_seq)
            if curr_vol == 0: curr_vol = 0.001

            raw_matches = self.db.search(current_vector, k=top_k * 20)
            valid_matches = []
            
            for m in raw_matches:
                if m['distance'] < 1e-5: continue
                hist_macro = self.macro_loader.get_context_at_date(m['date'])
                if not hist_macro or abs(current_macro['yield'] - hist_macro['yield']) > 2.0: continue 
                
                ticker = m['ticker']
                df = self.global_store[ticker]
                try:
                    if m['date'] not in df.index: continue
                    start_ptr = m['start_index']
                    match_price_seq = df['Close'].values[start_ptr : start_ptr + self.window_size]
                    corr = np.corrcoef(current_price_shape, match_price_seq)[0, 1]
                    if corr < 0.85: continue 
                    
                    hist_vol = np.std(match_price_seq)
                    vol_ratio = hist_vol / curr_vol
                    if vol_ratio > 3.0 or vol_ratio < 0.3: continue
                    
                    match_end_idx = df.index.get_loc(m['date']) + self.window_size + projection_days
                    if match_end_idx >= len(df) or df.index[match_end_idx] >= decision_date: continue 

                    p_match = df['Close'].values[start_ptr + self.window_size - 1]
                    p_future = df['Close'].values[start_ptr + self.window_size + projection_days - 1]
                    valid_matches.append((p_future - p_match) / p_match)
                    if len(valid_matches) >= top_k: break
                except: continue
            
            if len(valid_matches) >= 3:
                pred_return = np.mean(valid_matches)
                # Count positive returns for consistency check
                pos_count = sum(1 for r in valid_matches if r > 0)
                neg_count = len(valid_matches) - pos_count
                
                bull_consistency = pos_count / len(valid_matches)
                bear_consistency = neg_count / len(valid_matches)
                
                # LONG Entry
                if pred_return > 0.01 and bull_consistency >= min_confidence:
                    holding = True
                    position_type = "LONG"
                    entry_price = current_price
                    max_price = current_price
                    days_held = 0
                
                # SHORT Entry
                elif pred_return < -0.01 and bear_consistency >= min_confidence:
                    holding = True
                    position_type = "SHORT"
                    entry_price = current_price
                    min_price = current_price
                    days_held = 0
            
            equity_curve.append(equity_curve[-1])

        return pd.DataFrame(results), equity_curve
    
class StrategyConfig:
    """
    Manages default trading parameters based on stock volatility profiles.
    """
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

    @staticmethod
    def get_default_params(ticker):
        high_beta = [
            "NVDA", "AMD", "TSLA", "MSTR", "COIN", "HOOD", 
            "PLTR", "SMCI", "MARA", "RIOT", "DKNG", "SQ", "ROKU", "NET"
        ]
        low_beta = [
            "KO", "PEP", "JNJ", "MCD", "WMT", "PG", "VZ", "T",
            "XOM", "CVX", "JPM", "BAC", "BRK-B", "UNH"
        ]
        
        if ticker in high_beta:
            return StrategyConfig.STRATEGY_BENCHMARKS['HIGH_VOLATILITY']
        elif ticker in low_beta:
            return StrategyConfig.STRATEGY_BENCHMARKS['LOW_VOLATILITY']
        else:
            return StrategyConfig.STRATEGY_BENCHMARKS['STANDARD_GROWTH']