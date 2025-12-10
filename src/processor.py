import numpy as np
import pandas as pd

# --- HELPER FUNCTIONS ---

def z_score_normalize(window: np.array) -> np.array:
    mean = np.mean(window)
    std = np.std(window)
    if std < 1e-6: 
        return np.zeros_like(window)
    return (window - mean) / std

# --- INDICATOR CALCULATIONS ---

def calculate_rsi(data: np.array, period: int = 14) -> np.array:
    if len(data) < period + 1: return np.zeros_like(data)
    deltas = np.diff(data)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum()/period
    down = -seed[seed < 0].sum()/period
    rsi = np.zeros_like(data)
    if down == 0: rsi[:period] = 100. if up > 0 else 50.
    else: rsi[:period] = 100. - 100./(1. + up/down)

    for i in range(period, len(data)):
        delta = deltas[i - 1] 
        if delta > 0: upval, downval = delta, 0.
        else: upval, downval = 0., -delta
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        if down == 0: rsi[i] = 100. if up > 0 else 50.
        else: rsi[i] = 100. - 100./(1. + up/down)
    return rsi

def calculate_macd(data: np.array, fast=12, slow=26, signal=9) -> np.array:
    series = pd.Series(data)
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return (macd - signal_line).fillna(0).values

def calculate_bollinger_width(data: np.array, period: int = 20) -> np.array:
    series = pd.Series(data)
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * 2)
    lower = sma - (std * 2)
    middle = sma.replace(0, 1)
    return ((upper - lower) / middle).fillna(0).values

def calculate_obv(prices: np.array, volumes: np.array) -> np.array:
    obv = np.zeros_like(prices)
    obv[0] = volumes[0]
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]: obv[i] = obv[i-1] + volumes[i]
        elif prices[i] < prices[i-1]: obv[i] = obv[i-1] - volumes[i]
        else: obv[i] = obv[i-1]
    return obv

def calculate_ma_ratios(prices: np.array, window_5=5, window_60=60):
    s = pd.Series(prices)
    ma5 = s.rolling(window=window_5).mean()
    ma60 = s.rolling(window=window_60).mean()
    return (s / ma5).fillna(1.0).values, (s / ma60).fillna(1.0).values

def calculate_atr(high, low, close, period=14):
    """Average True Range (Volatility)"""
    h_l = high - low
    h_c = np.abs(high - np.roll(close, 1))
    l_c = np.abs(low - np.roll(close, 1))
    h_c[0], l_c[0] = 0, 0
    tr = np.maximum(h_l, np.maximum(h_c, l_c))
    atr = pd.Series(tr).rolling(window=period).mean().fillna(0).values
    return atr

def calculate_stochastic(high, low, close, period=14):
    """Stochastic Oscillator %K"""
    low_min = pd.Series(low).rolling(window=period).min()
    high_max = pd.Series(high).rolling(window=period).max()
    k = 100 * ((close - low_min) / (high_max - low_min))
    return k.fillna(50).values

def calculate_vwap_deviation(close, high, low, volume, window=20):
    tp = (high + low + close) / 3
    tp_v = tp * volume
    cum_vol = pd.Series(volume).rolling(window=window).sum()
    cum_tp_v = pd.Series(tp_v).rolling(window=window).sum()
    vwap = cum_tp_v / cum_vol.replace(0, 1)
    return (close - vwap) / vwap

# --- MAIN FEATURE ENGINEERING ---

def create_multifeature_windows(df: pd.DataFrame, window_size: int, benchmark_df: pd.DataFrame = None, macro_data: dict = None):
    # Extract all necessary columns
    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    volumes = df['Volume'].values
    
    # 1. Relative Strength
    if benchmark_df is not None:
        aligned_bench = benchmark_df['Close'].reindex(df.index, method='ffill')
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_strength = df['Close'] / aligned_bench
            rel_strength = rel_strength.fillna(1.0).values
    else:
        rel_strength = np.ones_like(closes)

    # 2. Calculate All Indicators (Original Set)
    rsi_vals = calculate_rsi(closes)
    bb_width = calculate_bollinger_width(closes)
    obv_vals = calculate_obv(closes, volumes)
    macd_vals = calculate_macd(closes)
    ma5_r, ma60_r = calculate_ma_ratios(closes)
    
    # New Volatile Indicators
    atr_vals = calculate_atr(highs, lows, closes)
    stoch_vals = calculate_stochastic(highs, lows, closes)
    vwap_dev = calculate_vwap_deviation(closes, highs, lows, volumes)

    features = []
    dates = []
    start_indices = []
    
    start_offset = 60 
    
    for i in range(start_offset, len(closes) - window_size):
        # Extract segments
        price_s = closes[i : i + window_size]
        vol_s = volumes[i : i + window_size]
        rs_s = rel_strength[i : i + window_size]
        rsi_s = rsi_vals[i : i + window_size]
        bb_s = bb_width[i : i + window_size]
        obv_s = obv_vals[i : i + window_size]
        macd_s = macd_vals[i : i + window_size]
        ma5_s = ma5_r[i : i + window_size]
        ma60_s = ma60_r[i : i + window_size]
        
        # New Segments
        atr_s = atr_vals[i : i + window_size]
        stoch_s = stoch_vals[i : i + window_size]
        vwap_s = vwap_dev[i : i + window_size]
        
        # Standard Normalization (No Decay)
        norm_p = z_score_normalize(price_s)
        
        if np.isnan(norm_p).any(): continue

        # --- WEIGHTING STRATEGY (ORIGINAL 12-FACTOR) ---
        combined_feature = np.concatenate([
            norm_p * 4.0,               # Price Shape (King)
            z_score_normalize(vol_s) * 1.0,
            z_score_normalize(rs_s)  * 2.0,
            (rsi_s - 50)/25.0 * 0.5,
            z_score_normalize(bb_s)  * 0.5,
            z_score_normalize(obv_s) * 0.5,
            z_score_normalize(macd_s) * 2.0, # Momentum
            z_score_normalize(ma5_s) * 1.0,
            z_score_normalize(ma60_s) * 1.5,
            
            # Volatile Stocks Factors
            z_score_normalize(atr_s) * 1.5,   
            (stoch_s - 50)/25.0 * 1.0,        
            z_score_normalize(vwap_s) * 1.5   
        ])
        
        if np.isnan(combined_feature).any(): continue

        features.append(combined_feature)
        dates.append(df.index[i])
        start_indices.append(i)
        
    return np.array(features), np.array(dates), np.array(start_indices)