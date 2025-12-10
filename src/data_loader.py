import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker: str, period: str = "max") -> pd.DataFrame:
    """
    Fetches historical data including Open, High, Low, Close, Volume.
    """
    ticker_obj = yf.Ticker(ticker)
    df = ticker_obj.history(period=period)
    
    # Basic cleaning
    if 'Volume' in df.columns:
        df = df[df['Volume'] > 0]
    
    df = df.dropna()
    
    # Ensure High/Low/Close/Volume columns exist
    required_cols = ['High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
            
    return df