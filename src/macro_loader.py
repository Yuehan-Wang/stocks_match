import yfinance as yf
import pandas as pd

class MacroLoader:
    def __init__(self):
        # Fetch data
        self.yields = yf.Ticker("^TNX").history(period="max")['Close']
        self.vix = yf.Ticker("^VIX").history(period="max")['Close']
        
        # 1. Ensure Index is DatetimeIndex
        self.yields.index = pd.to_datetime(self.yields.index)
        self.vix.index = pd.to_datetime(self.vix.index)

        # 2. Force Timezone-Naive (Remove timezone info)
        if self.yields.index.tz is not None:
            self.yields.index = self.yields.index.tz_localize(None)
        if self.vix.index.tz is not None:
            self.vix.index = self.vix.index.tz_localize(None)

    def get_latest_context(self):
        """Returns the most recent macro data available."""
        if len(self.yields) == 0:
            return {'yield': 4.0, 'vix': 15.0} 
            
        return {
            'yield': self.yields.iloc[-1],
            'vix': self.vix.iloc[-1]
        }

    def get_context_at_date(self, target_date):
        """Finds the macro data for a specific historical date."""
        try:
            ts = pd.Timestamp(target_date)
            if ts.tz is not None: ts = ts.tz_localize(None)
            y_val = self.yields.asof(ts)
            v_val = self.vix.asof(ts)
            if pd.isna(y_val) or pd.isna(v_val): return None
            return {'yield': y_val, 'vix': v_val}
        except:
            return None

    def get_full_series(self):
        """
        Returns the full Series objects for batch processing.
        Used for Improvement 3 (Soft Macro Vectorization).
        """
        return {'yield': self.yields, 'vix': self.vix}