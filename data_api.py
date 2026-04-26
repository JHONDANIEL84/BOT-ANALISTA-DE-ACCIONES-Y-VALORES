import yfinance as yf
import pandas as pd
import time


class DataFetcher:
    def __init__(self, ticker):
        self.ticker = ticker

    def _fetch_with_retry(self, period, interval, retries=3):
        """Internal helper to fetch data with retries."""
        for attempt in range(retries):
            try:
                stock = yf.Ticker(self.ticker)
                df = stock.history(period=period, interval=interval)
                if not df.empty:
                    return df
                # If empty, maybe wait a bit and retry
                time.sleep(1)
            except Exception as e:
                print(f"[!] Fetch attempt {attempt+1} failed for {self.ticker}: {e}")
                time.sleep(1)
        return pd.DataFrame()

    def fetch_historical_data(self, period="60d", interval="1h"):
        """
        Fetches historical OHLCV data for model training.
        Default: 60 days of hourly bars (reliable with yfinance).
        """
        df = self._fetch_with_retry(period=period, interval=interval)
        if df.empty:
            return pd.DataFrame()
        
        df.reset_index(inplace=True)
        # yfinance returns 'Datetime' for intraday and 'Date' for daily
        time_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
        df = df.rename(columns={time_col: 'Datetime'})

        return df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]

    def fetch_latest_data(self, period="5d", interval="5m", last_n=100):
        """
        Fetches the most recent bars for real-time monitoring.
        Default: last 5 days of 5-minute bars.
        """
        df = self._fetch_with_retry(period=period, interval=interval)
        if df.empty:
            # Fallback to a simpler period if 5d fails
            df = self._fetch_with_retry(period="1d", interval=interval)
            
        if df.empty:
            return pd.DataFrame()
        
        df.reset_index(inplace=True)
        time_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
        df = df.rename(columns={time_col: 'Datetime'})

        return df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(last_n)

    def verify_ticker(self):
        """Verify if the ticker is valid and has recent data."""
        try:
            stock = yf.Ticker(self.ticker)
            # Use 5d instead of 1d to account for weekends/holidays
            df = stock.history(period="5d")
            return not df.empty
        except:
            return False
