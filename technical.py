import numpy as np
import pandas as pd

def identify_support_resistance(prices, window=10):
    """
    Identifies support and resistance levels using local minima and maxima.
    prices: pandas Series of closing prices.
    window: Number of periods left and right to consider a local min/max.
    """
    supports = []
    resistances = []

    for i in range(window, len(prices) - window):
        # Local Minimum (Support)
        if prices.iloc[i] <= min(prices.iloc[i - window : i + window + 1]):
            supports.append(prices.iloc[i])
        
        # Local Maximum (Resistance)
        if prices.iloc[i] >= max(prices.iloc[i - window : i + window + 1]):
            resistances.append(prices.iloc[i])
            
    # Group levels that are close to each other (e.g., within 0.5%)
    def cluster_levels(levels, threshold=0.005):
        if not levels:
            return []
        levels.sort()
        groups = [[levels[0]]]
        for level in levels[1:]:
            if abs(level - groups[-1][-1]) / groups[-1][-1] <= threshold:
                groups[-1].append(level)
            else:
                groups.append([level])
        return [np.mean(group) for group in groups]
        
    return cluster_levels(supports), cluster_levels(resistances)

def detect_trend(prices, short_ma=10, long_ma=50):
    """
    Provides a simple Moving Average based trend definition.
    A more advanced trend can be calculated via higher highs/lower lows.
    Returns: 'bullish', 'bearish', or 'neutral'
    """
    if len(prices) < long_ma:
        return 'neutral'
        
    short_mean = prices.tail(short_ma).mean()
    long_mean = prices.tail(long_ma).mean()
    
    # Check if short MA is significantly above/below long MA (e.g., 0.1% diff)
    diff = (short_mean - long_mean) / long_mean
    
    if diff > 0.001:
        return 'bullish'
    elif diff < -0.001:
        return 'bearish'
    else:
        return 'neutral'
