"""
technical.py — Indicadores Técnicos de Grado Profesional
=========================================================
Bollinger Bands, ATR, ADX, EMA Ribbon, Stochastic RSI, VWAP,
RSI, MACD, Soporte/Resistencia, Price Action.
"""

import numpy as np
import pandas as pd


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Soporte y Resistencia
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def identify_support_resistance(prices, window=10):
    """
    Identifica niveles de soporte y resistencia usando mínimos/máximos locales.
    Los agrupa por proximidad (threshold 0.5%) para evitar ruido.
    """
    supports = []
    resistances = []

    for i in range(window, len(prices) - window):
        segment = prices.iloc[i - window: i + window + 1]
        if prices.iloc[i] <= segment.min():
            supports.append(prices.iloc[i])
        if prices.iloc[i] >= segment.max():
            resistances.append(prices.iloc[i])

    def cluster_levels(levels, threshold=0.005):
        if not levels:
            return []
        levels.sort()
        groups = [[levels[0]]]
        for level in levels[1:]:
            if abs(level - groups[-1][-1]) / max(groups[-1][-1], 1e-9) <= threshold:
                groups[-1].append(level)
            else:
                groups.append([level])
        return [np.mean(group) for group in groups]

    return cluster_levels(supports), cluster_levels(resistances)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RSI (Relative Strength Index)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calculate_rsi(prices, period=14):
    """RSI clásico de Wilder. Retorna Series."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1.0 / period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = gain / loss.replace(0, 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Stochastic RSI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calculate_stochastic_rsi(prices, rsi_period=14, stoch_period=14, smooth_k=3, smooth_d=3):
    """
    Stochastic RSI: aplica estocástico sobre el RSI.
    Retorna (%K, %D) como Series.
    """
    rsi = calculate_rsi(prices, rsi_period)
    rsi_min = rsi.rolling(window=stoch_period).min()
    rsi_max = rsi.rolling(window=stoch_period).max()
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-9) * 100
    k = stoch_rsi.rolling(window=smooth_k).mean()
    d = k.rolling(window=smooth_d).mean()
    return k, d


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MACD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calculate_macd(prices, fast=12, slow=26, smooth=9):
    """MACD estándar. Retorna (macd_line, signal_line, histogram)."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=smooth, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def detect_macd_cross(hist):
    """
    Detecta cruce del histograma MACD.
    Retorna: 'bullish_cross', 'bearish_cross', o 'none'.
    """
    if len(hist) < 3:
        return 'none'
    curr = hist.iloc[-1]
    prev = hist.iloc[-2]
    if prev <= 0 and curr > 0:
        return 'bullish_cross'
    elif prev >= 0 and curr < 0:
        return 'bearish_cross'
    return 'none'


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Bandas de Bollinger
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calculate_bollinger_bands(prices, period=20, std_dev=2.0):
    """
    Bandas de Bollinger.
    Retorna dict con: upper, middle, lower, bandwidth, %b (percent_b).
    """
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    bandwidth = (upper - lower) / (middle + 1e-9)
    percent_b = (prices - lower) / (upper - lower + 1e-9)
    return {
        'upper': upper,
        'middle': middle,
        'lower': lower,
        'bandwidth': bandwidth,
        'percent_b': percent_b,
    }


def detect_bollinger_squeeze(bandwidth, threshold=0.02):
    """
    Detecta Bollinger Squeeze: baja volatilidad que precede explosión.
    Retorna True si bandwidth actual está por debajo del threshold.
    """
    if bandwidth.empty:
        return False
    return float(bandwidth.iloc[-1]) < threshold


def detect_bollinger_breakout(prices, upper, lower):
    """
    Detecta breakout de Bollinger.
    Retorna: 'bullish_breakout', 'bearish_breakout', 'none'.
    """
    if len(prices) < 2:
        return 'none'
    curr_price = prices.iloc[-1]
    prev_price = prices.iloc[-2]
    curr_upper = upper.iloc[-1]
    curr_lower = lower.iloc[-1]
    prev_upper = upper.iloc[-2]
    prev_lower = lower.iloc[-2]

    if prev_price <= prev_upper and curr_price > curr_upper:
        return 'bullish_breakout'
    elif prev_price >= prev_lower and curr_price < curr_lower:
        return 'bearish_breakout'
    return 'none'


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ATR (Average True Range) — Volatilidad
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calculate_atr(df, period=14):
    """
    ATR de Wilder. Requiere columnas High, Low, Close.
    Retorna Series.
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1.0 / period, min_periods=period).mean()
    return atr


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ADX (Average Directional Index) — Fuerza de Tendencia
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calculate_adx(df, period=14):
    """
    ADX de Wilder. Mide la FUERZA de la tendencia (no la dirección).
    < 20: sin tendencia | 20–25: tendencia débil | 25–50: fuerte | > 50: muy fuerte.
    Retorna (adx, plus_di, minus_di) como Series.
    """
    high = df['High']
    low = df['Low']
    close = df['Close']

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr = calculate_atr(df, period)

    plus_di = 100 * (plus_dm.ewm(alpha=1.0 / period, min_periods=period).mean() / (atr + 1e-9))
    minus_di = 100 * (minus_dm.ewm(alpha=1.0 / period, min_periods=period).mean() / (atr + 1e-9))

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9))
    adx = dx.ewm(alpha=1.0 / period, min_periods=period).mean()

    return adx, plus_di, minus_di


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EMA Ribbon (8, 13, 21, 55)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calculate_ema_ribbon(prices, periods=(8, 13, 21, 55)):
    """
    Calcula un ribbon de EMAs.
    Retorna dict {period: ema_series}.
    """
    return {p: prices.ewm(span=p, adjust=False).mean() for p in periods}


def detect_ema_alignment(ribbon):
    """
    Detecta alineación del EMA Ribbon.
    Alcista: EMA8 > EMA13 > EMA21 > EMA55
    Bajista: EMA8 < EMA13 < EMA21 < EMA55
    Retorna: 'bullish', 'bearish', 'mixed'.
    """
    periods = sorted(ribbon.keys())
    vals = [float(ribbon[p].iloc[-1]) for p in periods]

    if all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)):
        return 'bullish'
    elif all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)):
        return 'bearish'
    return 'mixed'


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VWAP Simplificado
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calculate_vwap(df):
    """
    VWAP simplificado (acumulativo sobre el DataFrame disponible).
    Requiere columnas High, Low, Close, Volume.
    """
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3.0
    cumulative_tp_vol = (typical_price * df['Volume']).cumsum()
    cumulative_vol = df['Volume'].cumsum()
    vwap = cumulative_tp_vol / (cumulative_vol + 1e-9)
    return vwap


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Detección de Tendencia por Media Móvil
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_trend(prices, short_ma=10, long_ma=50):
    """
    Tendencia por cruce de medias móviles.
    Retorna: 'bullish', 'bearish', 'neutral'.
    """
    if len(prices) < long_ma:
        return 'neutral'
    short_mean = prices.tail(short_ma).mean()
    long_mean = prices.tail(long_ma).mean()
    diff = (short_mean - long_mean) / (long_mean + 1e-9)
    if diff > 0.0005:
        return 'bullish'
    elif diff < -0.0005:
        return 'bearish'
    return 'neutral'


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Price Action — Higher Highs / Lower Lows
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_price_action_trend(df, lookback=20):
    """
    Analiza estructura de precio: HH/HL (alcista) vs LH/LL (bajista).
    Usa los últimos `lookback` registros divididos en 4 segmentos.
    """
    if len(df) < lookback:
        return 'neutral'

    recent = df.tail(lookback)
    seg_len = lookback // 4
    segments = [recent.iloc[i * seg_len: (i + 1) * seg_len] for i in range(4)]

    highs = [seg['High'].max() for seg in segments]
    lows = [seg['Low'].min() for seg in segments]

    higher_highs = all(highs[i] < highs[i + 1] for i in range(len(highs) - 1))
    higher_lows = all(lows[i] < lows[i + 1] for i in range(len(lows) - 1))
    lower_highs = all(highs[i] > highs[i + 1] for i in range(len(highs) - 1))
    lower_lows = all(lows[i] > lows[i + 1] for i in range(len(lows) - 1))

    if higher_highs and higher_lows:
        return 'bullish'
    elif lower_highs and lower_lows:
        return 'bearish'
    return 'neutral'


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Análisis Técnico Completo (wrapper)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def full_technical_analysis(df):
    """
    Ejecuta TODOS los indicadores sobre un DataFrame OHLCV.
    Retorna un diccionario con todos los resultados listos para el motor de señales.
    """
    close = df['Close']

    # RSI
    rsi = calculate_rsi(close)
    rsi_val = float(rsi.iloc[-1]) if not rsi.empty else 50.0

    # Stochastic RSI
    stoch_k, stoch_d = calculate_stochastic_rsi(close)
    stoch_k_val = float(stoch_k.iloc[-1]) if not stoch_k.empty and not pd.isna(stoch_k.iloc[-1]) else 50.0
    stoch_d_val = float(stoch_d.iloc[-1]) if not stoch_d.empty and not pd.isna(stoch_d.iloc[-1]) else 50.0

    # MACD
    macd, macd_signal, macd_hist = calculate_macd(close)
    macd_cross = detect_macd_cross(macd_hist)

    # Bollinger
    bb = calculate_bollinger_bands(close)
    bb_squeeze = detect_bollinger_squeeze(bb['bandwidth'])
    bb_breakout = detect_bollinger_breakout(close, bb['upper'], bb['lower'])
    bb_percent_b = float(bb['percent_b'].iloc[-1]) if not bb['percent_b'].empty else 0.5

    # ATR
    atr = calculate_atr(df)
    atr_val = float(atr.iloc[-1]) if not atr.empty else 0.0

    # ADX
    adx, plus_di, minus_di = calculate_adx(df)
    adx_val = float(adx.iloc[-1]) if not adx.empty and not pd.isna(adx.iloc[-1]) else 0.0
    plus_di_val = float(plus_di.iloc[-1]) if not plus_di.empty and not pd.isna(plus_di.iloc[-1]) else 0.0
    minus_di_val = float(minus_di.iloc[-1]) if not minus_di.empty and not pd.isna(minus_di.iloc[-1]) else 0.0

    # EMA Ribbon
    ribbon = calculate_ema_ribbon(close)
    ema_align = detect_ema_alignment(ribbon)

    # VWAP
    vwap = calculate_vwap(df)
    vwap_val = float(vwap.iloc[-1]) if not vwap.empty else 0.0

    # Tendencia MA
    ma_trend = detect_trend(close, short_ma=5, long_ma=20)

    # Price Action
    pa_trend = detect_price_action_trend(df)

    # Soporte / Resistencia
    supports, resistances = identify_support_resistance(close, window=5)
    current_close = float(close.iloc[-1])
    closest_support = max([s for s in supports if s < current_close], default=0.0)
    closest_resistance = min([r for r in resistances if r > current_close], default=float('inf'))

    return {
        # Precio
        'close': current_close,
        'vwap': vwap_val,
        'above_vwap': current_close > vwap_val,

        # RSI
        'rsi': rsi_val,
        'rsi_overbought': rsi_val > 70,
        'rsi_oversold': rsi_val < 30,
        'rsi_favorable_buy': 30 <= rsi_val <= 65,
        'rsi_favorable_sell': 35 <= rsi_val <= 70,

        # Stochastic RSI
        'stoch_k': stoch_k_val,
        'stoch_d': stoch_d_val,
        'stoch_oversold': stoch_k_val < 20,
        'stoch_overbought': stoch_k_val > 80,

        # MACD
        'macd_cross': macd_cross,
        'macd_hist': float(macd_hist.iloc[-1]) if not macd_hist.empty else 0.0,

        # Bollinger
        'bb_squeeze': bb_squeeze,
        'bb_breakout': bb_breakout,
        'bb_percent_b': bb_percent_b,

        # ATR (volatilidad)
        'atr': atr_val,
        'atr_pct': (atr_val / current_close * 100) if current_close > 0 else 0.0,

        # ADX (fuerza de tendencia)
        'adx': adx_val,
        'adx_strong': adx_val > 25,
        'adx_very_strong': adx_val > 50,
        'plus_di': plus_di_val,
        'minus_di': minus_di_val,

        # EMA Ribbon
        'ema_alignment': ema_align,

        # Tendencias
        'ma_trend': ma_trend,
        'pa_trend': pa_trend,

        # Soporte / Resistencia
        'support': closest_support,
        'resistance': closest_resistance,
    }
