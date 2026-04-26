"""
signal_engine.py — Motor de Señales por Confluencia
=====================================================
Puntaje 0–100 basado en la confluencia de múltiples indicadores.
Solo dispara alertas cuando hay suficiente acuerdo entre señales.

Umbrales:
  COMPRA / VENTA ≥ 60 pts → Señal activa
  ≥ 80 pts → Señal FUERTE
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SignalDetail:
    """Detalle de un componente de señal individual."""
    nombre: str
    puntos: int          # puntos otorgados (positivo = a favor)
    max_puntos: int      # máximo posible
    razon: str           # explicación breve


@dataclass
class SignalResult:
    """Resultado consolidado del motor de señales."""
    direccion: str       # 'COMPRA', 'VENTA', 'NEUTRAL'
    score_compra: int    # 0–100
    score_venta: int     # 0–100
    fuerza: str          # 'DÉBIL', 'MEDIA', 'FUERTE', 'MUY FUERTE'
    detalles: List[SignalDetail] = field(default_factory=list)
    resumen: str = ""

    @property
    def es_alerta(self):
        """True si el score alcanza el umbral mínimo de 60."""
        return max(self.score_compra, self.score_venta) >= 60


def classify_strength(score: int) -> str:
    """Clasifica la fuerza de la señal."""
    if score >= 80:
        return "MUY FUERTE"
    elif score >= 70:
        return "FUERTE"
    elif score >= 60:
        return "MEDIA"
    else:
        return "DÉBIL"


def evaluate_signal(tech: Dict, ai_trend: str, ai_confidence: float) -> SignalResult:
    """
    Motor principal de confluencia.
    
    Args:
        tech: Diccionario de full_technical_analysis()
        ai_trend: Predicción del modelo ('bullish', 'bearish', 'neutral')
        ai_confidence: Confianza del modelo (0.0–1.0)
    
    Returns:
        SignalResult con score, dirección, fuerza y detalles.
    """
    buy_pts = 0
    sell_pts = 0
    details_buy = []
    details_sell = []

    # ── 1. Tendencia por Media Móvil (max 15 pts) ─────────────
    ma = tech['ma_trend']
    if ma == 'bullish':
        buy_pts += 15
        details_buy.append(SignalDetail("Media Móvil", 15, 15, "MA corta > MA larga → alcista"))
    elif ma == 'bearish':
        sell_pts += 15
        details_sell.append(SignalDetail("Media Móvil", 15, 15, "MA corta < MA larga → bajista"))
    else:
        details_buy.append(SignalDetail("Media Móvil", 0, 15, "Sin cruce claro"))
        details_sell.append(SignalDetail("Media Móvil", 0, 15, "Sin cruce claro"))

    # ── 2. ADX — Fuerza de Tendencia (max 10 pts) ─────────────
    adx = tech['adx']
    if tech['adx_strong']:
        if tech['plus_di'] > tech['minus_di']:
            buy_pts += 10
            details_buy.append(SignalDetail("ADX", 10, 10, f"ADX {adx:.0f} fuerte, +DI domina"))
        else:
            sell_pts += 10
            details_sell.append(SignalDetail("ADX", 10, 10, f"ADX {adx:.0f} fuerte, -DI domina"))
    else:
        details_buy.append(SignalDetail("ADX", 0, 10, f"ADX {adx:.0f} débil — sin tendencia clara"))
        details_sell.append(SignalDetail("ADX", 0, 10, f"ADX {adx:.0f} débil — sin tendencia clara"))

    # ── 3. RSI — Zona Favorable (max 10 pts) ──────────────────
    rsi = tech['rsi']
    if tech['rsi_favorable_buy'] and not tech['rsi_overbought']:
        buy_pts += 10
        details_buy.append(SignalDetail("RSI", 10, 10, f"RSI {rsi:.0f} en zona de compra favorable"))
    elif tech['rsi_oversold']:
        buy_pts += 10
        details_buy.append(SignalDetail("RSI", 10, 10, f"RSI {rsi:.0f} sobreventa → rebote probable"))
    
    if tech['rsi_favorable_sell'] and not tech['rsi_oversold']:
        sell_pts += 10
        details_sell.append(SignalDetail("RSI", 10, 10, f"RSI {rsi:.0f} en zona de venta favorable"))
    elif tech['rsi_overbought']:
        sell_pts += 10
        details_sell.append(SignalDetail("RSI", 10, 10, f"RSI {rsi:.0f} sobrecompra → caída probable"))

    # ── 4. MACD Cruce (max 15 pts) ────────────────────────────
    macd_cross = tech['macd_cross']
    if macd_cross == 'bullish_cross':
        buy_pts += 15
        details_buy.append(SignalDetail("MACD", 15, 15, "Cruce alcista del histograma"))
    elif macd_cross == 'bearish_cross':
        sell_pts += 15
        details_sell.append(SignalDetail("MACD", 15, 15, "Cruce bajista del histograma"))
    else:
        # Histograma positivo/negativo creciente da puntos parciales
        hist = tech['macd_hist']
        if hist > 0:
            buy_pts += 5
            details_buy.append(SignalDetail("MACD", 5, 15, "Histograma positivo (sin cruce)"))
        elif hist < 0:
            sell_pts += 5
            details_sell.append(SignalDetail("MACD", 5, 15, "Histograma negativo (sin cruce)"))

    # ── 5. Bollinger Bands (max 10 pts) ───────────────────────
    bb_breakout = tech['bb_breakout']
    if bb_breakout == 'bullish_breakout':
        buy_pts += 10
        details_buy.append(SignalDetail("Bollinger", 10, 10, "Breakout alcista por encima de banda superior"))
    elif bb_breakout == 'bearish_breakout':
        sell_pts += 10
        details_sell.append(SignalDetail("Bollinger", 10, 10, "Breakout bajista por debajo de banda inferior"))
    elif tech['bb_squeeze']:
        # Squeeze = volatilidad baja, explosión inminente — ambos lados
        buy_pts += 3
        sell_pts += 3
        details_buy.append(SignalDetail("Bollinger", 3, 10, "Squeeze detectado — explosión inminente"))
        details_sell.append(SignalDetail("Bollinger", 3, 10, "Squeeze detectado — explosión inminente"))
    else:
        pct_b = tech['bb_percent_b']
        if pct_b < 0.2:
            buy_pts += 5
            details_buy.append(SignalDetail("Bollinger", 5, 10, f"%B={pct_b:.2f} cerca de banda inferior → sobreventa"))
        elif pct_b > 0.8:
            sell_pts += 5
            details_sell.append(SignalDetail("Bollinger", 5, 10, f"%B={pct_b:.2f} cerca de banda superior → sobrecompra"))

    # ── 6. EMA Ribbon (max 10 pts) ────────────────────────────
    ema = tech['ema_alignment']
    if ema == 'bullish':
        buy_pts += 10
        details_buy.append(SignalDetail("EMA Ribbon", 10, 10, "EMAs perfectamente alineadas al alza"))
    elif ema == 'bearish':
        sell_pts += 10
        details_sell.append(SignalDetail("EMA Ribbon", 10, 10, "EMAs perfectamente alineadas a la baja"))
    else:
        details_buy.append(SignalDetail("EMA Ribbon", 0, 10, "EMAs mixtas — sin alineación"))
        details_sell.append(SignalDetail("EMA Ribbon", 0, 10, "EMAs mixtas — sin alineación"))

    # ── 7. Price Action HH/HL o LH/LL (max 10 pts) ───────────
    pa = tech['pa_trend']
    if pa == 'bullish':
        buy_pts += 10
        details_buy.append(SignalDetail("Price Action", 10, 10, "Máximos más altos + Mínimos más altos"))
    elif pa == 'bearish':
        sell_pts += 10
        details_sell.append(SignalDetail("Price Action", 10, 10, "Máximos más bajos + Mínimos más bajos"))
    else:
        details_buy.append(SignalDetail("Price Action", 0, 10, "Sin estructura clara"))
        details_sell.append(SignalDetail("Price Action", 0, 10, "Sin estructura clara"))

    # ── 8. Modelo IA (max 15 pts) ─────────────────────────────
    conf_bonus = int(ai_confidence * 15)  # Escalar confianza a 0–15
    if ai_trend == 'bullish':
        buy_pts += conf_bonus
        details_buy.append(SignalDetail("Modelo IA", conf_bonus, 15, f"Predicción ALCISTA ({ai_confidence:.0%})"))
    elif ai_trend == 'bearish':
        sell_pts += conf_bonus
        details_sell.append(SignalDetail("Modelo IA", conf_bonus, 15, f"Predicción BAJISTA ({ai_confidence:.0%})"))
    else:
        details_buy.append(SignalDetail("Modelo IA", 0, 15, f"Predicción NEUTRAL ({ai_confidence:.0%})"))
        details_sell.append(SignalDetail("Modelo IA", 0, 15, f"Predicción NEUTRAL ({ai_confidence:.0%})"))

    # ── 9. Stochastic RSI (max 5 pts) ─────────────────────────
    if tech['stoch_oversold']:
        buy_pts += 5
        details_buy.append(SignalDetail("Stoch RSI", 5, 5, f"K={tech['stoch_k']:.0f} sobreventa extrema"))
    elif tech['stoch_overbought']:
        sell_pts += 5
        details_sell.append(SignalDetail("Stoch RSI", 5, 5, f"K={tech['stoch_k']:.0f} sobrecompra extrema"))

    # ── Determinar dirección y fuerza ─────────────────────────
    # Cap at 100
    buy_pts = min(buy_pts, 100)
    sell_pts = min(sell_pts, 100)

    if buy_pts >= 60 and buy_pts > sell_pts:
        direccion = 'COMPRA'
        fuerza = classify_strength(buy_pts)
        detalles = [d for d in details_buy if d.puntos > 0]
        resumen = f"Señal de COMPRA ({buy_pts}/100) — {fuerza}"
    elif sell_pts >= 60 and sell_pts > buy_pts:
        direccion = 'VENTA'
        fuerza = classify_strength(sell_pts)
        detalles = [d for d in details_sell if d.puntos > 0]
        resumen = f"Señal de VENTA ({sell_pts}/100) — {fuerza}"
    else:
        score_max = max(buy_pts, sell_pts)
        fuerza = classify_strength(score_max)
        detalles = []
        resumen = f"Sin señal (Compra: {buy_pts}/100, Venta: {sell_pts}/100)"
        direccion = 'NEUTRAL'

    return SignalResult(
        direccion=direccion,
        score_compra=buy_pts,
        score_venta=sell_pts,
        fuerza=fuerza,
        detalles=detalles,
        resumen=resumen,
    )
