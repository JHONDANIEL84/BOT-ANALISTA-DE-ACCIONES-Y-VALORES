"""
risk_manager.py — Gestión de Riesgo Profesional
=================================================
Stops dinámicos ATR, ratio R:R, position sizing, trailing stops.
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Optional


@dataclass
class RiskProfile:
    """Perfil de riesgo calculado para una operación."""
    entrada: float
    stop_loss: float
    take_profit_1: float      # R:R 2:1
    take_profit_2: float      # R:R 3:1
    riesgo_por_unidad: float  # distancia al stop en $
    ratio_rr: float           # ratio riesgo/recompensa del TP1
    posicion_sugerida: float  # tamaño en unidades
    riesgo_total_usd: float   # $ que se arriesgan
    atr: float                # ATR actual
    atr_multiplier: float     # multiplicador usado
    kelly_fraction: float     # % sugerido por Kelly Criterion
    var_95_1d: float          # Value at Risk 1-Day 95% Confidence
    simulated_commission: float # Costo estimado de comisiones
    simulated_slippage: float # Costo estimado por deslizamiento


class RiskManager:
    """
    Gestor de riesgo profesional.
    
    Parámetros configurables:
        capital: Capital total disponible (USD)
        riesgo_pct: Porcentaje máximo de capital a arriesgar por operación (default 1%)
        atr_multiplier: Multiplicador ATR para el stop loss (default 1.5)
    """

    def __init__(self, capital: float = 10000.0, riesgo_pct: float = 1.0, atr_multiplier: float = 1.5,
                 win_rate: float = 0.45, commission_rate: float = 0.001):
        self.capital = capital
        self.riesgo_pct = riesgo_pct / 100.0   # convertir a decimal
        self.atr_multiplier = atr_multiplier
        self.win_rate = win_rate # Para Kelly Criterion (default 45% conservador)
        self.commission_rate = commission_rate # 0.1% por lado

    def calculate_risk(self, precio: float, atr: float, direccion: str,
                       soporte: float = 0.0, resistencia: float = float('inf'),
                       log_returns = None) -> RiskProfile:
        """
        Calcula el perfil de riesgo completo para una operación.
        
        Args:
            precio: Precio actual de entrada
            atr: ATR actual (volatilidad)
            direccion: 'COMPRA' o 'VENTA'
            soporte: Nivel de soporte más cercano
            resistencia: Nivel de resistencia más cercano
        
        Returns:
            RiskProfile con todos los niveles calculados
        """
        atr_stop_dist = atr * self.atr_multiplier

        if direccion == 'COMPRA':
            # Stop loss: el mayor entre ATR-based y justo debajo del soporte
            atr_stop = precio - atr_stop_dist
            support_stop = soporte * 0.995 if soporte > 0 else 0.0
            stop_loss = max(atr_stop, support_stop) if support_stop > 0 else atr_stop

            riesgo = precio - stop_loss
            take_profit_1 = precio + (riesgo * 2)   # R:R 2:1
            take_profit_2 = precio + (riesgo * 3)   # R:R 3:1

            # Si hay resistencia, usar como referencia adicional
            if resistencia < float('inf') and resistencia > precio:
                take_profit_1 = min(take_profit_1, resistencia)
                take_profit_2 = max(take_profit_2, resistencia * 1.005)

        else:  # VENTA
            atr_stop = precio + atr_stop_dist
            resistance_stop = resistencia * 1.005 if resistencia < float('inf') else float('inf')
            stop_loss = min(atr_stop, resistance_stop) if resistance_stop < float('inf') else atr_stop

            riesgo = stop_loss - precio
            take_profit_1 = precio - (riesgo * 2)
            take_profit_2 = precio - (riesgo * 3)

            if soporte > 0 and soporte < precio:
                take_profit_1 = max(take_profit_1, soporte)
                take_profit_2 = min(take_profit_2, soporte * 0.995)

        # Asegurar que riesgo sea positivo
        riesgo = abs(riesgo) if riesgo != 0 else atr_stop_dist

        # Institutional: Kelly Criterion
        # f* = W - ((1 - W) / R)
        # where W is Win Rate, R is Reward:Risk Ratio
        ratio_rr = (abs(take_profit_1 - precio)) / riesgo if riesgo > 0 else 0.0
        
        if ratio_rr > 0:
            kelly_f = self.win_rate - ((1.0 - self.win_rate) / ratio_rr)
            # Half-Kelly for safety, clamped between 0 and max_risk
            kelly_fraction = max(0.0, min(kelly_f / 2.0, self.riesgo_pct))
        else:
            kelly_fraction = 0.0
            
        # Si Kelly es 0 o negativo, usamos un riesgo fijo muy bajo (ej 0.25%) o riesgo pct original (depende de agresividad).
        # Usamos el menor entre Kelly y el riesgo_pct original para nunca sobre-arriesgar.
        riesgo_final_pct = kelly_fraction if kelly_fraction > 0 else self.riesgo_pct

        # Position sizing: cuántas unidades comprar para arriesgar solo X% del capital
        riesgo_max_usd = self.capital * riesgo_final_pct
        posicion = riesgo_max_usd / riesgo if riesgo > 0 else 0.0

        # Para activos caros (BTC, Gold), mostrar fracciones
        posicion = round(posicion, 6)
        
        # Institutional: Slippage & Commissions
        # Slippage assumes 5% of ATR as slippage per trade
        sim_slippage = (atr * 0.05) * posicion
        # Commission in/out (x2)
        sim_commission = (precio * posicion * self.commission_rate) * 2.0
        
        # Institutional: 1-Day Value at Risk (VaR) 95% Confidence
        var_95 = 0.0
        if log_returns is not None and not log_returns.empty:
            std_dev = log_returns.std()
            z_score_95 = norm.ppf(0.05) # ~ -1.645
            var_95_pct = abs(z_score_95 * std_dev)
            var_95 = (precio * posicion) * var_95_pct

        return RiskProfile(
            entrada=round(precio, 2),
            stop_loss=round(stop_loss, 2),
            take_profit_1=round(take_profit_1, 2),
            take_profit_2=round(take_profit_2, 2),
            riesgo_por_unidad=round(riesgo, 2),
            ratio_rr=round(ratio_rr, 2),
            posicion_sugerida=posicion,
            riesgo_total_usd=round(riesgo_max_usd, 2),
            atr=round(atr, 2),
            atr_multiplier=self.atr_multiplier,
            kelly_fraction=round(kelly_fraction * 100, 2),
            var_95_1d=round(var_95, 2),
            simulated_commission=round(sim_commission, 2),
            simulated_slippage=round(sim_slippage, 2)
        )

    def update_trailing_stop(self, current_price: float, current_stop: float, entry: float, direction: str, atr: float) -> tuple[float, bool]:
        """
        Updates the trailing stop.
        Moves stop to break-even if price moves 1x ATR in favor.
        Returns (new_stop, was_updated)
        """
        new_stop = current_stop
        updated = False

        if direction == 'COMPRA':
            # If price moved up by 1 ATR, move stop to entry (break-even)
            if current_price >= entry + atr and current_stop < entry:
                new_stop = entry
                updated = True
            # Basic trailing logic: if price continues to rise, trail by 1.5 ATR
            trail_level = current_price - (atr * self.atr_multiplier)
            if trail_level > new_stop:
                new_stop = trail_level
                updated = True
                
        else: # VENTA
            # If price moved down by 1 ATR, move stop to entry (break-even)
            if current_price <= entry - atr and current_stop > entry:
                new_stop = entry
                updated = True
            # Basic trailing logic
            trail_level = current_price + (atr * self.atr_multiplier)
            if trail_level < new_stop:
                new_stop = trail_level
                updated = True

        return round(new_stop, 2), updated


def format_risk_profile(rp: RiskProfile, direccion: str) -> str:
    """Formatea el perfil de riesgo como texto para consola / Telegram."""
    if direccion == 'COMPRA':
        return (
            f"📊 *GESTIÓN DE RIESGO*\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📍 Entrada: ${rp.entrada:,.2f}\n"
            f"🛑 Stop Loss: ${rp.stop_loss:,.2f} (-${rp.riesgo_por_unidad:,.2f})\n"
            f"🎯 TP1 (2:1): ${rp.take_profit_1:,.2f}\n"
            f"🎯 TP2 (3:1): ${rp.take_profit_2:,.2f}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📐 Ratio R:R: {rp.ratio_rr}:1\n"
            f"📏 ATR: ${rp.atr:,.2f} (×{rp.atr_multiplier})\n"
            f"💰 Posición: {rp.posicion_sugerida} unidades\n"
            f"⚠️ Riesgo: ${rp.riesgo_total_usd:,.2f} ({rp.riesgo_total_usd / max(rp.entrada * rp.posicion_sugerida, 1) * 100:.1f}%)\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"🏛️ *MÉTRICAS INSTITUCIONALES*\n"
            f"Kelly Fraction: {rp.kelly_fraction}%\n"
            f"VaR (95% 1D): ${rp.var_95_1d:,.2f}\n"
            f"Slippage Est.: ${rp.simulated_slippage:,.2f}\n"
            f"Comisiones Est.: ${rp.simulated_commission:,.2f}"
        )
    else:
        return (
            f"📊 *GESTIÓN DE RIESGO*\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📍 Entrada Short: ${rp.entrada:,.2f}\n"
            f"🛑 Stop Loss: ${rp.stop_loss:,.2f} (+${rp.riesgo_por_unidad:,.2f})\n"
            f"🎯 TP1 (2:1): ${rp.take_profit_1:,.2f}\n"
            f"🎯 TP2 (3:1): ${rp.take_profit_2:,.2f}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"📐 Ratio R:R: {rp.ratio_rr}:1\n"
            f"📏 ATR: ${rp.atr:,.2f} (×{rp.atr_multiplier})\n"
            f"💰 Posición: {rp.posicion_sugerida} unidades\n"
            f"⚠️ Riesgo: ${rp.riesgo_total_usd:,.2f} ({rp.riesgo_total_usd / max(rp.entrada * rp.posicion_sugerida, 1) * 100:.1f}%)\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"🏛️ *MÉTRICAS INSTITUCIONALES*\n"
            f"Kelly Fraction: {rp.kelly_fraction}%\n"
            f"VaR (95% 1D): ${rp.var_95_1d:,.2f}\n"
            f"Slippage Est.: ${rp.simulated_slippage:,.2f}\n"
            f"Comisiones Est.: ${rp.simulated_commission:,.2f}"
        )
