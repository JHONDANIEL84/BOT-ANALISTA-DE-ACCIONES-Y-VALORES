"""
agent.py — Agente de Trading Profesional
==========================================
Motor de señales por confluencia, gestión de riesgo ATR,
dashboard profesional en consola, logging operativo,
cooldown de alertas, análisis multi-indicador.

Uso:
  python agent.py                          # Monitoreo BTC + Oro por defecto
  python agent.py --tickers BTC-USD,GC=F   # Tickers personalizados
  python agent.py --once                   # Una sola ejecución
  python agent.py --capital 5000           # Capital para position sizing
"""

import os
import json
import argparse
import time
import logging
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from colorama import init, Fore, Style, Back
from dotenv import load_dotenv

from data_api import DataFetcher
from technical import full_technical_analysis
from signal_engine import evaluate_signal
from risk_manager import RiskManager, format_risk_profile
from transformer import TimeSeriesTransformer, create_sequences
from telegram_notifier import TelegramNotifier

# ──────────────────────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────────────────────
import sys
import io
# Forzar UTF-8 en Windows para caracteres Unicode
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

init(autoreset=True)
load_dotenv()

SEQ_LENGTH = 30
INPUT_DIM = 10  # OHLCV + RSI + MACD + ATR + BB_pct + ADX
D_MODEL = 32
NHEAD = 4
NUM_LAYERS = 2

ALERT_COOLDOWN_MIN = 15   # Minutos entre alertas del mismo tipo/ticker

# Logging profesional
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("MarketAgent")
logger.setLevel(logging.DEBUG)

# Normal text log
fh = logging.FileHandler(os.path.join(LOG_DIR, "agent.log"), encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(fh)

# Structured JSON log (Institutional Standard)
json_fh = logging.FileHandler(os.path.join(LOG_DIR, "structured.jsonl"), encoding="utf-8")
json_fh.setLevel(logging.INFO)
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "time": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage()
        }
        if hasattr(record, 'json_data'):
            log_record.update(record.json_data)
        return json.dumps(log_record)
json_fh.setFormatter(JsonFormatter())
logger.addHandler(json_fh)

ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)
logger.addHandler(ch)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def fmt(val, decimals=2):
    """Formatea un precio. Usa separador de miles."""
    if val in (float('inf'), float('-inf')):
        return "N/A"
    return f"${val:,.{decimals}f}"


def get_model_paths(ticker):
    clean = ticker.replace("=", "").replace("-", "_")
    return f"{clean}_model.pt", f"{clean}_scaler.npz"


def get_state_path(ticker):
    clean = ticker.replace("=", "").replace("-", "_")
    return f"{clean}_state.json"


def save_model(ticker, model, scaler):
    mp, sp = get_model_paths(ticker)
    torch.save(model.state_dict(), mp)
    np.savez(sp, mean=scaler.mean_, scale=scaler.scale_)
    logger.info(f"Modelo guardado: {mp}")


def load_model(ticker):
    mp, sp = get_model_paths(ticker)
    if not (os.path.exists(mp) and os.path.exists(sp)):
        return None, None
    print(Fore.CYAN + f"  ✓ Modelo encontrado para {ticker}")
    model = TimeSeriesTransformer(input_dim=INPUT_DIM, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS)
    model.load_state_dict(torch.load(mp, weights_only=True))
    model.eval()
    data = np.load(sp)
    scaler = StandardScaler()
    scaler.mean_ = data['mean']
    scaler.scale_ = data['scale']
    scaler.n_features_in_ = INPUT_DIM
    return model, scaler


def load_state(ticker):
    path = get_state_path(ticker)
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "last_trend": "neutral",
        "last_price": 0.0,
        "last_alert_time": None,
        "last_alert_type": None,
        "alert_history": [],
    }


def save_state(ticker, state):
    path = get_state_path(ticker)
    with open(path, 'w') as f:
        json.dump(state, f, default=str)


# ──────────────────────────────────────────────────────────────
# Entrenamiento
# ──────────────────────────────────────────────────────────────

def train_model(ticker, period="60d", interval="1h"):
    """Entrena el modelo Transformer con datos históricos."""
    print(Fore.YELLOW + f"  ⏳ Descargando datos históricos para {ticker} ({period}/{interval})...")
    fetcher = DataFetcher(ticker)
    df = fetcher.fetch_historical_data(period=period, interval=interval)
    if df.empty:
        print(Fore.RED + "  ✗ Sin datos de Yahoo Finance.")
        return None, None

    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Pre-calcular features técnicos
    from technical import calculate_rsi, calculate_macd, calculate_atr, calculate_bollinger_bands, calculate_adx
    df['RSI'] = calculate_rsi(df['Close']).fillna(50)
    _, _, macd_hist = calculate_macd(df['Close'])
    df['MACD_Hist'] = macd_hist.fillna(0)
    df['ATR'] = calculate_atr(df).fillna(0)
    bb = calculate_bollinger_bands(df['Close'])
    df['BB_pct'] = bb['percent_b'].fillna(0.5)
    adx, _, _ = calculate_adx(df)
    df['ADX'] = adx.fillna(0)

    tech_features = features + ['RSI', 'MACD_Hist', 'ATR', 'BB_pct', 'ADX']
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[tech_features].values)

    X, y = create_sequences(scaled, seq_length=SEQ_LENGTH, target_idx=3)
    if len(X) == 0:
        print(Fore.RED + "  ✗ Datos insuficientes para secuencias.")
        return None, None

    print(Fore.YELLOW + f"  ⏳ Entrenando modelo ({len(X)} secuencias)...")
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    model = TimeSeriesTransformer(input_dim=INPUT_DIM, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_loss, best_state = float('inf'), None
    for epoch in range(15):
        model.train()
        epoch_loss, batches = 0.0, 0
        for i in range(0, len(X_t), 64):
            bx, by = X_t[i:i+64], y_t[i:i+64]
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1
        avg = epoch_loss / max(batches, 1)
        scheduler.step()
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    print(Fore.GREEN + f"  ✓ Entrenamiento completo (loss: {best_loss:.4f})")
    save_model(ticker, model, scaler)
    return model, scaler


# ──────────────────────────────────────────────────────────────
# Dashboard Profesional
# ──────────────────────────────────────────────────────────────

def print_header(tickers_info, poll_interval, capital, notifier):
    """Imprime el encabezado profesional."""
    w = 72
    print()
    print(Fore.CYAN + Style.BRIGHT + "╔" + "═" * w + "╗")
    print(Fore.CYAN + Style.BRIGHT + "║" + "  AGENTE DE TRADING PROFESIONAL".center(w) + "║")
    print(Fore.CYAN + Style.BRIGHT + "║" + "  Análisis por Confluencia + IA".center(w) + "║")
    print(Fore.CYAN + Style.BRIGHT + "╠" + "═" * w + "╣")
    print(Fore.CYAN + "║" + f"  Activos     : {', '.join(tickers_info.keys())}".ljust(w) + "║")
    print(Fore.CYAN + "║" + f"  Capital     : ${capital:,.2f}".ljust(w) + "║")
    print(Fore.CYAN + "║" + f"  Intervalo   : {poll_interval}s".ljust(w) + "║")
    tg_status = "✓ Activado" if (notifier and notifier.bot_token) else "✗ Desactivado"
    print(Fore.CYAN + "║" + f"  Telegram    : {tg_status}".ljust(w) + "║")
    print(Fore.CYAN + "║" + f"  Cooldown    : {ALERT_COOLDOWN_MIN} min entre alertas".ljust(w) + "║")
    print(Fore.CYAN + "║" + f"  Inicio      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".ljust(w) + "║")
    print(Fore.CYAN + Style.BRIGHT + "╚" + "═" * w + "╝")
    print()


def print_ticker_analysis(ticker, price, tech, signal, risk_profile=None):
    """Imprime el panel de análisis para un ticker."""
    w = 72
    dir_color = Fore.GREEN if signal.direccion == 'COMPRA' else Fore.RED if signal.direccion == 'VENTA' else Fore.YELLOW
    trend_es = {
        'bullish': 'ALCISTA', 'bearish': 'BAJISTA', 'neutral': 'NEUTRAL', 'mixed': 'MIXTO'
    }

    # Score bar visual
    buy_bar = "█" * (signal.score_compra // 5) + "░" * (20 - signal.score_compra // 5)
    sell_bar = "█" * (signal.score_venta // 5) + "░" * (20 - signal.score_venta // 5)

    print(Fore.CYAN + Style.BRIGHT + "┌" + "─" * w + "┐")
    print(Fore.CYAN + Style.BRIGHT + "│" + f"  {ticker}  │  {fmt(price)}  │  {datetime.now().strftime('%H:%M:%S')}".ljust(w) + "│")
    print(Fore.CYAN + "├" + "─" * w + "┤")
    
    # ── Institutional Metrics
    regime_color = Fore.GREEN if tech['market_regime'] == 'trending' else Fore.YELLOW if tech['market_regime'] == 'mean_reverting' else Fore.WHITE
    print(Fore.CYAN + "│" + f"  🏛️ INSTITUCIONAL: Hurst={tech['hurst_exponent']:.2f} ({regime_color}{tech['market_regime'].upper()}{Fore.WHITE}) │ Vol Z={tech['volatility_zscore']:.2f}".ljust(w + 27) + Fore.CYAN + "│")
    print(Fore.CYAN + "├" + "─" * w + "┤")

    # Indicadores clave
    rsi_color = Fore.RED if tech['rsi_overbought'] else Fore.GREEN if tech['rsi_oversold'] else Fore.WHITE
    adx_color = Fore.GREEN if tech['adx_strong'] else Fore.YELLOW

    print(Fore.CYAN + "│" + f"  RSI: {rsi_color}{tech['rsi']:.1f}".ljust(w + 9) + Fore.CYAN + "│")
    print(Fore.CYAN + "│" + f"  ADX: {adx_color}{tech['adx']:.1f}  {Fore.WHITE}(+DI: {tech['plus_di']:.1f}  -DI: {tech['minus_di']:.1f})".ljust(w + 18) + Fore.CYAN + "│")
    print(Fore.CYAN + "│" + f"  ATR: {Fore.WHITE}{fmt(tech['atr'])} ({tech['atr_pct']:.2f}%)".ljust(w + 9) + Fore.CYAN + "│")
    print(Fore.CYAN + "│" + f"  VWAP: {Fore.WHITE}{fmt(tech['vwap'])}  {'↑' if tech['above_vwap'] else '↓'} precio".ljust(w + 9) + Fore.CYAN + "│")
    macd_labels = {'bullish_cross': '⬆ Cruce Alcista', 'bearish_cross': '⬇ Cruce Bajista', 'none': 'Sin cruce'}
    print(Fore.CYAN + "│" + f"  MACD: {Fore.WHITE}{macd_labels.get(tech['macd_cross'], 'Sin cruce')}  (Hist: {tech['macd_hist']:.4f})".ljust(w + 9) + Fore.CYAN + "│")

    bb_val = tech['bb_percent_b']
    if tech['bb_squeeze']:
        bb_display = "SQUEEZE ⚡"
    elif tech['bb_breakout'] != 'none':
        bb_display = tech['bb_breakout'].replace('_', ' ').title()
    else:
        bb_display = f"%B={bb_val:.2f}"
    print(Fore.CYAN + "│" + f"  Bollinger: {Fore.WHITE}{bb_display}".ljust(w + 9) + Fore.CYAN + "│")
    print(Fore.CYAN + "│" + f"  EMA Ribbon: {Fore.WHITE}{trend_es.get(tech['ema_alignment'], tech['ema_alignment'])}".ljust(w + 9) + Fore.CYAN + "│")

    print(Fore.CYAN + "│" + f"  Soporte: {Fore.GREEN}{fmt(tech['support'])}  {Fore.WHITE}Resistencia: {Fore.RED}{fmt(tech['resistance'])}".ljust(w + 27) + Fore.CYAN + "│")

    print(Fore.CYAN + "├" + "─" * w + "┤")

    # Score de confluencia
    print(Fore.CYAN + "│" + f"  Compra: {Fore.GREEN}{buy_bar} {signal.score_compra}/100".ljust(w + 9) + Fore.CYAN + "│")
    print(Fore.CYAN + "│" + f"  Venta:  {Fore.RED}{sell_bar} {signal.score_venta}/100".ljust(w + 9) + Fore.CYAN + "│")
    print(Fore.CYAN + "│" + f"  → {dir_color}{signal.resumen}".ljust(w + 9) + Fore.CYAN + "│")

    # Risk profile si hay alerta
    if risk_profile and signal.es_alerta:
        print(Fore.CYAN + "├" + "─" * w + "┤")
        print(Fore.CYAN + "│" + f"  {Fore.WHITE}Stop Loss: {Fore.RED}{fmt(risk_profile.stop_loss)}  {Fore.WHITE}TP1: {Fore.GREEN}{fmt(risk_profile.take_profit_1)}  {Fore.WHITE}TP2: {Fore.GREEN}{fmt(risk_profile.take_profit_2)}".ljust(w + 27) + Fore.CYAN + "│")
        print(Fore.CYAN + "│" + f"  {Fore.WHITE}R:R: {risk_profile.ratio_rr}:1  │  Posición: {risk_profile.posicion_sugerida} u  │  Riesgo: ${risk_profile.riesgo_total_usd:,.2f}".ljust(w + 9) + Fore.CYAN + "│")
        print(Fore.CYAN + "│" + f"  {Fore.WHITE}Kelly: {risk_profile.kelly_fraction:.2f}%  │  VaR(95%): ${risk_profile.var_95_1d:,.2f}  │  Fricción Est: ${risk_profile.simulated_commission + risk_profile.simulated_slippage:,.2f}".ljust(w + 9) + Fore.CYAN + "│")

    print(Fore.CYAN + Style.BRIGHT + "└" + "─" * w + "┘")


# ──────────────────────────────────────────────────────────────
# Alert Cooldown
# ──────────────────────────────────────────────────────────────

def is_alert_on_cooldown(state, alert_type):
    """Verifica si la alerta está en período de cooldown."""
    last_time = state.get("last_alert_time")
    last_type = state.get("last_alert_type")
    if last_time and last_type == alert_type:
        try:
            last_dt = datetime.fromisoformat(str(last_time))
            if datetime.now() - last_dt < timedelta(minutes=ALERT_COOLDOWN_MIN):
                return True
        except (ValueError, TypeError):
            pass
    return False


# ──────────────────────────────────────────────────────────────
# Bucle Principal
# ──────────────────────────────────────────────────────────────

def run_agent(tickers_info, risk_mgr, poll_interval=60, notifier=None, once=False):
    """Bucle principal del agente profesional."""
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    fetchers = {t: DataFetcher(t) for t in tickers_info}
    states = {t: load_state(t) for t in tickers_info}

    cycle = 0
    try:
        while True:
            cycle += 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if not once:
                print(Fore.CYAN + Style.DIM + f"\n{'─' * 74}")
                print(Fore.CYAN + Style.DIM + f"  Ciclo #{cycle}  │  {timestamp}")
                print(Fore.CYAN + Style.DIM + f"{'─' * 74}")

            for ticker, (model, scaler) in tickers_info.items():
                fetcher = fetchers[ticker]
                state = states[ticker]

                # ── Datos ──────────────────────────────────────────
                df = fetcher.fetch_latest_data(period="5d", interval="15m", last_n=100)
                if df.empty:
                    print(Fore.RED + f"  ✗ [{ticker}] Sin datos disponibles")
                    logger.warning(f"{ticker}: Sin datos en este ciclo")
                    continue

                current_close = float(df['Close'].iloc[-1])
                last_trend = state.get("last_trend", "neutral")

                # ── Datos Macro (MTF) ──────────────────────────────
                # Fetch 1d data for macro trend
                macro_df = fetcher.fetch_latest_data(period="6mo", interval="1d", last_n=50)
                macro_trend = 'neutral'
                if not macro_df.empty:
                    from technical import detect_trend
                    macro_trend = detect_trend(macro_df['Close'], short_ma=10, long_ma=50)

                # ── Análisis Técnico Completo ──────────────────────
                tech = full_technical_analysis(df)

                # ── Predicción IA (Con Datos Enriquecidos) ─────────
                # Construir DataFrame enriquecido como en train_model
                df['RSI'] = tech['rsi']
                df['MACD_Hist'] = tech['macd_hist']
                df['ATR'] = tech['atr']
                df['BB_pct'] = tech['bb_percent_b']
                df['ADX'] = tech['adx']
                
                tech_features = features + ['RSI', 'MACD_Hist', 'ATR', 'BB_pct', 'ADX']
                recent_data = df[tech_features].tail(SEQ_LENGTH).values
                
                if len(recent_data) < SEQ_LENGTH:
                    print(Fore.YELLOW + f"  ⏳ [{ticker}] Esperando datos ({len(recent_data)}/{SEQ_LENGTH})")
                    continue

                scaled = scaler.transform(recent_data).astype(np.float32)
                inp = torch.tensor(scaled).unsqueeze(0)
                model.eval()
                with torch.no_grad():
                    out = model(inp)
                    pred_idx = torch.argmax(out, dim=1).item()
                    confidence = torch.softmax(out, dim=1)[0][pred_idx].item()

                ai_trend = {0: 'bearish', 1: 'neutral', 2: 'bullish'}[pred_idx]

                # ── Motor de Señales ───────────────────────────────
                signal = evaluate_signal(tech, ai_trend, confidence, macro_trend=macro_trend)

                # ── Calcular Riesgo ────────────────────────────────
                risk_profile = None
                if signal.es_alerta:
                    risk_profile = risk_mgr.calculate_risk(
                        precio=current_close,
                        atr=tech['atr'],
                        direccion=signal.direccion,
                        soporte=tech['support'],
                        resistencia=tech['resistance'],
                        log_returns=tech.get('log_returns')
                    )

                    # Update win rate in risk manager based on state
                    wins = state.get("wins", 0)
                    losses = state.get("losses", 0)
                    total_trades = wins + losses
                    if total_trades > 10:
                        risk_mgr.win_rate = wins / total_trades

                # ── Dashboard ──────────────────────────────────────
                print_ticker_analysis(ticker, current_close, tech, signal, risk_profile)

                # ── Determinar tendencia combinada ─────────────────
                if signal.score_compra > signal.score_venta and signal.score_compra >= 40:
                    combined_trend = 'bullish'
                elif signal.score_venta > signal.score_compra and signal.score_venta >= 40:
                    combined_trend = 'bearish'
                else:
                    combined_trend = 'neutral'

                # ── Gestión Activa (Trailing Stop & Win Rate) ──────
                active_trade = state.get("active_trade")
                if active_trade:
                    # Update trailing stop
                    new_stop, updated = risk_mgr.update_trailing_stop(
                        current_close, active_trade["stop_loss"], active_trade["entry"],
                        active_trade["direction"], tech['atr']
                    )
                    
                    if updated:
                        active_trade["stop_loss"] = new_stop
                        logger.info(f"Trailing Stop {ticker} movido a {fmt(new_stop)}")
                        if notifier and notifier.bot_token:
                            notifier.send_message(f"🛡️ *TRAILING STOP* — {ticker}\nStop loss asegurado a: `${new_stop:,.2f}`")

                    # Check hit TP or SL
                    hit_tp = False
                    hit_sl = False
                    
                    if active_trade["direction"] == 'COMPRA':
                        if current_close >= active_trade["tp1"]: hit_tp = True
                        elif current_close <= active_trade["stop_loss"]: hit_sl = True
                    else: # VENTA
                        if current_close <= active_trade["tp1"]: hit_tp = True
                        elif current_close >= active_trade["stop_loss"]: hit_sl = True

                    if hit_tp or hit_sl:
                        outcome = "Ganancia (TP)" if hit_tp else "Pérdida (SL)"
                        emoji = "✅" if hit_tp else "❌"
                        state["wins"] = state.get("wins", 0) + (1 if hit_tp else 0)
                        state["losses"] = state.get("losses", 0) + (1 if hit_sl else 0)
                        
                        # Institutional PnL with commissions
                        pnl_gross = (active_trade["tp1"] - active_trade["entry"]) if hit_tp else (active_trade["stop_loss"] - active_trade["entry"])
                        if active_trade["direction"] == 'VENTA':
                            pnl_gross = -pnl_gross
                        
                        commission_cost = (current_close * risk_mgr.commission_rate) + (active_trade["entry"] * risk_mgr.commission_rate)
                        slippage_cost = tech['atr'] * 0.05
                        pnl_net = pnl_gross - commission_cost - slippage_cost
                        
                        logger.info(f"Trade cerrado: {ticker} -> {outcome} | Net PnL (1u): {pnl_net:.2f}", extra={
                            "json_data": {
                                "event": "trade_closed",
                                "ticker": ticker,
                                "outcome": outcome,
                                "pnl_net": pnl_net,
                                "hit_tp": hit_tp
                            }
                        })
                        if notifier and notifier.bot_token:
                            notifier.send_message(f"{emoji} *OPERACIÓN CERRADA* — {ticker}\nResultado: {outcome}\nPrecio de salida: `${current_close:,.2f}`\nNet PnL est (1u): `${pnl_net:,.2f}`")
                            
                        state["active_trade"] = None
                    else:
                        state["active_trade"] = active_trade

                # ── Alertas ────────────────────────────────────────
                if signal.es_alerta and not is_alert_on_cooldown(state, signal.direccion):
                    # Save active trade
                    if risk_profile:
                        state["active_trade"] = {
                            "direction": signal.direccion,
                            "entry": risk_profile.entrada,
                            "stop_loss": risk_profile.stop_loss,
                            "tp1": risk_profile.take_profit_1
                        }
                    
                    # Log detallado
                    logger.info(
                        f"ALERTA {signal.direccion} — {ticker} @ {fmt(current_close)} | "
                        f"Score: {max(signal.score_compra, signal.score_venta)}/100 | "
                        f"Fuerza: {signal.fuerza}",
                        extra={
                            "json_data": {
                                "event": "signal_alert",
                                "ticker": ticker,
                                "direction": signal.direccion,
                                "price": current_close,
                                "score": max(signal.score_compra, signal.score_venta),
                                "hurst": tech['hurst_exponent'],
                                "regime": tech['market_regime'],
                                "var_95": risk_profile.var_95_1d,
                                "kelly_fraction": risk_profile.kelly_fraction
                            }
                        }
                    )
                    for d in signal.detalles:
                        logger.info(f"  └─ {d.nombre}: +{d.puntos}/{d.max_puntos} — {d.razon}")

                    # Mostrar Win Rate si hay historial
                    wins = state.get("wins", 0)
                    losses = state.get("losses", 0)
                    total = wins + losses
                    wr_str = f"Win Rate: {wins/total*100:.1f}% ({wins}/{total})" if total > 0 else "Win Rate: N/A"

                    # Consola
                    alert_color = Fore.GREEN if signal.direccion == 'COMPRA' else Fore.RED
                    print()
                    print(alert_color + Style.BRIGHT + "  ╔══════════════════════════════════════════════════════════╗")
                    print(alert_color + Style.BRIGHT + f"  ║  ¡ALERTA {signal.direccion}!  │  {ticker}  │  {signal.fuerza}".ljust(61) + "║")
                    print(alert_color + Style.BRIGHT + f"  ║  {wr_str}".ljust(61) + "║")
                    print(alert_color + Style.BRIGHT + "  ╚══════════════════════════════════════════════════════════╝")

                    # Detalles confluencia
                    for d in signal.detalles:
                        print(f"    ✓ {d.nombre}: +{d.puntos} pts — {d.razon}")

                    # Telegram
                    if notifier and notifier.bot_token and risk_profile:
                        risk_text = format_risk_profile(risk_profile, signal.direccion)
                        detalles_text = "\n".join([f"  ✓ {d.nombre}: +{d.puntos} pts" for d in signal.detalles])
                        notifier.send_alert(
                            ticker=ticker,
                            direccion=signal.direccion,
                            score=max(signal.score_compra, signal.score_venta),
                            fuerza=signal.fuerza,
                            precio=current_close,
                            risk_text=risk_text,
                            detalles_text=detalles_text + f"\n\n📈 {wr_str}",
                        )

                    # Actualizar cooldown
                    state["last_alert_time"] = datetime.now().isoformat()
                    state["last_alert_type"] = signal.direccion

                    # Historial
                    history = state.get("alert_history", [])
                    history.append({
                        "time": datetime.now().isoformat(),
                        "type": signal.direccion,
                        "price": current_close,
                        "score": max(signal.score_compra, signal.score_venta),
                    })
                    state["alert_history"] = history[-50:]  # últimas 50

                # ── Notificar cambio de tendencia ──────────────────
                if last_trend != combined_trend and combined_trend != 'neutral' and last_trend != 'neutral':
                    trend_names = {'bullish': 'ALCISTA', 'bearish': 'BAJISTA'}
                    print(Fore.MAGENTA + Style.BRIGHT + f"\n  🔄 CAMBIO DE TENDENCIA: {trend_names.get(last_trend, last_trend)} → {trend_names.get(combined_trend, combined_trend)}")
                    logger.info(f"Cambio tendencia {ticker}: {last_trend} → {combined_trend}")

                    if notifier and notifier.bot_token:
                        notifier.send_trend_change(ticker, current_close, last_trend, combined_trend)

                # ── Guardar estado ─────────────────────────────────
                state["last_trend"] = combined_trend
                state["last_price"] = current_close
                states[ticker] = state
                save_state(ticker, state)

            if once:
                break
            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print(Fore.CYAN + "\n\n  ✓ Agente detenido. ¡Hasta pronto!")
        logger.info("Agente detenido por el usuario.")


# ──────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Agente de Trading Profesional — Confluencia + IA"
    )
    parser.add_argument("--tickers",   type=str, default="BTC-USD,GC=F",
                        help="Tickers separados por coma (default: BTC-USD,GC=F)")
    parser.add_argument("--interval",  type=int, default=60,
                        help="Intervalo de consulta en segundos (default: 60)")
    parser.add_argument("--capital",   type=float, default=10000.0,
                        help="Capital total para position sizing (default: 10000)")
    parser.add_argument("--riesgo",    type=float, default=1.0,
                        help="Porcentaje de riesgo por operación (default: 1.0)")
    parser.add_argument("--atr-mult",  type=float, default=1.5,
                        help="Multiplicador ATR para stop loss (default: 1.5)")
    parser.add_argument("--retrain",   action="store_true",
                        help="Forzar re-entrenamiento del modelo")
    parser.add_argument("--tg-token",  type=str, default=None,
                        help="Telegram Bot Token")
    parser.add_argument("--tg-chat",   type=str, default=None,
                        help="Telegram Chat ID")
    parser.add_argument("--once",      action="store_true",
                        help="Ejecutar una sola vez y salir")
    args = parser.parse_args()

    # Telegram
    tg_token = args.tg_token or os.getenv("TELEGRAM_BOT_TOKEN")
    tg_chat = args.tg_chat or os.getenv("TELEGRAM_CHAT_ID")
    notifier = TelegramNotifier(bot_token=tg_token, chat_id=tg_chat)

    # Risk Manager
    risk_mgr = RiskManager(capital=args.capital, riesgo_pct=args.riesgo, atr_multiplier=args.atr_mult)

    # Inicializar tickers
    ticker_list = [t.strip() for t in args.tickers.split(",")]
    tickers_info = {}

    print(Fore.CYAN + Style.BRIGHT + "\n  Inicializando activos...")
    print(Fore.CYAN + "  " + "─" * 40)

    for ticker in ticker_list:
        print(Fore.YELLOW + f"  → {ticker}...", end=" ")
        fetcher = DataFetcher(ticker)
        if not fetcher.verify_ticker():
            print(Fore.RED + "✗ inválido")
            continue

        model, scaler = None, None
        if not args.retrain:
            model, scaler = load_model(ticker)

        if model is None:
            print()
            model, scaler = train_model(ticker, period="60d", interval="1h")
            if model is None:
                print(Fore.RED + f"  ✗ Error al entrenar {ticker}")
                continue
        else:
            print(Fore.GREEN + "✓")

        tickers_info[ticker] = (model, scaler)

    if not tickers_info:
        print(Fore.RED + "\n  ✗ No hay activos válidos. Saliendo.")
        exit(1)

    # Header
    if not args.once:
        print_header(tickers_info, args.interval, args.capital, notifier)
        if notifier and notifier.bot_token:
            notifier.send_message(
                f"🚀 *Agente de Trading Iniciado*\n"
                f"Activos: `{', '.join(tickers_info.keys())}`\n"
                f"Capital: `${args.capital:,.2f}`\n"
                f"Riesgo: `{args.riesgo}%` por operación\n"
                f"Intervalo: `{args.interval}s`"
            )

    logger.info(f"Agente iniciado: {', '.join(tickers_info.keys())} | Capital: ${args.capital:,.2f}")

    # ── Ejecutar ───────────────────────────────────────────
    run_agent(
        tickers_info,
        risk_mgr,
        poll_interval=args.interval,
        notifier=notifier,
        once=args.once,
    )
