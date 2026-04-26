"""
telegram_notifier.py — Notificaciones Telegram Profesionales
==============================================================
Soporta Markdown y HTML, con formato estructurado para alertas de trading.
"""

import requests
from datetime import datetime


class TelegramNotifier:
    def __init__(self, bot_token=None, chat_id=None):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage" if self.bot_token else None

    def send_message(self, text, parse_mode="Markdown"):
        """Envía un mensaje de texto con formato Markdown o HTML."""
        if not self.bot_token or not self.chat_id:
            return False

        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }

        try:
            response = requests.post(self.base_url, json=payload, timeout=10)
            if response.status_code == 200:
                return True
            else:
                print(f"[!] Telegram error: {response.status_code} — {response.text[:100]}")
                return False
        except Exception as e:
            print(f"[!] Telegram excepción: {e}")
            return False

    def send_alert(self, ticker, direccion, score, fuerza, precio, risk_text, detalles_text):
        """Envía una alerta de trading estructurada."""
        emoji = "🟢" if direccion == "COMPRA" else "🔴" if direccion == "VENTA" else "🟡"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        msg = (
            f"{emoji} *[{direccion}] — {ticker}*\n"
            f"⏰ {timestamp}\n"
            f"💲 Precio: ${precio:,.2f}\n"
            f"📊 Score: {score}/100 — {fuerza}\n"
            f"\n"
            f"{risk_text}\n"
            f"\n"
            f"📋 *CONFLUENCIA:*\n"
            f"{detalles_text}"
        )
        return self.send_message(msg)

    def send_trend_change(self, ticker, precio, old_trend, new_trend):
        """Envía notificación de cambio de tendencia."""
        trend_map = {'bullish': '📈 ALCISTA', 'bearish': '📉 BAJISTA', 'neutral': '➡️ NEUTRAL'}
        msg = (
            f"🔄 *CAMBIO DE TENDENCIA — {ticker}*\n"
            f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"💲 Precio: ${precio:,.2f}\n"
            f"De: {trend_map.get(old_trend, old_trend)}\n"
            f"A: {trend_map.get(new_trend, new_trend)}"
        )
        return self.send_message(msg)

    def send_summary(self, summary_text):
        """Envía resumen periódico."""
        msg = (
            f"📋 *RESUMEN DEL MONITOR*\n"
            f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"{summary_text}"
        )
        return self.send_message(msg)
