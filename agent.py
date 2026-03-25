import time
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from colorama import init, Fore, Style
from dotenv import load_dotenv

from data_api import DataFetcher
from technical import identify_support_resistance, detect_trend
from transformer import TimeSeriesTransformer, create_sequences
from telegram_notifier import TelegramNotifier

# Initialize colorama for cross-platform colored output
init(autoreset=True)

# Load .env file if present (for Telegram credentials)
load_dotenv()

SEQ_LENGTH = 30  # bars used per sequence (works for both 1h and 5m data)
INPUT_DIM = 5
D_MODEL = 32
NHEAD = 4
NUM_LAYERS = 2

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def fmt_price(val):
    """Format a price, returns 'N/A' if infinite."""
    return f"${val:.2f}" if val not in (float('inf'), float('-inf')) else "N/A"


def get_model_paths(ticker):
    # Sanitize ticker name for filename
    clean_ticker = ticker.replace("=", "").replace("-", "_")
    return f"{clean_ticker}_model.pt", f"{clean_ticker}_scaler.npz"


def save_model(ticker, model, scaler):
    m_path, s_path = get_model_paths(ticker)
    torch.save(model.state_dict(), m_path)
    np.savez(s_path, mean=scaler.mean_, scale=scaler.scale_)
    print(Fore.CYAN + f"[*] Model & scaler saved to {m_path} / {s_path}")


def load_model(ticker):
    """Try to load a pre-trained model and scaler from disk."""
    m_path, s_path = get_model_paths(ticker)
    if not (os.path.exists(m_path) and os.path.exists(s_path)):
        return None, None
    print(Fore.CYAN + f"[*] Found saved model for {ticker}. Loading from disk...")
    model = TimeSeriesTransformer(input_dim=INPUT_DIM, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS)
    model.load_state_dict(torch.load(m_path, weights_only=True))
    model.eval()

    data = np.load(s_path)
    scaler = StandardScaler()
    scaler.mean_ = data['mean']
    scaler.scale_ = data['scale']
    scaler.n_features_in_ = INPUT_DIM
    print(Fore.GREEN + f"[*] Model for {ticker} loaded successfully.\n")
    return model, scaler


# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────

def train_model(ticker, period="60d", interval="5m"):
    """Fetches historical data and trains the Transformer model."""
    # Yahoo Finance limits: 5m -> max 5d, 1h -> max 60d, 1d -> max years.
    # We use 1h/60d for training to get enough history.
    print(Fore.YELLOW + f"\n[*] Fetching historical data for {ticker}  (period={period}, interval={interval})...")
    fetcher = DataFetcher(ticker)
    df = fetcher.fetch_historical_data(period=period, interval=interval)
    if df.empty:
        print(Fore.RED + "[!] No data received from Yahoo Finance.")
        return None, None

    print(Fore.YELLOW + f"[*] Fetched {len(df)} rows. Preparing sequences...")
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features].values)

    X, y = create_sequences(scaled_data, seq_length=SEQ_LENGTH)
    if len(X) == 0:
        print(Fore.RED + "[!] Not enough data to create sequences.")
        return None, None

    print(Fore.YELLOW + f"[*] Created {len(X)} sequences. Starting training...")
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Class balance info
    unique, counts = np.unique(y, return_counts=True)
    label_names = {0: "Bearish", 1: "Neutral", 2: "Bullish"}
    for u, c in zip(unique, counts):
        print(f"    {label_names[u]}: {c} samples ({100*c/len(y):.1f}%)")

    model = TimeSeriesTransformer(input_dim=INPUT_DIM, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    epochs = 15
    batch_size = 64
    best_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        for i in range(0, len(X_tensor), batch_size):
            batch_X = X_tensor[i:i + batch_size]
            batch_y = y_tensor[i:i + batch_size]
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        scheduler.step()
        print(f"  Epoch {epoch+1:02d}/{epochs}  |  Loss: {avg_loss:.4f}  |  LR: {scheduler.get_last_lr()[0]:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)

    print(Fore.GREEN + f"\n[*] Training complete. Best loss: {best_loss:.4f}\n")
    save_model(ticker, model, scaler)
    return model, scaler


# ──────────────────────────────────────────────────────────────
# Real-Time Agent Loop
# ──────────────────────────────────────────────────────────────

def run_realtime_agent(ticker, model, scaler, poll_interval=60, telegram_notifier=None):
    """Main real-time monitoring loop."""
    fetcher = DataFetcher(ticker)
    features = ['Open', 'High', 'Low', 'Close', 'Volume']

    last_trend = 'neutral'
    last_price = 0.0

    print(Fore.CYAN + Style.BRIGHT + f"\n{'='*58}")
    print(Fore.CYAN + Style.BRIGHT + f"  Real-Time Market Monitor  |  Ticker: {ticker}")
    print(Fore.CYAN + Style.BRIGHT + f"{'='*58}")
    print(f"  Poll interval : {poll_interval}s")
    print(f"  Telegram alerts: {'Enabled' if (telegram_notifier and telegram_notifier.bot_token) else 'Disabled'}")
    print(Fore.CYAN + Style.BRIGHT + f"{'='*58}\n")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            # ── Fetch latest bars ──────────────────────────────
            # 5m interval is only available for the last 5 days — perfect for real-time monitoring
            df = fetcher.fetch_latest_data(period="5d", interval="5m", last_n=100)
            if df.empty:
                print(Fore.RED + "[!] Failed to fetch recent data. Retrying...")
                time.sleep(poll_interval)
                continue

            current_time = df['Datetime'].iloc[-1]
            current_close = float(df['Close'].iloc[-1])

            # ── Technical Analysis ─────────────────────────────
            supports, resistances = identify_support_resistance(df['Close'], window=5)
            below_supports = [s for s in supports if s < current_close]
            above_resistances = [r for r in resistances if r > current_close]

            closest_supp = max(below_supports) if below_supports else 0.0
            closest_res = min(above_resistances) if above_resistances else float('inf')

            tech_trend = detect_trend(df['Close'], short_ma=5, long_ma=20)

            # ── Transformer Prediction ─────────────────────────
            recent_data = df[features].tail(SEQ_LENGTH).values
            if len(recent_data) < SEQ_LENGTH:
                print(Fore.YELLOW + f"[{current_time}] Waiting for more data ({len(recent_data)}/{SEQ_LENGTH} bars)...")
                time.sleep(poll_interval)
                continue

            scaled_recent = scaler.transform(recent_data).astype(np.float32)
            input_tensor = torch.tensor(scaled_recent).unsqueeze(0)  # (1, SEQ_LENGTH, 5)

            model.eval()
            with torch.no_grad():
                output = model(input_tensor)
                pred_idx = torch.argmax(output, dim=1).item()
                confidence = torch.softmax(output, dim=1)[0][pred_idx].item()

            pred_mapping = {0: 'bearish', 1: 'neutral', 2: 'bullish'}
            model_trend = pred_mapping[pred_idx]

            # ── Combined Signal ────────────────────────────────
            # Bullish only if BOTH tech MA and model agree (or model is neutral/bullish when tech is bullish)
            if tech_trend == 'bullish' and model_trend != 'bearish':
                combined_trend = 'bullish'
            elif tech_trend == 'bearish' and model_trend != 'bullish':
                combined_trend = 'bearish'
            else:
                combined_trend = model_trend  # model as tiebreaker

            # ── Status Print ───────────────────────────────────
            trend_color = Fore.GREEN if combined_trend == 'bullish' else Fore.RED if combined_trend == 'bearish' else Fore.YELLOW
            print(f"\n--- Update @ {current_time} ---")
            print(f"  Price        : {Fore.WHITE}{fmt_price(current_close)}")
            print(f"  Support      : {Fore.GREEN}{fmt_price(closest_supp)}   "
                  f"Resistance: {Fore.RED}{fmt_price(closest_res)}")
            print(f"  Tech Trend   : {trend_color}{tech_trend.upper()}")
            print(f"  AI Prediction: {trend_color}{model_trend.upper()} (confidence: {confidence:.1%})")
            print(f"  Combined     : {trend_color}{combined_trend.upper()}")

            # ── Alert Logic ────────────────────────────────────
            if last_trend == 'bearish' and combined_trend == 'bullish':
                msg = (
                    f"🟢 *[ALERT] BULLISH REVERSAL — {ticker}*\n"
                    f"Price: {fmt_price(current_close)}\n"
                    f"AI confidence: {confidence:.1%}\n"
                )
                if closest_supp > 0:
                    msg += f"Bounced off support at {fmt_price(closest_supp)}\n"
                if closest_res < float('inf'):
                    msg += (
                        f"━━━━━━━━━━━━━━━━━\n"
                        f"✅ *ACTION*: BUY {ticker}\n"
                        f"🛑 *STOP LOSS*: {fmt_price(closest_supp * 0.995)}\n"
                        f"🎯 *TAKE PROFIT*: {fmt_price(closest_res)}"
                    )
                else:
                    msg += f"━━━━━━━━━━━━━━━━━\n✅ *ACTION*: BUY {ticker}\n🛑 *STOP LOSS*: {fmt_price(closest_supp * 0.995)}"

                print(Fore.GREEN + Style.BRIGHT + "\n" + "=" * 58)
                print(Fore.GREEN + Style.BRIGHT + msg.replace("*", ""))
                print(Fore.GREEN + Style.BRIGHT + "=" * 58)
                if telegram_notifier:
                    telegram_notifier.send_message(msg)

            elif last_trend == 'bullish' and combined_trend == 'bearish':
                msg = (
                    f"🔴 *[ALERT] BEARISH REVERSAL — {ticker}*\n"
                    f"Price: {fmt_price(current_close)}\n"
                    f"AI confidence: {confidence:.1%}\n"
                    f"━━━━━━━━━━━━━━━━━\n"
                    f"⚠️ *ACTION*: SELL / TAKE PROFITS"
                )
                print(Fore.RED + Style.BRIGHT + "\n" + "=" * 58)
                print(Fore.RED + Style.BRIGHT + msg.replace("*", ""))
                print(Fore.RED + Style.BRIGHT + "=" * 58)
                if telegram_notifier:
                    telegram_notifier.send_message(msg)

            else:
                print(f"  Status       : No structural change. Holding {trend_color}{combined_trend.upper()}{'':>5}")

            # Update state
            last_trend = combined_trend
            last_price = current_close

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print(Fore.CYAN + "\n\n[*] Agent stopped by user. Goodbye!")


# ──────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time Market Monitor Agent — Transformer + Technical Analysis"
    )
    parser.add_argument("--ticker",    type=str, default="AAPL",
                        help="Ticker symbol to monitor (default: AAPL)")
    parser.add_argument("--interval",  type=int, default=60,
                        help="Polling interval in seconds (default: 60)")
    parser.add_argument("--retrain",   action="store_true",
                        help="Force re-training even if a saved model exists")
    parser.add_argument("--tg-token",  type=str, default=None,
                        help="Telegram Bot Token (overrides .env)")
    parser.add_argument("--tg-chat",   type=str, default=None,
                        help="Telegram Chat ID (overrides .env)")
    args = parser.parse_args()

    # Telegram credentials: CLI arg > .env file
    tg_token = args.tg_token or os.getenv("TELEGRAM_BOT_TOKEN")
    tg_chat  = args.tg_chat  or os.getenv("TELEGRAM_CHAT_ID")
    notifier = TelegramNotifier(bot_token=tg_token, chat_id=tg_chat)

    # ── Ticker Verification ────────────────────────────────
    fetcher = DataFetcher(args.ticker)
    if not fetcher.verify_ticker():
        print(Fore.RED + f"[!] Ticker {args.ticker} seems invalid or has no data. Please check.")
        exit(1)

    # ── Model Initialization ────────────────────────────────
    model, scaler = None, None
    if not args.retrain:
        model, scaler = load_model(args.ticker)

    if model is None:
        # Train on 60d of hourly data (1h is the finest resolution yfinance allows over 60d)
        model, scaler = train_model(args.ticker, period="60d", interval="1h")
        if model is None:
            print(Fore.RED + "[!] Model initialization failed. Exiting.")
            exit(1)
    
    # Send startup notification
    if notifier and notifier.bot_token:
        notifier.send_message(f"🚀 *Market Monitor Started*\nTicker: `{args.ticker}`\nInterval: `{args.interval}s`\nStatus: Monitoring for reversals...")

    # ── Start Agent ─────────────────────────────────────────
    run_realtime_agent(
        args.ticker,
        model,
        scaler,
        poll_interval=args.interval,
        telegram_notifier=notifier,
    )
