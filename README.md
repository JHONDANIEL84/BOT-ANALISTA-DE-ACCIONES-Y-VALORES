# 📈 Real-Time Market Monitor Agent

An AI-powered market monitoring agent that combines **Transformer-based sequence modeling** with **classical technical analysis** to detect trend reversals and issue actionable buy/sell signals in real time.

---

## Architecture

```
Yahoo Finance (yfinance)
        │
        ▼
  data_api.py          ─── Fetches OHLCV data
        │
  ┌─────┴─────────────────────────────┐
  │                                   │
  ▼                                   ▼
technical.py                   transformer.py
(Support/Resistance,            (TimeSeriesTransformer:
 Moving Average Trend)           Bearish / Neutral / Bullish)
  │                                   │
  └──────────────┬────────────────────┘
                 ▼
             agent.py          ─── Combined signal + Alert logic
                 │
                 ▼
        telegram_notifier.py  ─── Optional Telegram alerts
```

---

## Installation

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Configuration (Optional — Telegram Alerts)

Copy `.env.example` to `.env` and fill in your credentials:

```bash
copy .env.example .env
```

```
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

> Get a bot token from [@BotFather](https://t.me/botfather) and your chat ID from [@userinfobot](https://t.me/userinfobot).

---

## Usage

### Basic — Monitor AAPL every 60 seconds

```bash
python agent.py
```

### Custom ticker & poll interval

```bash
python agent.py --ticker MSFT --interval 30
```

### Force retraining even if a saved model exists

```bash
python agent.py --ticker TSLA --retrain
```

### With Telegram alerts (overrides .env)

```bash
python agent.py --ticker BTC-USD --tg-token YOUR_TOKEN --tg-chat YOUR_CHAT_ID
```

### All options

| Flag | Default | Description |
|------|---------|-------------|
| `--ticker` | `AAPL` | Stock/crypto ticker symbol |
| `--interval` | `60` | Polling interval in seconds |
| `--retrain` | False | Force model re-training |
| `--tg-token` | `.env` | Telegram Bot Token |
| `--tg-chat` | `.env` | Telegram Chat ID |

---

## How It Works

1. **Training** (first run): fetches 60 days of **hourly** OHLCV bars from Yahoo Finance, trains a Transformer to classify the next bar as `Bearish / Neutral / Bullish`. Model is saved to `model.pt` — subsequent runs load it instantly. Early training on hourly data provides a stable structural model that the agent then applies to 5-minute real-time data.

2. **Real-time loop** (every N seconds):
   - Fetches the last 100 bars
   - Detects **support & resistance** levels via local minima/maxima clustering
   - Computes **Moving Average trend** (5-bar vs 20-bar)
   - Runs **Transformer inference** on the last 30 bars
   - Combines both signals into a final `bullish / bearish / neutral` verdict
   - **Alerts** when the trend flips: Bearish → Bullish (BUY signal) or Bullish → Bearish (SELL signal), with suggested Stop Loss & Take Profit levels

---

## 24/7 Free Monitoring (GitHub Actions)

You can run this agent for free 24/7 using GitHub Actions. It is configured to run every 30 minutes, check the market, and alert you if the trend changes.

### Setup

1. **Push your code to GitHub**:
   ```bash
   git add .
   git commit -m "Add GitHub Actions automation"
   git push
   ```

2. **Configure Secrets**:
   Go to your GitHub Repository -> **Settings** -> **Secrets and variables** -> **Actions** -> **New repository secret**.
   Add the following secrets:
   - `TELEGRAM_BOT_TOKEN`: Your bot token.
   - `TELEGRAM_CHAT_ID`: Your chat ID.

3. **Enable Permissions**:
   Go to **Settings** -> **Actions** -> **General**.
   Under **Workflow permissions**, select **Read and write permissions** (required for the bot to save the market state back to the repo).

4. **Profit**:
   The bot will now run automatically. You can check the progress in the **Actions** tab of your repository.

---

## Project Structure

```
market_monitor_agent/
├── agent.py              # Main entry point & real-time loop
├── data_api.py           # Yahoo Finance data fetcher
├── technical.py          # Support/Resistance & trend detection
├── transformer.py        # Transformer model & sequence labeling
├── telegram_notifier.py  # Telegram alert sender
├── requirements.txt      # Python dependencies
├── .env.example          # Template for Telegram credentials
├── model.pt              # Saved model weights (auto-generated)
└── scaler.npz            # Saved scaler parameters (auto-generated)
```
