import requests

class TelegramNotifier:
    def __init__(self, bot_token=None, chat_id=None):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage" if self.bot_token else None

    def send_message(self, text):
        if not self.bot_token or not self.chat_id:
            print("[!] Telegram credentials not configured. Skipping alert.")
            return False
            
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }
        
        try:
            response = requests.post(self.base_url, json=payload, timeout=10)
            if response.status_code == 200:
                print("[*] Alert sent to Telegram.")
                return True
            else:
                print(f"[!] Failed to send alert to Telegram. Status: {response.status_code}, Msg: {response.text}")
                return False
        except Exception as e:
            print(f"[!] Error sending Telegram message: {e}")
            return False
