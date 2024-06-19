import os
import requests
from dotenv import load_dotenv

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Variables
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def telegram_notification(MESSAGE):
    # sending notification to Telegram
    response = requests.post(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
        data={'chat_id': CHAT_ID, 'text': MESSAGE}
    )

    if response.status_code == 200:
        print("message sent")
    else:
        print("Error sending message.")
