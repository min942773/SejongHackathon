import telegram

api_key = '889568811:AAGsWQ8wNGTJYuzcKvffoZ6TG6xTpuE7-8w'

# https://api.telegram.org/bot889568811:AAGsWQ8wNGTJYuzcKvffoZ6TG6xTpuE7-8w/getUpdates
bot = telegram.Bot(token=api_key)

# chat_id = bot.get_updates()[-1].message.chat_id
chat_id = 834438777

bot.sendMessage(chat_id=chat_id, text='test')