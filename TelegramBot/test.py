import telegram

api_key = '비공개'

# https://api.telegram.org/bot(여기에 괄호 포함해서 api_key입력)/getUpdates
bot = telegram.Bot(token=api_key)

# chat_id = bot.get_updates()[-1].message.chat_id
chat_id = 834438777

bot.sendMessage(chat_id=chat_id, text='test')
