from telegram.ext import Updater, MessageHandler, Filters
from emoji import emojize

updater = Updater(token='889568811:AAGsWQ8wNGTJYuzcKvffoZ6TG6xTpuE7-8w')
dispatcher = updater.dispatcher
updater.start_polling()

def handler(bot, update):
  text = update.message.text
  chat_id = update.message.chat_id
  
  if '졸업요건' in text:
    bot.send_message(chat_id=chat_id, text='졸업요건 ')
  elif '수강신청' in text:
    bot.send_message(chat_id=chat_id, text=emojize('수강신청:heart_eyes:', use_aliases=True))
  elif '장학금' in text:
    bot.send_message(chat_id=chat_id, text='장학금')
  elif '고전강독' in text:
    bot.send_photo(chat_id=chat_id, photo=open('img/mj.jpg', 'rb'))
  else:
    bot.send_message(chat_id=chat_id, text='몰라')

echo_handler = MessageHandler(Filters.text, handler)
dispatcher.add_handler(echo_handler)