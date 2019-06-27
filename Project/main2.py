from telegram.ext import Updater, MessageHandler, Filters
from emoji import emojize
import numpy as np
import pandas as pd
import re, sys, os, csv, keras, pickle
from keras import regularizers, initializers, optimizers, callbacks
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten, Concatenate
from keras.layers import Conv1D, MaxPooling1D, Embedding, Add, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
import json
import re
import requests
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools, pickle
import os
import sys
import urllib.request
from konlpy.tag import Twitter
from keras import regularizers, initializers, optimizers, callbacks
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten, Concatenate
from keras.layers import Conv1D, MaxPooling1D, Embedding, Add, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
import pandas as pd
import konlpy
from konlpy.tag import Twitter

MAX_NB_WORDS = 500 # max no. of words for tokenizer
MAX_SEQUENCE_LENGTH = 30 # max length of text (words) including padding
VALIDATION_SPLIT = 0.2
EMBEDDING_DIM = 200 # embedding dimensions for word vectors (word2vec/GloVe)

def spellchecker(sentense):
    url = "https://m.search.naver.com/p/csearch/ocontent/util/SpellerProxy?_callback=mycallback&q=" + sentense + "&where=nexearch&color_blindness=0&_=1559640144299"

    response = requests.get(url).text
    json_string = response.replace('mycallback(', '').replace(');', '')

    result = json.loads(json_string, encoding='UTF-8')
    result = result['message']['result']['html']
    return result

class_file = pd.read_csv("./model/keyword_and_answers.csv", encoding='utf-8')
classes = []
for i in range(len(class_file)):
    classes.append(class_file["answer"][i])
    
with open('./model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
def detect_language(text):
    client_id = "mdPnME0GxjrCKi50UrgM"
    client_secret = "Jrob_59Asc"
    encQuery = urllib.parse.quote(text)
    data = "query=" + encQuery
    url = "https://openapi.naver.com/v1/papago/detectLangs"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id",client_id)
    request.add_header("X-Naver-Client-Secret",client_secret)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()
    if(rescode==200):
        response_body = response.read()
        result = response_body.decode('utf-8')
        return result[result.find(":")+2 : -2]
    else:
        return ("Error Code:" + rescode)
    
def translater(text, tar = 'ko'):
    client_id = "mdPnME0GxjrCKi50UrgM"
    client_secret = "Jrob_59Asc"
    encText = urllib.parse.quote(text)
    sor = detect_language(text)
    # print(sor)
    forma = "source="+sor+"&target="+tar+"&text="
    data = forma + encText
    url = "https://openapi.naver.com/v1/papago/n2mt"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id",client_id)
    request.add_header("X-Naver-Client-Secret",client_secret)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()
    if(rescode==200):
        response_body = response.read()
        result = response_body.decode('utf-8')
        result = result[result.find("translatedText") + 17:-5]
        return result
    else:
        return "Error Code:" + rescode

def classify(text):
    if detect_language(text) != 'ko':
        language = detect_language(text)
        text = translater(text)
    else:
        language = 'ko'   
    text = spellchecker(text)
    word = []
    word.append(text)
    tokenizer = Twitter()
    word = [tokenizer.morphs(row) for row in word]
    with open('./model/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    sequences_test = tokenizer.texts_to_sequences(word)
    data_int_t = pad_sequences(sequences_test, padding='pre', maxlen=(MAX_SEQUENCE_LENGTH-5))
    data_test = pad_sequences(data_int_t, padding='post', maxlen=(MAX_SEQUENCE_LENGTH))
    model = load_model('./model/train_model.h5')
    y_prob = model.predict(data_test)
    for n, prediction in enumerate(y_prob):
        pred = y_prob.argmax(axis=-1)[n]
        if pred < 2.0:
            return ("질문을 이해하지 못했어요. 다시 입력해주세요.")
        else:
            if language == 'ko':
                return (classes[pred])
            else:
                return (translater(classes[pred], language))



updater = Updater(token='비공개')
dispatcher = updater.dispatcher
updater.start_polling()

def handler(bot, update):
  text = update.message.text
  chat_id = update.message.chat_id

  result = classify(text)
  
  bot.send_message(chat_id=chat_id, text=result)
  
echo_handler = MessageHandler(Filters.text, handler)
dispatcher.add_handler(echo_handler)
