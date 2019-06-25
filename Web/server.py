# -*- coding: utf-8 -*-
import requests
from flask import Flask, render_template, request
import nltk
from nltk.stem.lancaster import LancasterStemmer
import os
import json
import datetime
import pandas as pd
from soynlp.normalizer import *
import numpy as np
import time
import re
from googletrans import Translator
import json
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
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import itertools, pickle
from keras.models import model_from_json 
from flask_phantom_emoji import PhantomEmoji

app = Flask(__name__)
PhantomEmoji(app)

#################################################################################
# load classes
classes = ["2019년 입학자 기준 중핵 필수는 14학점 이상, 중핵필수선택은 21학점 이상, 전공기초교양은 9학점 이상, 전공 필수 40학점 이상, 전공 선택은 42학점 이상 들으셔야합니다.자세한 사항은 수강 편람을 참고해주세요.", "수강 정정 기간은 3월 6일부터 3월 13일까지입니다. 수강 정정은 학사정보시스템에서 하실 수 있습니다.", "컴퓨터공학과, 정보보호학과, 데이터 사이언스학과에서 개설된 전공과목을 수강할 경우 이를 모두 전공 선택으로 인정합니다. 이외에도 전공으로 인정되는 과목은 아래와 같으며 아래 과목은 합산 최대 6학점 까지만 인정됩니다.- 정보보호학과: 시스템관리 및 보안, 보안프로그래밍- 엔터네인먼트소프트웨어: ES-엔터테인먼트미디어프로젝트- 소셜미디어매니지먼트 소프트웨어: SM-소셜미디어프로젝트- 지능기전공학부(무인이동체공학,스마트기기공학): 창의융합노마드-> 상기한 7개 과목은 타학과 교과목이지만 우리과 과목으로 인정이 됩니다.(최대 6학점 까지만)", "수강신청은 학사정보시스템에 로그인 후 좌측에 수업/성적>강좌조회 및 수강신청>수강신청 에서 하실 수 있습니다. 학사정보시스템 링크를 알려드릴께요~ http://uis.sejong.ac.kr/app/", "수강 신청 기간은 2월 18(월) ~ 21(목) 입니다. 4학년은 18(월), 3학년은 19(화), 2학년은 20(수), 1학년은 21(목)일 입니다.수강 신청 관련해서 http://uis.sejong.ac.kr/app/sys.Login.servj 에서 확인해 볼 수 있습니다.", "장학금 제도에 관해서 링크를 참고해 주세요! http://www.sejong.ac.kr/unilife/campusscholarship.html?menu_id=4.2", "각 학기에 신청해야할 과목을 선택하시는 http://sejong.ac.kr/unilife/subject_2018.html?menu_id=1.2 이 곳이 도움이 될 것 같습니다.", "수강 신청 하시는 법에 대해서는 http://sejong.ac.kr/unilife/study_02.html?menu_id=1.3 에 자세히 나와있으니 참고 바랍니다.", "서양의 역사와 사상 4권, 동양의 역사와 사상 2권, 동.서양의 문학 3권, 과학 사상 1권, 최소 총 10권의 책을 인증 받으셔야 졸업요건을 충족할 수 있어요., 더 자세한 내용은 대향휴머니티칼리지를 참조해 주세요.https://classic.sejong.ac.kr/", "소프트웨어 융합대학 교수님 정보에 관련해서 다음 주소해서 확인할 수 있어요! http://home.sejong.ac.kr/~digitdpt/2.html"]
# translate
def spellchecker(sentense):
    url = "https://m.search.naver.com/p/csearch/ocontent/util/SpellerProxy?_callback=mycallback&q=" + sentense + "&where=nexearch&color_blindness=0&_=1559640144299"

    response = requests.get(url).text
    json_string = response.replace('mycallback(', '').replace(');', '')

    result = json.loads(json_string, encoding='UTF-8')
    result = result['message']['result']['html']
    return result


def classify(text):
    text = spellchecker(text)
    word = []
    with open('./model/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    word.append(text)
    sequences_test = tokenizer.texts_to_sequences(word)
    data_int_t = pad_sequences(sequences_test, padding='pre', maxlen=(MAX_SEQUENCE_LENGTH-5))
    data_test = pad_sequences(data_int_t, padding='post', maxlen=(MAX_SEQUENCE_LENGTH))
    model = load_model('./model/train_model.h5')
    y_prob = model.predict(data_test)
    for n, prediction in enumerate(y_prob):
        pred = y_prob.argmax(axis=-1)[n]
    
    return classes[pred]


MAX_NB_WORDS = 40000 # max no. of words for tokenizer
MAX_SEQUENCE_LENGTH = 30 # max length of text (words) including padding
VALIDATION_SPLIT = 0.2
EMBEDDING_DIM = 200 # embedding dimensions for word vectors (word2vec/GloVe)
###############################################################################
@app.route("/", methods=['GET', 'POST'])
def index():
    # ì ¼ë°˜ì  ìœ¼ë¡œ ì ‘ì†  
    if request.method == 'GET':
        return render_template('index.html')

    # ë °ì ´í„° ìž…ë ¥
    if request.method == 'POST':
        result = request.form['result']


    sentence = "__label__1"
    result = classify(result)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)