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
from sklearn.model_selection import train_test_split


app = Flask(__name__)
PhantomEmoji(app) # Print Emoji

#################################################################################
# load classes
class_file = pd.read_csv("./model/keyword_and_answers.csv", encoding='utf-8')
classes = []
for i in range(len(class_file)):
    classes.append(class_file["answer"][i])
# get file
file_name = './model/train_data_new.csv'
key_and_ans = './model/keyword_and_answers.csv'
# translate
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

def initial_boost(epoch):
    if epoch==0: return float(8.0)
    elif epoch==1: return float(4.0)
    elif epoch==2: return float(2.0)
    elif epoch==3: return float(1.5)
    else: return float(1.0)

def step_cyclic(epoch):
    try:
        l_r, decay = 1.0, 0.0001
        if epoch%33==0:multiplier = 10
        else:multiplier = 1
        rate = float(multiplier * l_r * 1/(1 + decay * epoch))
        #print("Epoch",epoch+1,"- learning_rate",rate)
        return rate
    except Exception as e:
        print("Error in lr_schedule:",str(e))
        return float(1.0)

def train_def(file_name, NUM_LABELS):
    MAX_NB_WORDS = 500 # max no. of words for tokenizer
    MAX_SEQUENCE_LENGTH = 30 # max length of text (words) including padding
    VALIDATION_SPLIT = 0.2
    EMBEDDING_DIM = 200 # embedding dimensions for word vectors (word2vec/GloVe)

    texts, labels = [], []
    readCSV = pd.read_csv(open(file_name, encoding='utf-8'))
    for i in range(len(readCSV)):
        texts.append(readCSV["text"][i])
        labels.append(readCSV["label"][i])
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    with open('./model/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./model/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print('[i] Found %s unique tokens.' % len(word_index))
    data_int = pad_sequences(sequences, padding='pre', maxlen=(MAX_SEQUENCE_LENGTH-5))
    data = pad_sequences(data_int, padding='post', maxlen=(MAX_SEQUENCE_LENGTH))
    labels = to_categorical(np.asarray(labels)) # convert to one-hot encoding vectors
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
    x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.3, random_state=42)
    embeddings_index = {}
    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    
    embedding_matrix_ns = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix_ns[i] = embedding_vector
    
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedding_layer_frozen = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    embedded_sequences_frozen = embedding_layer_frozen(sequence_input)
    embedding_layer_train = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix_ns],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
    embedded_sequences_train = embedding_layer_train(sequence_input)
    l_lstm1f = Bidirectional(LSTM(6,return_sequences=True,dropout=0.3, recurrent_dropout=0.0))(embedded_sequences_frozen)
    l_lstm1t = Bidirectional(LSTM(6,return_sequences=True,dropout=0.3, recurrent_dropout=0.0))(embedded_sequences_train)
    l_lstm1 = Concatenate(axis=1)([l_lstm1f, l_lstm1t])
    l_conv_2 = Conv1D(filters=24,kernel_size=2,activation='relu')(l_lstm1)
    l_conv_2 = Dropout(0.3)(l_conv_2)
    l_conv_3 = Conv1D(filters=24,kernel_size=3,activation='relu')(l_lstm1)
    l_conv_3 = Dropout(0.3)(l_conv_3)
    l_conv_5 = Conv1D(filters=24,kernel_size=5,activation='relu',)(l_lstm1)
    l_conv_5 = Dropout(0.3)(l_conv_5)
    l_conv_6 = Conv1D(filters=24,kernel_size=6,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(l_lstm1)
    l_conv_6 = Dropout(0.3)(l_conv_6)
    l_conv_8 = Conv1D(filters=24,kernel_size=8,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(l_lstm1)
    l_conv_8 = Dropout(0.3)(l_conv_8)
    conv_1 = [l_conv_6,l_conv_5, l_conv_8,l_conv_2,l_conv_3]
    l_lstm_c = Concatenate(axis=1)(conv_1)
    l_conv_4f = Conv1D(filters=12,kernel_size=4,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(embedded_sequences_frozen)
    l_conv_4f = Dropout(0.3)(l_conv_4f)
    l_conv_4t = Conv1D(filters=12,kernel_size=4,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(embedded_sequences_train)
    l_conv_4t = Dropout(0.3)(l_conv_4t)
    l_conv_3f = Conv1D(filters=12,kernel_size=3,activation='relu',)(embedded_sequences_frozen)
    l_conv_3f = Dropout(0.3)(l_conv_3f)
    l_conv_3t = Conv1D(filters=12,kernel_size=3,activation='relu',)(embedded_sequences_train)
    l_conv_3t = Dropout(0.3)(l_conv_3t)
    l_conv_2f = Conv1D(filters=12,kernel_size=2,activation='relu')(embedded_sequences_frozen)
    l_conv_2f = Dropout(0.3)(l_conv_2f)
    l_conv_2t = Conv1D(filters=12,kernel_size=2,activation='relu')(embedded_sequences_train)
    l_conv_2t = Dropout(0.3)(l_conv_2t)
    conv_2 = [l_conv_4f, l_conv_4t,l_conv_3f, l_conv_3t, l_conv_2f, l_conv_2t]
    l_merge_2 = Concatenate(axis=1)(conv_2)
    l_c_lstm = Bidirectional(LSTM(12,return_sequences=True,dropout=0.3, recurrent_dropout=0.0))(l_merge_2)
    l_merge = Concatenate(axis=1)([l_lstm_c, l_c_lstm])
    l_pool = MaxPooling1D(4)(l_merge)
    l_drop = Dropout(0.5)(l_pool)
    l_flat = Flatten()(l_drop)
    l_dense = Dense(26, activation='relu')(l_flat)
    preds = Dense(NUM_LABELS, activation='softmax')(l_dense)
    model = Model(sequence_input, preds)
    adadelta = optimizers.Adadelta(lr=0.9, rho=0.95, epsilon=None, decay=0.002)
    lr_metric = get_lr_metric(adadelta)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adadelta,
                  metrics=['acc'])
    tensorboard = callbacks.TensorBoard(log_dir='./model/logs', histogram_freq=0, batch_size=16, write_grads=True , write_graph=True)
    model_checkpoints = callbacks.ModelCheckpoint("./model/checkpoint-0.91.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=0)
    lr_schedule = callbacks.LearningRateScheduler(initial_boost)
    model_log = model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=200, batch_size=128,
              callbacks=[tensorboard, model_checkpoints])
    model.save('./model/train_assis_model.h5')

def make_datas(word):
    made = []
    sentences = ["%s 관련 정보는 어디서 얻나요?",
                 "%s에 대한 자료가 궁금해요.",
                 "%s에 대한 자료는 어디서 찾을 수 있나요.",
                 "%s가 뭔가요?",
                 "%s 무슨 내용이죠",
                 "%s 하고싶은데 알려주세요.",
                 "%s 하려면 어떻게 해야하죠.",
                 "%s 알려줘",
                 "%s 자료를 찾기가 어려워요",
                 "%s에 대하여 알고싶어요.",
                 "질문이 있는데 %s 가 뭔지 모르겠습니다.",
                 "%s 알려주세요,",
                 "%s 하려면 어떤 걸 해야하죠",
                 "%s가 궁금합니다.",
                 "%s에 대해 알고 싶습니다.",
                 "%s 종류들이 뭐가 있나요",
                 "%s 무엇을 준비해야하나요.",
                 "%s 어떻게 하는거야?",
                 "학교 홈페이지에서 %s를 찾으려면 어디로 가야하나요.",
                 "%s 어디서 알 수 있어?",
                 "%s 하고 싶어요",
                 "%s 하고싶어", 
                 "학교 페이지에서 %s에 대한 정보는 어디에 있나요.",
                 "%s가 무엇인가요",
                 "%s가 뭐야?",
                 "준비하고 있는 %s에 대한 자료를 알고 싶어요",
                 "%s 어떻게 해?",
                 "%s 어떻게 하나요?",
                 "%s 정보는 어디서 알 수 있나요?",
                 "%s 관련 정보는 어디있어?",
                 "%s가 알고싶습니다.",
                 "%s 하려면 어떻게 해야 하나요?"   
    ]

    
    for i in range(len(sentences)):
        made.append(sentences[i] %(word))
    return made

def add_data(key, answer):
    train_datas = pd.read_csv(file_name)
    k_and_a = pd.read_csv(key_and_ans)
    k_and_a = k_and_a.append({'k':key, 'answer':answer}, ignore_index=True)
    label_num = len(k_and_a) - 1
    sentences = make_datas(key)
    for i in range(len(sentences)):
        train_datas = train_datas.append({'text':sentences[i], 'label':str(label_num)}, ignore_index=True)
    train_datas.to_csv(file_name, index=None, encoding = 'utf-8')
    k_and_a.to_csv(key_and_ans, index=None, encoding = 'utf-8')
    
    NUM_LABELS = len(k_and_a)
    
    train_def(file_name, NUM_LABELS)



###############################################################################
@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        result1 = request.form['result1']
        result2 = request.form['result2']

    sentence = "__label__1"

    add_data(result1, result2)
    temp = '입력해주신 데이터를 바탕으로 학습을 완료하였습니다.'
    return render_template('index.html', temp=temp)

if __name__ == '__main__':
    app.run(debug=True)