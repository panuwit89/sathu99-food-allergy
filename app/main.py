from flask import Flask,request,abort
import requests
from app.Config import *
import json
import pythainlp
from pythainlp.tokenize import word_tokenize
import pandas as pd
import numpy as np
import copy
import tensorflow as tf
from keras.utils import pad_sequences

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Dropout
from keras.utils import to_categorical

def tokenize(sentence) :
  return word_tokenize(sentence, engine="newmm")

raw_data = pd.read_csv("testdataset_new.csv", encoding='utf8')
answer = pd.read_csv("classanswer.csv")
data = np.array(raw_data.values)
trainToken = [tokenize(st) for st in data[:, 0]]
max_length = max([len(tk) for tk in trainToken])
y = raw_data['Answer'].values

all = set()
words = {}

for st in trainToken :
  for w in st :
    all.add(w)

for w in all :
  words[w] = len(words) + 1

tts = copy.deepcopy(trainToken)
for st in range(len(tts)) :
  for w in range(len(tts[st])) :
    if words.get(tts[st][w]) != None :
      tts[st][w] = words[tts[st][w]]
    else :
      tts[st][w] = 0

vocab_size = len(words) + 1

X = pad_sequences(tts, maxlen=max_length, padding="post")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=48)

y_train_encoded = to_categorical(y_train - 1, num_classes=10)
y_test_encoded = to_categorical(y_test - 1, num_classes=10)

model_DL = Sequential()
model_DL.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=max_length))
model_DL.add(Conv1D(filters=128, kernel_size=1, activation='relu'))
model_DL.add(Dropout(0.5))
model_DL.add(GlobalMaxPooling1D())
model_DL.add(Dense(10, activation='softmax'))

model_DL.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_DL.fit(X_train, y_train_encoded, epochs=16, batch_size=16, validation_split=0.1)

app=Flask(__name__)

@app.route('/webhook',methods=['POST','GET'])

def webhook():
    if request.method == 'POST':
        payload = request.json
        Reply_token = payload['events'][0]['replyToken']
        message = payload['events'][0]['message']['text']

        #tokenize
        test_text = word_tokenize(message)
        
        #word index
        for w in range(len(test_text)) :
            if words.get(test_text[w]) != None :
                test_text[w] = words[test_text[w]]
            else :
                test_text[w] = 0
        
        #padding
        pad_test_text = pad_sequences([test_text], maxlen=max_length, padding="post")
        
        #predict
        predictions = model_DL.predict(pad_test_text)
        predicted_labels = np.argmax(predictions, axis=1)
        
        #reply
        if (np.max(predictions)) >= 0.5 :
            label = predicted_labels[0]
            Reply_text = answer["Output"][label]
        else:
            Reply_text = "ขออภัย ฉันไม่สามารถเข้าใจคำถามของคุณได้ กรุณาถามใหม่"
        
        print(Reply_text,flush=True)
        ReplyMessage(Reply_token,Reply_text,Channel_access_token)
        return request.json,200
    elif request.method == 'GET':
        return "this is method GET!!!",200
    else:
        abort(400)

def ReplyMessage(Reply_token,TextMessage,Line_Acees_Token):
    LINE_API='https://api.line.me/v2/bot/message/reply/'

    Authorization='Bearer {}'.format(Line_Acees_Token)
    print(Authorization)
    headers={
        'Content-Type':'application/json; char=UTF-8',
        'Authorization':Authorization
    }

    data={
        "replyToken":Reply_token,
        "messages":[{
            "type":"text",
            "text":TextMessage
        }]
    }
    data=json.dumps(data) # ทำเป็น json
    r=requests.post(LINE_API,headers=headers,data=data)
    return 200