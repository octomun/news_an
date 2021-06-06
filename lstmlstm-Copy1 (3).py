#!/usr/bin/env python
# coding: utf-8

# In[71]:


#import JPype1
from konlpy.tag import Hannanum
from konlpy.utils import pprint
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
from datetime import datetime, timedelta
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split
import requests
import urllib.request
import urllib.parse
from urllib.request import urlopen
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.text import Tokenizer
import os
os.chdir('F:/news/newss')
# ## 주가 읽기


# In[89]:


def read_file():
    stock = pd.read_csv("sk하이닉스.csv")
    stock['date'] = pd.to_datetime(stock['date'], format ='%Y-%m-%d')
    for i in range(len(stock)):
        stock['per'][i]=(float(stock['per'][i].replace("%",""))/100)

    # # 뉴스 url읽기
    url = pd.read_csv("crawling_news2.txt", header = None, names = ["date","href"])
    url['per']= "A"
    return url, stock


# In[90]:


url, stock = read_file()
print("read_file")


# In[91]:


def lte_year(data,year):
    url_lte_2019=[]
    for i in range(len(data)):
        if int(url['date'].iloc[i][:2]) > int(str(pd.Timestamp.today())[2:4]):
            year = int(url['date'].iloc[i][:2]) + 1900
        else :
            year = int(url['date'].iloc[i][:2]) + 2000
        url['date'].iloc[i] = pd.Timestamp(year , int(url['date'].iloc[i][3:5]) , int(url['date'].iloc[i][6:8]))
        if pd.Timestamp(url['date'].iloc[i]) >= pd.Timestamp(year,1,1):
            url_lte_2019.append(i)
    data = data.iloc[url_lte_2019]
    return data


# In[92]:


int(url['date'].iloc[1][0:2])


# In[93]:


url = lte_year(url,2000)
# url = url.reset_index()


# In[94]:


url.to_csv('url.csv')


# In[95]:


url = pd.read_csv('url.csv')


# In[96]:


#전처리
def Pretreatment(url,stock):
    for i in range(len(url)):
        while url['date'].iloc[i] not in list(stock['date']):
            url['date'].iloc[i]=pd.to_datetime(url['date'].iloc[i], format ='%Y-%m-%d')+timedelta(days=1)
        for j in range(len(stock)):
            if url['date'].iloc[i] == stock['date'].iloc[j]:
                url['per'].iloc[i] = stock['per'].iloc[j-1] #다음날 주가

    te4 = pd.DataFrame(url)
    te4 =te4.drop_duplicates(["date"])
    q1 = te4["per"].quantile(.25)
    q3 = te4["per"].quantile(.75)
    for i in range(len(url)):
        if url['per'].iloc[i] > 0:
            url['per'].iloc[i] = 0
        else:
            url['per'].iloc[i] = 1
#         if url['per'].iloc[i] > q3:
#             url['per'].iloc[i] = 0
#         elif url['per'].iloc[i] <= q3 and url['per'].iloc[i]>0:
#             url['per'].iloc[i] = 1
#         elif url['per'].iloc[i] <= 0 and url['per'].iloc[i]>q1:
#             url['per'].iloc[i] = 2
#         else:
#             url['per'].iloc[i] = 3
    return url

    # ### 퍼센트 원 핫 인코딩

    # ##### 원핫 인코딩 y값 제거해야 한다 함

    # from tensorflow.keras.utils import to_categorical
    # #float(te4["per"][1])
    # url['per'] = list(to_categorical(url['per']))
    #


# In[97]:


url = Pretreatment(url,stock)
print("Pretreatment")


# In[98]:


url


# In[99]:


url.to_csv('url_Pretreatment.csv')


# In[100]:


url = pd.read_csv('url_Pretreatment.csv')


# In[101]:


#test와 train데이터 분리

# # 전처리 및 토큰화
def morpheme(data):
    soup_content=[]
    j=0
    for i in range(len(data)):
        webpage = requests.get(data['href'].iloc[i])
        soup = BeautifulSoup(webpage.content, "html.parser")
        soup = soup.select_one('#newsViewArea').get_text()
        soup = re.sub('[^ㄱ-ㅎㅏ-ㅣ가-힣A-Za-z ]', ' ', soup)
        soup_content.append(soup)
    return soup_content


# In[102]:


url_content = url
url_content['content']=morpheme(url)
print("morpheme")


# In[103]:


url_content


# In[104]:


url_content.to_csv('url_content_morpheme.csv')


# In[105]:


url_content = pd.read_csv('url_content_morpheme.csv')


# In[106]:


def Delete_cospi(data):
    delete_cospi=[]
    for i in range(len(data)):
        if re.search("코스피", data['content'].iloc[i]) != None:
            delete_cospi.append(i)
        if re.search("주가", data['content'].iloc[i]) != None:
            delete_cospi.append(i)
        if re.search("주식", data['content'].iloc[i]) != None:
            delete_cospi.append(i)
    return data.drop(delete_cospi)


# In[107]:


url_content = Delete_cospi(url_content)


# In[108]:


url_content.to_csv('url_content_dropkospi.csv')


# In[109]:


url_content= pd.read_csv('url_content_dropkospi.csv')


# In[110]:


from sklearn.cluster import KMeans
def cluster_group(data):
    vocab_size = 60  # 상위 500 단어만 사용
    tokenizer = Tokenizer(num_words = vocab_size + 1)
#     tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data['content'])
    clustering = tokenizer.texts_to_sequences(data['content'])
    clustering = pad_sequences(clustering, maxlen = 500, padding='post')
    # 군집화 할 그룹의 갯수 정의
    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters).fit(clustering)

    # trained labels and cluster centers
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    #labels에 merge
    return labels


# In[111]:


url_content['labels'] = cluster_group(url_content)


# In[112]:


# row 생략 없이 출력
pd.set_option('display.max_rows', None)
# col 생략 없이 출력
pd.set_option('display.max_columns', None)
url_content


# url_content.to_csv('labels.csv', encoding='utf-8')

# In[113]:


all_data=url_content.reset_index()


# In[114]:


#합치면 라벨을 못살림
# def merge_group(data):
#     for i in range(len(data)):
#         if i == 0 :
#             news_group_date = data['date'][i]
#             news_group_content = soup
#             count = 1
#             news_group = pd.DataFrame([news_group_date,news_group_content,data['per'][i],count],['date','content','per','count']).transpose()
#             continue
#         if news_group['date'][j] == data['date'][i]:
#             news_group['content'][j] = news_group['content'][j] + soup
#             count + 1
#             news_group['count'][j] = count
#         else :
#             j = j + 1
#             count = 1
#             a = {'date':data['date'][i],'content':soup,'per':data['per'][i],'count':count}
#             news_group = news_group.append(a,ignore_index=True)
# return news_group


# In[ ]:





# In[115]:


all_data


# In[116]:


def train_test(data):
    train, test = train_test_split(data, test_size= 0.2, random_state=42)

    #인덱스 초기화
    train = train.reset_index()
    test = test.reset_index()
    return train, test


# In[117]:


all_data_train, all_data_test = train_test(all_data)


# In[118]:


import copy
def label_split(data):
    label0=copy.deepcopy(data)
    label1=copy.deepcopy(data)
    label2=copy.deepcopy(data)
    label3=copy.deepcopy(data)
    label4=copy.deepcopy(data)
    label5=copy.deepcopy(data)
    label6=copy.deepcopy(data)
    label7=copy.deepcopy(data)
    label8=copy.deepcopy(data)
    label9=copy.deepcopy(data)
    for i in range(len(data)):
        if label0['labels'][i] != 0:
            label0.loc[i,'content'] = '삭제'
        if label1['labels'][i] != 1:
            label1.loc[i,'content'] = '삭제'
        if label2['labels'][i] != 2:
            label2.loc[i,'content'] = '삭제'
        if label3['labels'][i] != 3:
            label3.loc[i,'content'] = '삭제'
        if label4['labels'][i] != 4:
            label4.loc[i,'content'] = '삭제'
        if label5['labels'][i] != 5:
            label5.loc[i,'content'] = '삭제'
        if label6['labels'][i] != 6:
            label6.loc[i,'content'] = '삭제'
        if label7['labels'][i] != 7:
            label7.loc[i,'content'] = '삭제'
        if label8['labels'][i] != 8:
            label8.loc[i,'content'] = '삭제'
        if label9['labels'][i] != 9:
            label9.loc[i,'content'] = '삭제'
    return label1,label2,label3,label0, label4,label5,label6,label7,label8,label9


# In[119]:


url_label0,url_label1,url_label2,url_label3,url_label4,url_label5,url_label6,url_label7,url_label8,url_label9 = label_split(all_data_train)
#train_url_label0,train_url_label1,train_url_label2,train_url_label3


# In[120]:


def text_except_all(data):
    tokken = []
    tok=[]
    hannanum=Hannanum()
    for i in range(len(data)):
        tokken.append(hannanum.pos(data['content'][i]))

    # ### 토큰 중 가장 긴 토큰을 기준으로 반복 및 형태소 중 명사 동사 선택

    lenA = []
    for i in range(len(tokken)):
        lenA.append(len(tokken[i]))
    max(lenA)

    
    Stopword = pd.read_csv("한국어불용어100.txt", header=None, names=['text','x','num'],delimiter = '\t')
    for i in range(len(tokken)):
        all_tokken=[]
        for j in range(lenA[i]):
            if tokken[i][j][1] == 'N' and tokken[i][j][0] not in Stopword['text'].values:
                all_tokken.append(tokken[i][j][0])
        tok.append(all_tokken)
    return tok


# In[121]:


url_label0['content']=text_except_all(url_label0)
url_label1['content']=text_except_all(url_label1)
url_label2['content']=text_except_all(url_label2)
url_label3['content']=text_except_all(url_label3)
url_label4['content']=text_except_all(url_label4)
url_label5['content']=text_except_all(url_label5)
url_label6['content']=text_except_all(url_label6)
url_label7['content']=text_except_all(url_label7)
url_label8['content']=text_except_all(url_label8)
url_label9['content']=text_except_all(url_label9)


# In[ ]:





# In[ ]:





# In[122]:


url_label8['content'][1]


# In[123]:


max_len = 400


# In[124]:


def pad(train):
    vocab_size = 1000  # 상위 500 단어만 사용
    tokenizer = Tokenizer(num_words = vocab_size + 1)
    tokenizer.fit_on_texts(all_data['content'])
    
    X_train = tokenizer.texts_to_sequences(train['content'])
    X_train = pad_sequences(X_train, maxlen = max_len, padding='post')

    X_train2 = pd.DataFrame(X_train)
    X_train2=X_train2.astype('float64')
#     la = pd.DataFrame(test['labels'].astype(np.float64))
#     X_train2 = pd.concat([X_train2,la],ignore_index=True,axis=1)
    
    


    y_train = train['per'].astype(np.float64)
    y_train = pd.DataFrame(y_train)



    X_train2 = np.array(X_train2).reshape(X_train2.shape[0], 1,X_train2.shape[1])
    y_train = np.array(y_train).reshape(y_train.shape[0], 1, y_train.shape[1])

    return X_train2,y_train


# In[125]:


X_train2_label0, y_train_label0 = pad(url_label0)
X_train2_label1, y_train_label1 = pad(url_label1)
X_train2_label2, y_train_label2 = pad(url_label2)
X_train2_label3, y_train_label3 = pad(url_label3)
X_train2_label4, y_train_label4 = pad(url_label4)
X_train2_label5, y_train_label5 = pad(url_label5)
X_train2_label6, y_train_label6 = pad(url_label6)
X_train2_label7, y_train_label7 = pad(url_label7)
X_train2_label8, y_train_label8 = pad(url_label8)
X_train2_label9, y_train_label9 = pad(url_label9)


# In[126]:


X_train2_label0.shape


# In[127]:


# test셋 준비
url_label0,url_label1,url_label2,url_label3,url_label4,url_label5,url_label6,url_label7,url_label8,url_label9 = label_split(all_data_test)
#train_url_label0,train_url_label1,train_url_label2,train_url_label3
# 불용어 처리
url_label0['content']=text_except_all(url_label0)
url_label1['content']=text_except_all(url_label1)
url_label2['content']=text_except_all(url_label2)
url_label3['content']=text_except_all(url_label3)
url_label4['content']=text_except_all(url_label4)
url_label5['content']=text_except_all(url_label5)
url_label6['content']=text_except_all(url_label6)
url_label7['content']=text_except_all(url_label7)
url_label8['content']=text_except_all(url_label8)
url_label9['content']=text_except_all(url_label9)
# 패딩
X_test2_label0, y_test_label0 = pad(url_label0)
X_test2_label1, y_test_label1 = pad(url_label1)
X_test2_label2, y_test_label2 = pad(url_label2)
X_test2_label3, y_test_label3 = pad(url_label3)
X_test2_label4, y_test_label4 = pad(url_label4)
X_test2_label5, y_test_label5 = pad(url_label5)
X_test2_label6, y_test_label6 = pad(url_label6)
X_test2_label7, y_test_label7 = pad(url_label7)
X_test2_label8, y_test_label8 = pad(url_label8)
X_test2_label9, y_test_label9 = pad(url_label9)


# In[128]:


from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Bidirectional, concatenate
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# In[134]:



# Define the Keras TensorBoard callback.

from datetime import datetime



logdir="logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


# In[129]:


len(globals()['X_train2_label'+str(1)])


# In[155]:


X_train2_label4.shape


# In[148]:


def buildModel(length, label):
    label0_input= Input(shape=(1,length), name='label0')
    label1_input= Input(shape=(1,length), name='label1')
    label2_input= Input(shape=(1,length), name='label2')
    label3_input= Input(shape=(1,length), name='label3')
    label4_input= Input(shape=(1,length), name='label4')
    label5_input= Input(shape=(1,length), name='label5')
    label6_input= Input(shape=(1,length), name='label6')
    label7_input= Input(shape=(1,length), name='label7')
    label8_input= Input(shape=(1,length), name='label8')
    label9_input= Input(shape=(1,length), name='label9')
    
    
    label0_layer = LSTM(length, return_sequences=True,activation = 'relu')(label0_input)
    label1_layer = LSTM(length, return_sequences=True, activation = 'relu')(label1_input)
    label2_layer = LSTM(length, return_sequences=True, activation = 'relu')(label2_input)
    label3_layer = LSTM(length, return_sequences=True, activation = 'relu')(label3_input)
    label4_layer = LSTM(length, return_sequences=True, activation = 'relu')(label4_input)
    label5_layer = LSTM(length, return_sequences=True, activation = 'relu')(label5_input)
    label6_layer = LSTM(length, return_sequences=True, activation = 'relu')(label6_input)
    label7_layer = LSTM(length, return_sequences=True, activation = 'relu')(label7_input)
    label8_layer = LSTM(length, return_sequences=True, activation = 'relu')(label8_input)
    label9_layer = LSTM(length, return_sequences=True, activation = 'relu')(label9_input)
    
    label0_layer = Dropout(0.3)(label0_layer)
    label0_layer = Dense(length, activation='relu')(label0_layer)
    label0_layer = Dropout(0.3)(label0_layer)
    label0_layer = Dense(length, activation='relu')(label0_layer)
    
    label1_layer = Dropout(0.3)(label1_layer)
    label1_layer = Dense(length, activation='relu')(label1_layer)
    label1_layer = Dropout(0.3)(label1_layer)
    label1_layer = Dense(length, activation='relu')(label1_layer)
    
    label2_layer = Dropout(0.3)(label2_layer)
    label2_layer = Dense(length, activation='relu')(label2_layer)
    label2_layer = Dropout(0.3)(label2_layer)
    label2_layer = Dense(length, activation='relu')(label2_layer)
    
    label3_layer = Dropout(0.3)(label3_layer)
    label3_layer = Dense(length, activation='relu')(label3_layer)
    label3_layer = Dropout(0.3)(label3_layer)
    label3_layer = Dense(length, activation='relu')(label3_layer)
    
    label4_layer = Dropout(0.3)(label4_layer)
    label4_layer = Dense(length, activation='relu')(label4_layer)
    label4_layer = Dropout(0.3)(label4_layer)
    label4_layer = Dense(length, activation='relu')(label4_layer)
    
    label5_layer = Dropout(0.3)(label5_layer)
    label5_layer = Dense(length, activation='relu')(label5_layer)
    label5_layer = Dropout(0.3)(label5_layer)
    label5_layer = Dense(length, activation='relu')(label5_layer)
    
    label6_layer = Dropout(0.3)(label6_layer)
    label6_layer = Dense(length, activation='relu')(label6_layer)
    label6_layer = Dropout(0.3)(label6_layer)
    label6_layer = Dense(length, activation='relu')(label6_layer)
    
    label7_layer = Dropout(0.3)(label7_layer)
    label7_layer = Dense(length, activation='relu')(label7_layer)
    label7_layer = Dropout(0.3)(label7_layer)
    label7_layer = Dense(length, activation='relu')(label7_layer)
    
    label8_layer = Dropout(0.3)(label8_layer)
    label8_layer = Dense(length, activation='relu')(label8_layer)
    label8_layer = Dropout(0.3)(label8_layer)
    label8_layer = Dense(length, activation='relu')(label8_layer)
    
    label9_layer = Dropout(0.3)(label9_layer)
    label9_layer = Dense(length, activation='relu')(label9_layer)
    label9_layer = Dropout(0.3)(label9_layer)
    label9_layer = Dense(length, activation='relu')(label9_layer)
    
    
    output = concatenate(
        [
            label0_layer,
            label1_layer,
            label2_layer,
            label3_layer,
            label4_layer,
            label5_layer,
            label6_layer,
            label7_layer,
            label8_layer,
            label9_layer
        ]
    )
    output = LSTM(length, return_sequences=True, activation = 'relu')(output)
    output = Dense(length, activation = 'relu')(output)
    output = Dense(label, activation='softmax')(output)
    
    model = tf.keras.Model(
        inputs=
        [
            label0_input,
            label1_input,
            label2_input,
            label3_input,
            label4_input,
            label5_input,
            label6_input,
            label7_input,
            label8_input,
            label9_input,
        ],
        outputs=
        [
            output
        ]
    )
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['acc','sparse_categorical_accuracy'])
    return model


# In[149]:


# max_len 패딩시 사요한 기준
max_len= 400
y_train = pd.DataFrame(all_data_train['per'])
y_train = np.array(y_train).reshape(y_train.shape[0], 1,y_train.shape[1])
y_test = pd.DataFrame(all_data_test['per'])
y_test = np.array(y_test).reshape(y_test.shape[0], 1,y_test.shape[1])
lstm = buildModel(max_len,4)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
history = lstm.fit(
    [
        X_train2_label0,
        X_train2_label1,
        X_train2_label2,
        X_train2_label3,
        X_train2_label4,
        X_train2_label5,
        X_train2_label6,
        X_train2_label7,
        X_train2_label8,
        X_train2_label9
    ],
    
        y_train
    ,
    epochs=100,
    callbacks=[es,mc,tensorboard_callback],
    batch_size=62,
    validation_split=0.2,
    validation_data=(
        [
            X_test2_label0,
            X_test2_label1,
            X_test2_label2,
            X_test2_label3,
            X_test2_label4,
            X_test2_label5,
            X_test2_label6,
            X_test2_label7,
            X_test2_label8,
            X_test2_label9
        ],
        
            y_test
        
    )
)


# In[150]:


get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir logs')


# In[151]:


lstm.summary()

loaded_model = load_model('best_model.h5')

print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate( [
            X_test2_label0,
            X_test2_label1,
            X_test2_label2,
            X_test2_label3,
            X_test2_label4,
            X_test2_label5,
            X_test2_label6,
            X_test2_label7,
            X_test2_label8,
            X_test2_label9
        ],
    y_test)[1]))
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(history.history['loss'], 'y', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')

acc_ax.plot(history.history['sparse_categorical_accuracy'], 'g', label='train acc')
acc_ax.plot(history.history['val_sparse_categorical_accuracy'], 'b', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




