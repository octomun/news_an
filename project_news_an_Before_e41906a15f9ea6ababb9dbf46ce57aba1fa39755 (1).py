#!/usr/bin/env python
# coding: utf-8

# In[3]:


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


# In[5]:


def read_file():
    stock = pd.read_csv("sk하이닉스.csv")
    stock['date'] = pd.to_datetime(stock['date'], format ='%Y-%m-%d')
    for i in range(len(stock)):
        stock['per'][i]=(float(stock['per'][i].replace("%",""))/100)

    # # 뉴스 url읽기
    url = pd.read_csv("crawling_news2.txt", header = None, names = ["date","href"])
    url['per']= "A"
    return url, stock


# In[6]:


url, stock = read_file()
print("read_file")


# In[7]:


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


# In[8]:


int(url['date'].iloc[1][0:2])


# In[9]:


url = lte_year(url,2015)
# url = url.reset_index()


# In[10]:


url.to_csv('url.csv')


# In[4]:


url = pd.read_csv('url.csv')


# In[15]:


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
        if url['per'].iloc[i] > q3:
            url['per'].iloc[i] = 0
        elif url['per'].iloc[i] <= q3 and url['per'].iloc[i]>0:
            url['per'].iloc[i] = 1
        elif url['per'].iloc[i] <= 0 and url['per'].iloc[i]>q1:
            url['per'].iloc[i] = 2
        else:
            url['per'].iloc[i] = 3
    return url

    # ### 퍼센트 원 핫 인코딩

    # ##### 원핫 인코딩 y값 제거해야 한다 함

    # from tensorflow.keras.utils import to_categorical
    # #float(te4["per"][1])
    # url['per'] = list(to_categorical(url['per']))
    #


# In[16]:


url = Pretreatment(url,stock)
print("Pretreatment")


# In[17]:


url


# In[18]:


url.to_csv('url_Pretreatment.csv')


# In[5]:


url = pd.read_csv('url_Pretreatment.csv')


# In[20]:


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


# In[21]:


url_content = url
url_content['content']=morpheme(url)
print("morpheme")


# In[22]:


url_content


# In[23]:


url_content.to_csv('url_content_morpheme.csv')


# In[24]:


url_content = pd.read_csv('url_content_morpheme.csv')


# In[25]:


def Delete_cospi(data):
    delete_cospi=[]
    for i in range(len(data)):
        if re.search("코스피", data['content'].iloc[i]) != None:
            delete_cospi.append(i)
#         if re.search("주가", data['content'].iloc[i]) != None:
#             delete_cospi.append(i)
        if re.search("주식", data['content'].iloc[i]) != None:
            delete_cospi.append(i)
    return data.drop(delete_cospi)


# In[26]:


url_content = Delete_cospi(url_content)


# In[27]:


url_content.to_csv('url_content_dropkospi.csv')


# In[6]:


url_content= pd.read_csv('url_content_dropkospi.csv')


# In[29]:


from sklearn.cluster import KMeans
def cluster_group(data):
#     vocab_size = 1000  # 상위 500 단어만 사용
#     tokenizer = Tokenizer(num_words = vocab_size + 1)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data['content'])
    clustering = tokenizer.texts_to_sequences(data['content'])
    clustering = pad_sequences(clustering, maxlen = 500, padding='post')
    # 군집화 할 그룹의 갯수 정의
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters).fit(clustering)

    # trained labels and cluster centers
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    #labels에 merge
    return labels


# In[30]:


url_content['labels'] = cluster_group(url_content)


# In[31]:


# row 생략 없이 출력
pd.set_option('display.max_rows', None)
# col 생략 없이 출력
pd.set_option('display.max_columns', None)
url_content


# In[1]:


url_content.to_csv('labels.csv', encoding='utf-8')


# In[7]:


url_content = pd.read_csv('labels.csv')


# In[8]:


all_data=url_content.reset_index()


# In[34]:


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





# In[9]:


all_data


# In[10]:


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


# In[11]:


all_data['content'] = text_except_all(all_data)
# print("text_except")


# In[12]:


all_data


# In[13]:


from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_data['content'])

vocab_size = 1000  # 상위 500 단어만 사용
tokenizer = Tokenizer(num_words = vocab_size + 1)
tokenizer.fit_on_texts(all_data['content'])

print(tokenizer.word_index) #인덱스가 어떻게 부여됬는지(입력된 단어 순서)
print(tokenizer.word_counts) #상위 몇개 단어를 했을 때 어떻게 부여됬는지(입력된 단어 순서)

def text_size(num):
    threshold = num
    total_cnt = len(tokenizer.word_index) # 단어의 수
    rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
    total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
    rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

    # 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
    for key, value in tokenizer.word_counts.items():
        total_freq = total_freq + value

        # 단어의 등장 빈도수가 threshold보다 작으면
        if(value < threshold):
            rare_cnt = rare_cnt + 1
            rare_freq = rare_freq + value

    print('단어 집합(vocabulary)의 크기 :',total_cnt)
    print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
    print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
    print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

    #단어수가 2개인 단어의 빈도가 6.1%라 유의미한 영향을 줄 수 있어 제외하지 않는다

    # 전체 단어 개수 중 빈도수 2이하인 단어는 제거.
    # 0번 패딩 토큰을 고려하여 + 1
    vocab_size = total_cnt - rare_cnt + 1
    print('단어 집합의 크기 :',vocab_size)


    # ## 앞의 형태소분석을 붙여씀
    # ### 불필요하게 주가를 넣는 부분이 있고 href에서 본문을 따오는 부분 함수화 고려

text_size(2)


# In[14]:


def train_test(data):
    train, test = train_test_split(data, test_size= 0.2, random_state=1234)

    #인덱스 초기화
    train = train.reset_index()
    test = test.reset_index()
    return train, test

train, test = train_test(all_data)
print("train_test")


# In[15]:


print(len(train))
print(len(test))


# In[16]:


all_data['content']


# In[17]:


# ## 패딩

def below_threshold_len(max_len, nested_list):
  cnt = 0
  for s in nested_list:
    if(len(s) <= max_len):
        cnt = cnt + 1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))*100))


print('뉴스의 최대 길이 :',max(len(l) for l in train['content']))
print('뉴스의 평균 길이 :',sum(map(len, train['content']))/len(url))
plt.hist([len(s) for s in train['content']], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()


# In[18]:


max_len = 400
below_threshold_len(max_len, train['content'])


# In[19]:


# train_data = morpheme(train)
# test_data = morpheme(test)

# train_data['content'] = text_except(train_data['content'])
# test_data['content'] = text_except(test_data['content'])

# X_train = train['content']
# X_test = test['content']

X_train = tokenizer.texts_to_sequences(train['content'])
X_test = tokenizer.texts_to_sequences(test['content'])
X_train = pad_sequences(X_train, maxlen = max_len, padding='post')
X_test = pad_sequences(X_test, maxlen = max_len, padding='post')



X_train2 = pd.DataFrame(X_train)
la = pd.DataFrame(train['labels'].astype(np.float64))
X_train2=X_train2.astype('float64')
# X_train2 = pd.concat([X_train2,la],ignore_index=True,axis=1)

X_test2 = pd.DataFrame(X_test)
X_test2=X_test2.astype('float64')
la2 = pd.DataFrame(test['labels'].astype(np.float64))
# X_test2 = pd.concat([X_test2,la2],ignore_index=True,axis=1)


y_train = train['per'].astype(np.float64)
y_train = pd.DataFrame(y_train)
y_test = test['per'].astype(np.float64)
y_test = pd.DataFrame(y_test)


#(size, timestep, feature)
X_train2 = np.array(X_train2).reshape(X_train2.shape[0], X_train2.shape[1],1)
y_train = np.array(y_train).reshape(y_train.shape[0], y_train.shape[1], 1)
X_test2 = np.array(X_test2).reshape(X_test2.shape[0], X_test2.shape[1], 1)
y_test = np.array(y_test).reshape(y_test.shape[0], y_test.shape[1], 1)

drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]
# 빈 샘플들을 제거
# X_train = np.delete(X_train, drop_train, axis=0)
# y_train = np.delete(y_train, drop_train, axis=0)
print(len(X_train))
print(len(y_train))


# In[20]:


y_test


# In[21]:


from tensorflow.keras.layers import Embedding, Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# In[28]:


model = Sequential()
model.add(Bidirectional(LSTM(len(X_train2), input_shape = (len(X_train2),1), return_sequence=True, dropout=0.5)))
# model.add(Dropout(0.2, input_shape=(len(X_train2),1)))
# model.add(Dropout(0.2))
initializer = tf.keras.initializers.Ones() #가중치 초기화
model.add(Dense(90, activation='relu',kernel_initializer=initializer))
model.add(Dropout(0.2))
# model.add(Dense(90, activation='relu'))
model.add(Dense(4, activation='softmax'))


# In[ ]:


model = Sequential()
#model.add(Embedding(vocab_size, 100)) #모델에 입력크기를 고정된 크기고 제한
#model.add(Dense(2, activation='softmax'))
model.add(LSTM(128, input_shape = (len(X_train),1)))
#,return_sequences=True, input_shape = (300,1) 입력형식, stateful=True 상태유지
# model.add(Dropout(0.2, input_shape=(270,1)))
initializer = tf.keras.initializers.HeNormal() #가중치 초기화
# model.add(Dense(90, activation='relu',kernel_initializer=initializer))
model.add(Dense(90, activation='relu'))

model.add(Dense(90, activation='relu'))
model.add(Dense(4, activation='softmax'))


# In[26]:


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)


# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc','sparse_categorical_accuracy'])
model.compile(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['acc','sparse_categorical_accuracy'])

history = model.fit(X_train2, y_train, epochs=100, callbacks=[es,mc], batch_size=60, validation_split=0.2)
#validation_split 전체데이터(train)중 얼마를 검토(test)할 것이냐
#batch_size 계산 후 가중치를 넘길 계산 단위


# In[27]:


loaded_model = load_model('best_model.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test2, y_test)[1]))

# 학습 결과 그래프 그리기

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

# # 끝


# In[ ]:





# In[ ]:


# train_data = morpheme(train)
# test_data = morpheme(test)

# train_data['content'] = text_except(train_data['content'])
# test_data['content'] = text_except(test_data['content'])

# X_train = train['content']
# X_test = test['content']

X_train = tokenizer.texts_to_sequences(train['content'])
X_test = tokenizer.texts_to_sequences(test['content'])
X_train = pad_sequences(X_train, maxlen = max_len, padding='post')
X_test = pad_sequences(X_test, maxlen = max_len, padding='post')



X_train2 = pd.DataFrame(X_train)
la = pd.DataFrame(train['labels'].astype(np.float64))
X_train2=X_train2.astype('float64')
X_train2 = pd.concat([X_train2,la],ignore_index=True,axis=1)

X_test2 = pd.DataFrame(X_test)
X_test2=X_test2.astype('float64')
la2 = pd.DataFrame(test['labels'].astype(np.float64))
X_test2 = pd.concat([X_test2,la2],ignore_index=True,axis=1)


y_train = train['per'].astype(np.float64)
y_train = pd.DataFrame(y_train)
y_test = test['per'].astype(np.float64)
y_test = pd.DataFrame(y_test)


#(size, timestep, feature)
X_train2 = np.array(X_train2).reshape(X_train2.shape[0], X_train2.shape[1]/2,2)
y_train = np.array(y_train).reshape(y_train.shape[0], y_train.shape[1], 1)
X_test2 = np.array(X_test2).reshape(X_test2.shape[0], X_test2.shape[1], 1)
y_test = np.array(y_test).reshape(y_test.shape[0], y_test.shape[1], 1)


# x_3d = pd.DataFrame({'content':X_train, 'labels':test['labels']})

drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]
# 빈 샘플들을 제거
# X_train = np.delete(X_train, drop_train, axis=0)
# y_train = np.delete(y_train, drop_train, axis=0)
print(len(X_train))
print(len(y_train))


# In[ ]:


X_test


# In[ ]:





# In[ ]:




