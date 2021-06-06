import os

from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from preprocess import delet_and_cluster
os.chdir('F:/news/newss')
from sklearn.model_selection import train_test_split
from konlpy.tag import Hannanum
import copy
import pandas as pd
import numpy as np



def train_test(data):
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    # 인덱스 초기화
    train = train.reset_index()
    test = test.reset_index()
    return train, test




def label_split(data):
    label0 = copy.deepcopy(data)
    label1 = copy.deepcopy(data)
    label2 = copy.deepcopy(data)
    label3 = copy.deepcopy(data)
    label4 = copy.deepcopy(data)
    label5 = copy.deepcopy(data)
    label6 = copy.deepcopy(data)
    label7 = copy.deepcopy(data)
    label8 = copy.deepcopy(data)
    label9 = copy.deepcopy(data)
    for i in range(len(data)):
        if label0['labels'][i] != 0:
            label0.loc[i, 'content'] = '삭제'
        if label1['labels'][i] != 1:
            label1.loc[i, 'content'] = '삭제'
        if label2['labels'][i] != 2:
            label2.loc[i, 'content'] = '삭제'
        if label3['labels'][i] != 3:
            label3.loc[i, 'content'] = '삭제'
        if label4['labels'][i] != 4:
            label4.loc[i, 'content'] = '삭제'
        if label5['labels'][i] != 5:
            label5.loc[i, 'content'] = '삭제'
        if label6['labels'][i] != 6:
            label6.loc[i, 'content'] = '삭제'
        if label7['labels'][i] != 7:
            label7.loc[i, 'content'] = '삭제'
        if label8['labels'][i] != 8:
            label8.loc[i, 'content'] = '삭제'
        if label9['labels'][i] != 9:
            label9.loc[i, 'content'] = '삭제'
    return label1, label2, label3, label0, label4, label5, label6, label7, label8, label9



def text_except_all(data):
    tokken = []
    tok = []
    hannanum = Hannanum()
    for i in range(len(data)):
        tokken.append(hannanum.pos(data['content'][i]))

    # ### 토큰 중 가장 긴 토큰을 기준으로 반복 및 형태소 중 명사 동사 선택

    lenA = []
    for i in range(len(tokken)):
        lenA.append(len(tokken[i]))
    max(lenA)

    Stopword = pd.read_csv("한국어불용어100.txt", header=None, names=['text', 'x', 'num'], delimiter='\t')
    for i in range(len(tokken)):
        all_tokken = []
        for j in range(lenA[i]):
            if tokken[i][j][1] == 'N' and tokken[i][j][0] not in Stopword['text'].values:
                all_tokken.append(tokken[i][j][0])
        tok.append(all_tokken)
    return tok





# In[124]:


def pad(train):
    max_len = 400
    all_data = delet_and_cluster.del_and_cluster.url_content.reset_index()
    vocab_size = 1000  # 상위 500 단어만 사용
    tokenizer = Tokenizer(num_words=vocab_size + 1)
    tokenizer.fit_on_texts(all_data['content'])

    X_train = tokenizer.texts_to_sequences(train['content'])
    X_train = pad_sequences(X_train, maxlen=max_len, padding='post')

    X_train2 = pd.DataFrame(X_train)
    X_train2 = X_train2.astype('float64')
    #     la = pd.DataFrame(test['labels'].astype(np.float64))
    #     X_train2 = pd.concat([X_train2,la],ignore_index=True,axis=1)

    y_train = train['per'].astype(np.float64)
    y_train = pd.DataFrame(y_train)

    X_train2 = np.array(X_train2).reshape(X_train2.shape[0], 1, X_train2.shape[1])
    y_train = np.array(y_train).reshape(y_train.shape[0], 1, y_train.shape[1])

    return X_train2, y_train



