import os
os.chdir('F:/news/newss')
import pandas as pd
from preprocess import tokkenize
import re
from sklearn.cluster import KMeans
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class del_and_cluster:
    def Delete_cospi(data):
        delete_cospi = []
        for i in range(len(data)):
            if re.search("코스피", data['content'].iloc[i]) != None:
                delete_cospi.append(i)
            if re.search("주가", data['content'].iloc[i]) != None:
                delete_cospi.append(i)
            if re.search("주식", data['content'].iloc[i]) != None:
                delete_cospi.append(i)
        return data.drop(delete_cospi)

    def cluster_group(data):
        vocab_size = 100  # 상위 500 단어만 사용
        tokenizer = Tokenizer(num_words=vocab_size + 1)
        # tokenizer = Tokenizer()
        tokenizer.fit_on_texts(data['content'])
        clustering = tokenizer.texts_to_sequences(data['content'])
        clustering = pad_sequences(clustering, maxlen=500, padding='post')
        # 군집화 할 그룹의 갯수 정의
        n_clusters = 10
        kmeans = KMeans(n_clusters=n_clusters).fit(clustering)

        # trained labels and cluster centers
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        # labels에 merge
        return labels

    if os.path.isfile('url_content_del_cluster.csv'):
        url_content = pd.read_csv('url_content_del_cluster.csv')
    else:
        url_content = tokkenize.tokkenz.url_content
        url_content = Delete_cospi(url_content)
        url_content['labels'] = cluster_group(url_content)
        url_content.to_csv('url_content_del_cluster.csv')
        # return url_content




