# # 전처리 및 토큰화
import os
os.chdir('F:/news/newss')

import requests
from bs4 import BeautifulSoup
from preprocess import read_csv
import re
import pandas as pd
class tokkenz:
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

    if os.path.isfile('url_content_morpheme'):
         url_content = pd.read_csv('url_content_morpheme.csv')
    else:
        url_content = read_csv.ReadCSV.url
        url_content['content'] = morpheme(url_content)
        print("morpheme")
        url_content.to_csv('url_content_morpheme.csv')
    # return url_content