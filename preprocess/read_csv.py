import os
os.chdir('F:/news/newss')
import pandas as pd
from datetime import timedelta

class ReadCSV:
    def read_file(url):
        stock = pd.read_csv(url)
        stock['date'] = pd.to_datetime(stock['date'], format ='%Y-%m-%d')
        for i in range(len(stock)):
            float_per= (float(stock['per'][i].replace("%",""))/100)
            stock['per'][i] = float_per
        # # 뉴스 url읽기
        url = pd.read_csv("crawling_news2.txt", header = None, names = ["date","href"])
        url['per']= "A"
        return url, stock

    def lte_year(data,year):
        url_lte_2019=[]
        for i in range(len(data)):
            if int(data['date'].iloc[i][:2]) > int(str(pd.Timestamp.today())[2:4]):
                year = int(data['date'].iloc[i][:2]) + 1900
            else :
                year = int(data['date'].iloc[i][:2]) + 2000
            data['date'].iloc[i] = pd.Timestamp(year , int(data['date'].iloc[i][3:5]) , int(data['date'].iloc[i][6:8]))
            if pd.Timestamp(data['date'].iloc[i]) >= pd.Timestamp(year,1,1):
                url_lte_2019.append(i)
        data = data.iloc[url_lte_2019]
        return data



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


    if os.path.isfile('url_Pretreatment.csv'):
        url = pd.read_csv('url_Pretreatment.csv')
    else:
        url, stock = read_file(url="sk하이닉스.csv")
        print("read_file")
        url = lte_year(url, 2015)
        print("read_file")
        url = Pretreatment(url, stock)
        print("Pretreatment")
        url.to_csv('url_Pretreatment.csv')
