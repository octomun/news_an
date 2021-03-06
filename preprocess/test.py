import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
os.chdir('F:/stock')
stock = pd.read_csv('output_A005930.csv')
print(stock['times'].shape)
print(str(stock['dates'].values[1])[:4])
print(str(stock['times'].values[1])[-2:])
date_time=[]
i=0
for i in range(len(stock)):
    date_time.append(datetime.datetime(year=int(str(stock['dates'].values[i])[:4]),
                                           month=int(str(stock['dates'].values[i])[4:6]),
                                           day=int(str(stock['dates'].values[i])[6:]),
                                           hour=int(str(stock['times'].values[i])[:-2]),
                                           second=int(str(stock['times'].values[i])[-2:])))
#datetime으로 날자 조합
stock['date_time']=date_time
stock = stock.sort_values(by='date_time',ascending=True)
print(stock)






#이동평균선
#rolling이 낮은 인덱스(최근 값)을 기준으로 아래로 진행되어 최근값의 이동평균이 구해지지 않아
#뒤집어서 구한다
# stock=stock.sort_values('date_time',ascending=True)
# stock = stock.reindex()
#소용 없음
#make_rolling을 만듬


ma_day = [5, 10]
for ma in ma_day:
    column_name = f"{ma}days_rolling"
    stock[column_name] = stock['closes'].rolling(ma * 380).mean()

#min_max 정규화
max_min=[]
i=0
for i in range(len(stock)):
    max_min.append((stock['closes'][i] - stock['closes'][i-760:i].min())
                   / (stock['closes'][i-760:i].max() - stock['closes'][i-760:i].min()))
stock['max_min']=max_min
print(stock['date_time'],"  ",stock['max_min'])

#일일 수익률(종가 - 시작가)
stock['pct'] = stock['closes'].pct_change()

#이동평균 이상치, 결측치 제거 and min_max시 마지막 라인 결측치 생김)
#과거 380*10 (2일데이터로 예측기간 < 이동평균 10일)는 결측치가 있어 삭제
stock = stock[380*10:]
print(stock)

week_day=[]
month = []
for i in range(len(stock)):
    week_day.append(stock['date_time'][i].weekday())
    month.append(int(str(stock['dates'][i])[4:6]))
stock['week_day'] =week_day
stock['week_day'] =month


stock.to_csv('stock_dropna_sort.csv', index=False, encoding='utf-8')
