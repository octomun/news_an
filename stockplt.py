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
for i in range(len(stock)):
    date_time.append(datetime.datetime(year=int(str(stock['dates'].values[i])[:4]),
                                           month=int(str(stock['dates'].values[i])[4:6]),
                                           day=int(str(stock['dates'].values[i])[6:]),
                                           hour=int(str(stock['times'].values[i])[:-2]),
                                           second=int(str(stock['times'].values[i])[-2:])))
stock['date_time']=date_time
# plt.plot(stock['date_time'],stock['closes'])
# plt.show()
#이동평균선
ma_day = [5, 10, 20]
for ma in ma_day:
    column_name = f"{ma}일 이동평균선"
    stock[column_name] = stock['closes'].rolling(ma*380).mean()
plt.plot(stock['date_time'],stock[['closes', '5일 이동평균선', '10일 이동평균선', '20일 이동평균선']])
plt.show()