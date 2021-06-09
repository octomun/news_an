import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from stock_mysql_64 import stock_setting
import pandas as pd
os.chdir('F:/stock')

conn = stock_setting.connect_db()
fetch = stock_setting.fetch_all_stock_code(conn)
fetch = pd.DataFrame(fetch)
code = fetch['code']
for fetch in range(len(fetch)):
    try :
        stock_setting.drop_table_predict_pro(conn, code.values[fetch])
        stock_setting.create_table_predict_pro(conn, code.values[fetch])
    except:
        stock_setting.create_table_predict_pro(conn, code.values[fetch])
    stock = stock_setting.fetch_all_now(conn,code.values[fetch])
    stock = pd.DataFrame(stock)
    standard=1
    testing_data_len=standard*60 #2달
    day=30
    days=standard*day

    #datetime으로 날자 조합
    stock['date_time']=stock['dates']
    stock = stock.sort_values(by='date_time',ascending=True)

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
        stock[column_name] = stock['closes'].rolling(ma).mean()


    #min_max 정규화
    max_min=[]
    i=0
    for i in range(len(stock)):
        max_min.append((stock['closes'][i] - stock['closes'][i-days:i].min())
                       / (stock['closes'][i-days:i].max() - stock['closes'][i-days:i].min()))
    stock['max_min']=max_min


    #일일 수익률(종가 - 시작가)
    def day_pct_change(data):
        pct = []
        for i in range(len(stock)):
            pct.append((data['closes'][i]-data['opens'][i])/data['opens'][i]*100)
        return pct

    stock['pct'] = day_pct_change(stock)


    #이동평균 이상치, 결측치 제거 and min_max시 마지막 라인 결측치 생김)
    #과거 380*10 (60일데이터로 예측기간 > 이동평균 10일)는 결측치가 있어 삭제
    stock = stock[days:]

    week_day=[]
    month = []
    for i in range(len(stock)):
        week_day.append(stock['date_time'][i].weekday())
        month.append(int(str(stock['dates'][i])[4:6]))
    stock['week_day'] =week_day
    stock['month'] =month
    print(stock)
    for i in range(len(stock)):
        stock_setting.insert_tabel_predict_pro(dates=stock['dates'][i],times=stock['times'][i],opens= float(stock['opens'][i]),
                                       highs=float(stock['highs'][i]),  lows=float(stock['lows'][i]),  closes=float(stock['closes'][i]),
                                       vols=float(stock['vols'][i]), date_time=stock['date_time'][i],  fdays_rolling=float(stock['5days_rolling'][i]),
                                       tdays_rolling=float(stock['10days_rolling'][i]),  max_min=float(stock['max_min'][i]),
                                       pct=float(stock['pct'][i]), week_day=int(stock['week_day'][i]), month=int(stock['month'][i]),
                                       db =conn, db_name=code[fetch]
        )
    stock_setting.drop_table_now(conn, code.values[fetch])
    # stock.to_csv('predict_preprocess.csv')