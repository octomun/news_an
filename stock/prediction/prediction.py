import pandas as pd
import numpy as np
from stock_mysql_64 import stock_setting
import os
os.chdir('F:/stock')
standard=1
testing_data_len=standard*60 #2달
day=30
days=standard*day

stock= pd.read_csv('predict_preprocess.csv', encoding='utf-8')
print(stock)
print(stock.iloc[-1,3])
print(stock.iloc[-testing_data_len:-testing_data_len+1,3])
data = stock[['pct','5days_rolling','10days_rolling','max_min','week_day','month','vols']]
x_data = np.array(data)
x_data = x_data.reshape(1,x_data.shape[0],7)
from keras.models import load_model
loaded_model = load_model('best_model_day.h5')
predictions=loaded_model.predict(x_data)
predictions = pd.DataFrame(predictions)
predictions = (predictions/100*stock.iloc[-1,3])+stock.iloc[-1,3]

conn = stock_setting.connect_db()

if stock_setting.fetch_all_predict(conn):
    stock_setting.drop_table_predict(conn)
    stock_setting.create_table_predict(conn)
stock_setting.insert_tabel_predict(predictions[0][0],conn)
print(predictions)

