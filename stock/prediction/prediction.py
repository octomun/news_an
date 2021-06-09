import pandas as pd
import numpy as np
from stock_mysql_64 import stock_setting
import os
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
    stock = stock_setting.fetch_all_predict_pro(conn,code.values[fetch])
    stock = pd.DataFrame(stock)

    standard=1
    testing_data_len=standard*60 #2달
    day=30
    days=standard*day



    stock= pd.read_csv('predict_preprocess.csv', encoding='utf-8')
    data = stock[['pct','5days_rolling','10days_rolling','max_min','week_day','month','vols']]
    x_data = np.array(data)
    x_data = x_data.reshape(1,x_data.shape[0],7)
    from keras.models import load_model
    loaded_model = load_model(f'{code.values[fetch]}.h5')
    predictions=loaded_model.predict(x_data)
    predictions = pd.DataFrame(predictions)
    predictions = (predictions/100*stock.iloc[-1,3])+stock.iloc[-1,3]
    print(predictions[0][0])
    # exit()
    stock_setting.update_table_stock_code(conn,code.values[fetch],float(predictions[0][0]))
