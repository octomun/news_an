#!/usr/bin/python
# coding = utf-8
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import codecs, json
import scaler
import datetime
from stock_mysql_64 import stock_setting
import os
import sys
try:
    os.chdir(sys._MEIPASS)
    print(sys._MEIPASS)
except:
    os.chdir(os.getcwd())


conn = stock_setting.connect_db()
fetch = stock_setting.fetch_all_stock_code(conn)
fetch = pd.DataFrame(fetch)
code = fetch['code']

for fetch in range(len(fetch)):
    if os.path.isfile(f'{code[fetch]}.h5'):
        print(f'{code[fetch]} pass')
        continue
    try :
        stock_setting.drop_table_predict_pro(conn, code.values[fetch])
        stock_setting.create_table_predict_pro(conn, code.values[fetch])
    except:
        stock_setting.create_table_predict_pro(conn, code.values[fetch])
    stock = stock_setting.fetch_all_model_pro(conn,code.values[fetch])
    stock = pd.DataFrame(stock)
    standard=1
    testing_data_len=standard*60 #2달
    day=30
    days=standard*day


    # i=60
    # un_pct=[]
    # a=pd.DataFrame([0.134281])
    # un_pct.append((a/100*stock['opens'][-testing_data_len+i:-testing_data_len+i+1].values)+stock['opens'][-testing_data_len+i:-testing_data_len+i+1].values)
    # print(un_pct[0])
    # # print(a[i]/100)
    # print(stock.iloc[-testing_data_len+i,2])
    # exit()

    # print(stock)
    #
    # date_time=[]
    # i=0
    # for i in range(len(stock)):
    #     date_time.append(datetime.datetime(year=int(str(stock['dates'].values[i])[:4]),
    #                                            month=int(str(stock['dates'].values[i])[4:6]),
    #                                            day=int(str(stock['dates'].values[i])[6:])))
    # #datetime으로 날자 조합
    # stock['date_time']=date_time


    #1달을 제외하고 나머진 train
    train_data = stock[:-testing_data_len+1]
    # Split the data into x_train and y_train data sets
    print(train_data)
    #380*2=760 = 2일

    x_train=[]
    x_test=[]
    y_train=[]
    y_test=[]




    i=0
    for i in range(len(train_data)-days):
        # if i == 0:
        #     x_train = np.array(stock[i:days+i])
        #     y_train = np.array(stock[i+days:i+days+1])
        #     continue
        x_train.append(stock[i:days+i][['pct','5days_rolling','10days_rolling','max_min','week_day','month','vols']])
        y_train.append(stock[i+days:i+days+1]['pct'])
        if i <= 5:
            print(x_train)
            print(y_train)
            print()

    # Convert the x_train and y_train to numpy arrays



    # x_train = pd.DataFrame({'5days':x_train['5days_rolling'],'10days':x_train['10days_rolling'],
    #                     'max_min':x_train['max_min'],'pct':x_train['pct'],
    #                     'week_day':x_train['week_day']})
    # y_train = pd.DataFrame({'closes':y_train['closes']})
    # Reshape the data


    # print(x_train.shape)

    # Create the testing data set
    # Create a new array containing scaled values from index 1543 to 2002
    test_data = stock[-testing_data_len:]
    # Create the data sets x_test and y_test
    i=0
    for i in range(len(test_data)):
        # if i == 0:
        #     x_test = np.array(test_data[i:days+i])
        #     y_test = np.array(test_data[i+days:i+days+1])
        #     continue
        x_test.append(stock[-testing_data_len+i-days:-testing_data_len+i][['pct','5days_rolling','10days_rolling','max_min','week_day','month','vols']])
        y_test.append(stock[-testing_data_len+i:-testing_data_len+i+1]['pct'])

    # Convert the data to a numpy array
    # x_test = pd.DataFrame({'5days':x_test['5days_rolling'],'10days':x_test['10days_rolling'],
    #                     'max_min':x_test['max_min'],'pct':x_test['pct'],
    #                     'week_day':x_test['week_day']})
    # y_test = pd.DataFrame({'closes':y_test['closes']})

    # x_train.to_csv('x_train.csv', index=False,encoding='utf-8')
    # x_test.to_csv('x_test.csv', index=False,encoding='utf-8')
    # y_train.to_csv('y_train.csv', index=False,encoding='utf-8')
    # y_test.to_csv('y_test.csv', index=False,encoding='utf-8')





    # y_test = pd.read_csv('y_test.csv')
    # print(x_train.shape)
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_test = np.array(x_test)
    x_train = x_train.tolist()
    y_train = y_train.tolist()
    x_test = x_test.tolist()
    # y_test = y_test.tolist()

    # with open(f'y_test.json', "w") as file_write:
    #     json.dump(y_test, file_write)


        # file_path = f"{i}.json" ## your path variable
        # json.dump(i, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
        # json.dumps(i, fp.write(file_path)])

    # print('DataFrame')
    # x_train = pd.DataFrame(x_train,columns=['index','standard','data'])
    # y_train = pd.DataFrame(y_train,columns=['data'])
    # x_test = pd.DataFrame(x_test,columns=['index','standard','data'])
    # print('to_json')
    # x_train.to_json("x_train_day.json", orient = 'table')
    # y_train.to_json("y_train_day.json", orient = 'table')
    # x_test.to_json("x_test_day.json", orient = 'table')
    print('to_json end')
    x_train=np.array(x_train)
    y_train=np.array(y_train)
    x_test=np.array(x_test)
    # y_test=np.array(y_test)

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 7))
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 7))


    from keras.models import Sequential
    from keras.layers import Dense, LSTM
    from keras.callbacks import ModelCheckpoint
    from keras.models import load_model
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(x_train.shape[1], return_sequences=True, input_shape= (x_train.shape[1], 7)))
    model.add(LSTM(x_train.shape[1], return_sequences=False))
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error',metrics=['acc'])
    mc = ModelCheckpoint(f'{code.values[fetch]}.h5', monitor='acc', mode='max', verbose=1, save_best_only=True)# Train the model
    history=model.fit(x_train, y_train, batch_size=10, epochs=20,callbacks=mc)#,validation_data=([x_test,y_test]) #y_test에서 오류남
    # history=model.fit(x_train, y_train, batch_size=10, epochs=20)

    predictions = model.predict(x_test) # numpy serise
    predictions = pd.DataFrame(predictions)
    print(predictions.iloc[1])
    print(type(predictions))
    def un_day_pct_change(data):
        un_pct = []
        for i in range(testing_data_len): # testing_data_len = 60
            # print(data.iloc[i]) #iloc   stock.iloc[-testing_data_len+i,2]
            un_pct.append((data.iloc[i]/100*stock.iloc[-testing_data_len+i,2])+stock.iloc[-testing_data_len+i,2])
            # un_pct.append((data.iloc[i]/100*stock['opens'][-testing_data_len+i:-testing_data_len+i+1].values)+stock['opens'][-testing_data_len+i:-testing_data_len+i+1].values)
            print(un_pct)
        return un_pct

    predictions = un_day_pct_change(predictions)
    predictions= pd.DataFrame(predictions)
    # predictions = scaler.inverse_transform(predictions)
    #
    # # Get the root mean squared error (RMSE)
    # rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    # rmse

    # print(type(predictions))
    # train = stock[:-testing_data_len]
    # train.to_csv('train123123.csv')
    # valid = stock[-testing_data_len:]
    # valid['Predictions'] = predictions.copy()
    # valid.to_csv('vrain123123.csv')
    # # Visualize the data
    # plt.figure(figsize=(16,6))
    # plt.title('Model')
    # plt.xlabel('date_time', fontsize=18)
    # plt.ylabel('closes', fontsize=18)

    # a = stock[['date_time','closes','5days_rolling','10days_rolling']]
    # a = a[-testing_data_len-100:]
    # a['closes'][-testing_data_len:]=None
    # a['val']=None
    # a['pro']=None
    # print(a.iloc[-testing_data_len+i])
    #
    # for i in range(len(test_data)):
    #     a.iloc[-testing_data_len+i,4] = test_data.iloc[i,5] #val <= closes
    #     a.iloc[-testing_data_len + i, 5] = predictions.iloc[i] #pro <= predictions
    # plt.plot(a['date_time'],a[['closes','val','pro','5days_rolling','10days_rolling']])
    # plt.legend(['Train', 'Val', 'Predictions','5days','10days'], loc='lower right')
    # plt.show()
    #
    #
    # exit()
    # plt.plot(train['date_time'],train['closes'])
    # plt.plot(train['date_time'],valid['closes'])
    # plt.plot(train['date_time'],valid['Predictions'])
    # plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    # plt.show()
    #
    #
    # loaded_model = load_model('best_model_day.h5')
    #
    # print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate( [x_test],y_test)[1]))
    # fig, loss_ax = plt.subplots()
    # acc_ax = loss_ax.twinx()
    #
    # loss_ax.plot(history.history['loss'], 'y', label='train loss')
    # loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
    #
    # acc_ax.plot(history.history['acc'], 'g', label='train acc')
    # acc_ax.plot(history.history['val_acc'], 'b', label='val acc')
    #
    # loss_ax.set_xlabel('epoch')
    # loss_ax.set_ylabel('loss')
    # acc_ax.set_ylabel('accuracy')
    #
    # loss_ax.legend(loc='upper left')
    # acc_ax.legend(loc='lower left')
    # plt.show()


