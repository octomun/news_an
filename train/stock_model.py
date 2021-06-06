import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
os.chdir('F:/stock')
stock= pd.read_csv('stock_dropna_sort.csv', encoding='utf-8')
standard=380
testing_data_len=standard*30 #1달
day=2
days=standard*day


#1달을 제외하고 나머진 train
train_data = stock[0:-testing_data_len]
# Split the data into x_train and y_train data sets

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
    x_train.append(stock[i:days+i][['5days_rolling','10days_rolling','max_min','pct','week_day']])
    y_train.append(stock[i+days:i+days+1]['closes'])
    if i <= 5:
        print(x_train)
        print(y_train)
        print()

# Convert the x_train and y_train to numpy arrays



# x_train = pd.DataFrame({'5days':x_train['5days_rolling'],'10days':x_train['10days_rolling'],
#                     'max_min':x_train['max_min'],'pct':x_train['pct'],
#                     'week_day':x_train['week_day']})
# y_train = pd.DataFrame({'closes':y_train['closes']})
print('333333333333333333333333333333333')
# Reshape the data


# print(x_train.shape)

# Create the testing data set
# Create a new array containing scaled values from index 1543 to 2002
test_data = stock[-testing_data_len:]
# Create the data sets x_test and y_test
i=0
for i in range(len(test_data)-days):
    # if i == 0:
    #     x_test = np.array(test_data[i:days+i])
    #     y_test = np.array(test_data[i+days:i+days+1])
    #     continue
    x_test.append(stock[-testing_data_len+i:-testing_data_len+i+days][['5days_rolling','10days_rolling','max_min','pct','week_day']])
    y_test.append(stock[-testing_data_len+i+days:-testing_data_len+i+days+1]['closes'])

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

x_train = pd.DataFrame(x_train,columns=['index','standard','data'])
y_train = pd.DataFrame(y_train,columns=['data'])
x_test = pd.DataFrame(x_test,columns=['index','standard','data'])

x_train.to_json("x_train.json", orient = 'table')
y_train.to_json("y_train.json", orient = 'table')
x_test.to_json("x_test.json", orient = 'table')

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 5))
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 5))

exit()
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 5)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
mc = ModelCheckpoint('best_stock_model.h5',monitor='val_acc', mode='max',verbose=1,save_best_only=True)
# Train the model
history=model.fit(x_train, y_train, batch_size=100, epochs=5,callbacks=mc)

predictions = model.predict(x_test)
# predictions = scaler.inverse_transform(predictions)
#
# # Get the root mean squared error (RMSE)
# rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
# rmse
loaded_model = load_model('best_model.h5')

print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate( [x_test],y_test)[1]))
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(history.history['loss'], 'y', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')

acc_ax.plot(history.history['acc'], 'g', label='train acc')
acc_ax.plot(history.history['val_acc'], 'b', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
plt.show()


train = stock[:testing_data_len]
valid = stock[testing_data_len:]
valid['Predictions'] = predictions
# Visualize the data
plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()