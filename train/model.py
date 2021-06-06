from preprocess import train_test_label, delet_and_cluster
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
def main():
    all_data = delet_and_cluster.del_and_cluster.url_content
    all_data_train, all_data_test = train_test_label.train_test(all_data)
    url_label0, url_label1, url_label2, url_label3, url_label4, url_label5, url_label6, url_label7, url_label8, url_label9 = train_test_label.label_split(all_data_train)
    url_label0['content'] = train_test_label.text_except_all(url_label0)
    url_label1['content'] = train_test_label.text_except_all(url_label1)
    url_label2['content'] = train_test_label.text_except_all(url_label2)
    url_label3['content'] = train_test_label.text_except_all(url_label3)
    url_label4['content'] = train_test_label.text_except_all(url_label4)
    url_label5['content'] = train_test_label.text_except_all(url_label5)
    url_label6['content'] = train_test_label.text_except_all(url_label6)
    url_label7['content'] = train_test_label.text_except_all(url_label7)
    url_label8['content'] = train_test_label.text_except_all(url_label8)
    url_label9['content'] = train_test_label.text_except_all(url_label9)
    print('text-except')
    X_train2_label0, y_train_label0 = train_test_label.pad(url_label0)
    X_train2_label1, y_train_label1 = train_test_label.pad(url_label1)
    X_train2_label2, y_train_label2 = train_test_label.pad(url_label2)
    X_train2_label3, y_train_label3 = train_test_label.pad(url_label3)
    X_train2_label4, y_train_label4 = train_test_label.pad(url_label4)
    X_train2_label5, y_train_label5 = train_test_label.pad(url_label5)
    X_train2_label6, y_train_label6 = train_test_label.pad(url_label6)
    X_train2_label7, y_train_label7 = train_test_label.pad(url_label7)
    X_train2_label8, y_train_label8 = train_test_label.pad(url_label8)
    X_train2_label9, y_train_label9 = train_test_label.pad(url_label9)
    print('label.pad')
    url_label0,url_label1,url_label2,url_label3,url_label4,url_label5,url_label6,url_label7,url_label8,url_label9 = train_test_label.label_split(all_data_test)
    #train_url_label0,train_url_label1,train_url_label2,train_url_label3
    # 불용어 처리
    url_label0['content']=train_test_label.text_except_all(url_label0)
    url_label1['content']=train_test_label.text_except_all(url_label1)
    url_label2['content']=train_test_label.text_except_all(url_label2)
    url_label3['content']=train_test_label.text_except_all(url_label3)
    url_label4['content']=train_test_label.text_except_all(url_label4)
    url_label5['content']=train_test_label.text_except_all(url_label5)
    url_label6['content']=train_test_label.text_except_all(url_label6)
    url_label7['content']=train_test_label.text_except_all(url_label7)
    url_label8['content']=train_test_label.text_except_all(url_label8)
    url_label9['content']=train_test_label.text_except_all(url_label9)
    # 패딩
    X_test2_label0, y_test_label0 = train_test_label.pad(url_label0)
    X_test2_label1, y_test_label1 = train_test_label.pad(url_label1)
    X_test2_label2, y_test_label2 = train_test_label.pad(url_label2)
    X_test2_label3, y_test_label3 = train_test_label.pad(url_label3)
    X_test2_label4, y_test_label4 = train_test_label.pad(url_label4)
    X_test2_label5, y_test_label5 = train_test_label.pad(url_label5)
    X_test2_label6, y_test_label6 = train_test_label.pad(url_label6)
    X_test2_label7, y_test_label7 = train_test_label.pad(url_label7)
    X_test2_label8, y_test_label8 = train_test_label.pad(url_label8)
    X_test2_label9, y_test_label9 = train_test_label.pad(url_label9)
    print('test_clear')
    from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Bidirectional, concatenate
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.models import load_model
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    # In[134]:


    # Define the Keras TensorBoard callback.

    from datetime import datetime

    logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    # In[129]:


    # In[155]:




    # In[148]:


    def buildModel(length, label):
        label0_input = Input(shape=(1, length), name='label0')
        label1_input = Input(shape=(1, length), name='label1')
        label2_input = Input(shape=(1, length), name='label2')
        label3_input = Input(shape=(1, length), name='label3')
        label4_input = Input(shape=(1, length), name='label4')
        label5_input = Input(shape=(1, length), name='label5')
        label6_input = Input(shape=(1, length), name='label6')
        label7_input = Input(shape=(1, length), name='label7')
        label8_input = Input(shape=(1, length), name='label8')
        label9_input = Input(shape=(1, length), name='label9')

        label0_layer = LSTM(length, return_sequences=True, activation='relu')(label0_input)
        label1_layer = LSTM(length, return_sequences=True, activation='relu')(label1_input)
        label2_layer = LSTM(length, return_sequences=True, activation='relu')(label2_input)
        label3_layer = LSTM(length, return_sequences=True, activation='relu')(label3_input)
        label4_layer = LSTM(length, return_sequences=True, activation='relu')(label4_input)
        label5_layer = LSTM(length, return_sequences=True, activation='relu')(label5_input)
        label6_layer = LSTM(length, return_sequences=True, activation='relu')(label6_input)
        label7_layer = LSTM(length, return_sequences=True, activation='relu')(label7_input)
        label8_layer = LSTM(length, return_sequences=True, activation='relu')(label8_input)
        label9_layer = LSTM(length, return_sequences=True, activation='relu')(label9_input)

        label0_layer = Dropout(0.3)(label0_layer)
        label0_layer = Dense(length, activation='relu')(label0_layer)
        # label0_layer = Dropout(0.3)(label0_layer)
        # label0_layer = Dense(length, activation='relu')(label0_layer)

        label1_layer = Dropout(0.3)(label1_layer)
        label1_layer = Dense(length, activation='relu')(label1_layer)
        # label1_layer = Dropout(0.3)(label1_layer)
        # label1_layer = Dense(length, activation='relu')(label1_layer)

        label2_layer = Dropout(0.3)(label2_layer)
        label2_layer = Dense(length, activation='relu')(label2_layer)
        # label2_layer = Dropout(0.3)(label2_layer)
        # label2_layer = Dense(length, activation='relu')(label2_layer)

        label3_layer = Dropout(0.3)(label3_layer)
        label3_layer = Dense(length, activation='relu')(label3_layer)
        # label3_layer = Dropout(0.3)(label3_layer)
        # label3_layer = Dense(length, activation='relu')(label3_layer)

        label4_layer = Dropout(0.3)(label4_layer)
        label4_layer = Dense(length, activation='relu')(label4_layer)
        # label4_layer = Dropout(0.3)(label4_layer)
        # label4_layer = Dense(length, activation='relu')(label4_layer)

        label5_layer = Dropout(0.3)(label5_layer)
        label5_layer = Dense(length, activation='relu')(label5_layer)
        # label5_layer = Dropout(0.3)(label5_layer)
        # label5_layer = Dense(length, activation='relu')(label5_layer)

        label6_layer = Dropout(0.3)(label6_layer)
        label6_layer = Dense(length, activation='relu')(label6_layer)
        # label6_layer = Dropout(0.3)(label6_layer)
        # label6_layer = Dense(length, activation='relu')(label6_layer)

        label7_layer = Dropout(0.3)(label7_layer)
        label7_layer = Dense(length, activation='relu')(label7_layer)
        # label7_layer = Dropout(0.3)(label7_layer)
        # label7_layer = Dense(length, activation='relu')(label7_layer)

        label8_layer = Dropout(0.3)(label8_layer)
        label8_layer = Dense(length, activation='relu')(label8_layer)
        # label8_layer = Dropout(0.3)(label8_layer)
        # label8_layer = Dense(length, activation='relu')(label8_layer)

        label9_layer = Dropout(0.3)(label9_layer)
        label9_layer = Dense(length, activation='relu')(label9_layer)
        # label9_layer = Dropout(0.3)(label9_layer)
        # label9_layer = Dense(length, activation='relu')(label9_layer)

        output = concatenate(
            [
                label0_layer,
                label1_layer,
                label2_layer,
                label3_layer,
                label4_layer,
                label5_layer,
                label6_layer,
                label7_layer,
                label8_layer,
                label9_layer
            ]
        )
        output = LSTM(length, return_sequences=True, activation='relu')(output)
        output = Dense(length, activation='relu')(output)
        # output = Dense(label, activation='softmax')(output)
        output = Dense(label, activation='sigmoid')(output)
        model = tf.keras.Model(
            inputs=
            [
                label0_input,
                label1_input,
                label2_input,
                label3_input,
                label4_input,
                label5_input,
                label6_input,
                label7_input,
                label8_input,
                label9_input,
            ],
            outputs=
            [
                output
            ]
        )

        # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
        #               metrics=['acc', 'sparse_categorical_accuracy'])
        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['acc'])
        return model


    # In[149]:


    # max_len 패딩시 사요한 기준
    max_len = 400
    y_train = pd.DataFrame(all_data_train['per'])
    y_train = np.array(y_train).reshape(y_train.shape[0], 1, y_train.shape[1])
    y_test = pd.DataFrame(all_data_test['per'])
    y_test = np.array(y_test).reshape(y_test.shape[0], 1, y_test.shape[1])
    # lstm = buildModel(max_len, 4)
    lstm = buildModel(max_len, 1)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    history = lstm.fit(
        [
            X_train2_label0,
            X_train2_label1,
            X_train2_label2,
            X_train2_label3,
            X_train2_label4,
            X_train2_label5,
            X_train2_label6,
            X_train2_label7,
            X_train2_label8,
            X_train2_label9
        ],

        y_train
        ,
        epochs=100,
        callbacks=[es, mc, tensorboard_callback],
        batch_size=20,
        validation_split=0.2,
        validation_data=(
            [
                X_test2_label0,
                X_test2_label1,
                X_test2_label2,
                X_test2_label3,
                X_test2_label4,
                X_test2_label5,
                X_test2_label6,
                X_test2_label7,
                X_test2_label8,
                X_test2_label9
            ],

            y_test

        )
    )

    # In[150]:


    # get_ipython().run_line_magic('load_ext', 'tensorboard')
    # get_ipython().run_line_magic('tensorboard', '--logdir logs')

    # In[151]:


    lstm.summary()

    loaded_model = load_model('best_model.h5')

    print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate([
        X_test2_label0,
        X_test2_label1,
        X_test2_label2,
        X_test2_label3,
        X_test2_label4,
        X_test2_label5,
        X_test2_label6,
        X_test2_label7,
        X_test2_label8,
        X_test2_label9
    ],
        y_test)[1]))
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




if __name__ == "__main__":
    main()