# LSTM example for sine wave and stocks from blog:
# http://www.jakob-aungiers.com/articles/a/LSTM-Neural-Network-for-Time-Series-Prediction
# base code from:
# https://github.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction

import lstm
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

def plot_results(predicted_data, true_data, figtitle):
    ''' use when predicting just one analysis window '''
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.title(figtitle)
    plt.show()

def plot_results_multiple(predicted_data, true_data, prediction_len):
    ''' use when predicting multiple analyses windows in data '''
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        if i != 0:
            padding = [None for p in range(i * prediction_len)]
            plt.plot(padding + data, label='Prediction')
            plt.legend()
    plt.show()

#Main Run Thread
if __name__=='__main__':
    global_start_time = time.time()
    # run with values below first to see how long it takes on your machine
    epochs  = 10  # suggest 100 for sine wave, 10 for stock
    seq_len = 20 # suggest using 25 for sine wave, 50 for stock

    print('> Loading data... ')
    # load data
    X = pickle.load( open( "aspen_X_nosummer.p", "rb" ) )
    y = pickle.load( open( "aspen_y_nosummer.p", "rb" ) )

    # train test split in time
    splitdate = pd.to_datetime('2016-06-01')
    X_train = X[X.index <= splitdate]
    X_test = X[X.index > splitdate]
    y_train = y[y.index <= splitdate]
    y_test = y[y.index > splitdate]

    print('> Data Loaded. Compiling...')

    model = lstm.build_model([1, seq_len, 100, 1]) # 1 input layer, layer 1 has seq_len neurons, layer 2 has 100 neurons, 1 output

    model.fit(
        X_train,
        y_train,
        batch_size=512,
        nb_epoch=epochs,
        validation_split=0.05)

    print('> Completed.')
    print('Training duration (s) : ', time.time() - global_start_time)

    # comment out either sine wave prediction code or stock prediction code
    # sine wave code
    predicted = lstm.predict_point_by_point(model, X_test)
    plot_results(predicted, y_test, 'Sine wave - predict one point ahead')
    predicted_full = lstm.predict_sequence_full(model, X_test, seq_len)
    plot_results(predicted_full, y_test, 'Sine wave - predict full sequence from start seed')

    # stock prediction code
    #predictions = lstm.predict_sequences_multiple(model, X_test, seq_len, seq_len) #model, data, window_size, prediction length)
    #plot_results_multiple(predictions, y_test, seq_len) # prediction, true data, prediction length)
