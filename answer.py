import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from keras import optimizers
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation

def load_data(data_path, hour_prior):
    """
    input:
        data_path: path to training data file
        hour_prior: use how many hours as input x to predict the y( case in our slides:
                    hour_prior=5)
    output:
        X: data of Xs
        Y: data of Ys
    """
    data = pd.read_csv('train.csv')
    data.drop(data.columns[[0, 1]], axis=1, inplace=True)
    data = data.values
    raw = np.ravel(data) #flatten matrix to 1-D array
    total_month = 12
    X = []
    Y = []
    for m in range(total_month):
        for d in range(480-1-hour_prior):
            # x is an array of pm2.5 of [hour1 hour2 hour3 hour4 hour5]
            x = raw[m*480 + d : m*480 + d + hour_prior]
            X.append(x)
            # y is value of pm2.5 of [hour6]
            Y.append(raw[m*480 + d + hour_prior])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y
def plot_result(predicted, Y):
    #given predicted y and true Y from training data
    #save the plot
    fig, ax = plt.subplots()
    ax.scatter(Y, predicted)
    ax.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.savefig('linear_regression')

def train(X_train, Y_train):

    #TODO, 1
    #Define model arch
    feat_dim = X_train.shape[1]
    model = Sequential()
    model.add(Dense(units=1, kernel_initializer='zero', input_dim=feat_dim, use_bias=True))
    # output = dot(input, weight) + bias

    # TODO, 2
    # Define optimizer
    #you can also change the learning rate
    sgd = optimizers.SGD(lr=0.0001)

    # TODO, 3
    # Compile model
    model.compile(loss='mean_squared_error',optimizer=sgd)

    # TODO, 4
    # Start training

    model.fit(X_train, Y_train, batch_size=32, epochs=20, verbose=1)
    model.save('./linear_regression_params/model.h5')

def infer(X_test, Y_test):
    # TODO, 5
    # load and inference
    model = load_model('./linear_regression_params/model.h5')
    predict = model.predict(X_test)
    plot_result(predict, Y_test)
    loss = model.evaluate(X_test, Y_test)

    print('Loss : {}'.format(loss))

def main(opts):
    #create empty directory for saving model
    save_model_path = 'linear_regression_params/'
    if not os.path.isdir(save_model_path):
        os.mkdir(save_model_path)

    # you can change the hours to predict pm2.5
    hour_prior = 5
    #Load data
    X, Y = load_data(opts.data_path, hour_prior)
    #split data into training and testing
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    print('Training:')
    train(X_train, Y_train)
    print('Test: ')
    infer(X_test, Y_test)

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='Linear Regression with Gradient Descent Method')
    group = parser.add_mutually_exclusive_group()
    parser.add_argument('--data_path', type=str, default='train.csv', help='path to data')
    opts = parser.parse_args()
    main(opts)









