from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN
from keras import backend as K
from itertools import islice
from keras import optimizers
from sklearn.metrics import mean_squared_error
from math import sqrt




# Create model
def create_fc_model(inputLayerSize):
    ##### YOUR MODEL GOES HERE #####
    model = Sequential()
    model.add(Dense(20, input_dim=inputLayerSize, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[root_mean_squared_error])
    return model

# Returns a sliding window (of width n) over data from the iterable
def window(seq, n=2):
    it = iter(seq)
    result = list(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + [elem]
        yield result

def returnMovinWindowArray(inputData,targetData,windowsize):
        X=[]
        Y=[]
        X = list(window(inputData, n=windowsize))
        Y = np.array(targetData[(windowsize - 1):len(targetData)], dtype=float)
        return pd.DataFrame(np.array(np.flip(X, axis=1), dtype=float)),pd.DataFrame(Y)

#plot the model history
def plot_model_loss(model_history):
    plt.plot(model_history.history['loss'])
    plt.title('model loss')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig('/Users/mac/Downloads/part.png')

    plt.show()


#plot the model history
def plot_model_rmse(model_history):
    plt.plot(model_history.history['root_mean_squared_error'])
    plt.title('model root_mean_squared_error')
    plt.ylabel('rmse')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig('/Users/mac/Downloads/part.png')

    plt.show()
# split train/test data
def split_data(x, y, ratio=0.8):
    to_train = int(len(x.index) * ratio)
    # tweak to match with batch_size
    to_train -= to_train % batch_size

    x_train = x[:to_train]
    y_train = y[:to_train]
    x_test = x[to_train:]
    y_test = y[to_train:]

    # tweak to match with batch_size
    to_drop = x.shape[0] % batch_size
    if to_drop > 0:
        x_test = x_test[:-1 * to_drop]
        y_test = y_test[:-1 * to_drop]

    # some reshaping
    ##### RESHAPE YOUR DATA BASED ON YOUR MODEL #####

    return (x_train, y_train), (x_test, y_test)


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def root_mean_squared_error_layer(y_true, y_pred):
    rms = sqrt(mean_squared_error(np.array(y_true,dtype=np.float), np.array(y_pred,dtype=np.float)))
    return rms


# training parameters passed to "model.fit(...)"
batch_size = 1
epochs = 10

# The input sequence min and max length that the model is trained on for each output point
min_length = 1
max_length = 10

# load data from files
noisy_data = np.loadtxt('/Users/mac/Downloads/CS5242/assignment_03/filter_data/noisy_data.txt', delimiter='\t',
                        dtype=np.float)
smooth_data = np.loadtxt('/Users/mac/Downloads/CS5242/assignment_03/filter_data/smooth_data.txt', delimiter='\t',
                         dtype=np.float)

print('noisy_data shape:{}'.format(noisy_data.shape))
print('smooth_data shape:{}'.format(smooth_data.shape))
print('noisy_data first 5 data points:{}'.format(noisy_data[:5]))
print('smooth_data first 5 data points:{}'.format(smooth_data[:5]))

# List to keep track of root mean square error for different length input sequences
fc_rmse_list = list()

for num_input in range(min_length, max_length + 1):
    print(str(num_input) + "is the input")

    length = num_input

    print("*" * 33)
    print("INPUT DIMENSION:{}".format(length))
    print("*" * 33)

    # convert numpy arrays to pandas dataframe
    data_input = pd.DataFrame(noisy_data)
    expected_output = pd.DataFrame(smooth_data)

    # when length > 1, arrange input sequences
    if length > 1:
        print(length)
        data_input,expected_output=returnMovinWindowArray(noisy_data,smooth_data,length)

    print(data_input.shape)
    ##### ARRANGE YOUR DATA SEQUENCES #####

    print('data_input length:{}'.format(len(data_input.index)))

    # Split training and test data: use first 80% of data points as training and remaining as test
    (x_train, y_train), (x_test, y_test) = split_data(data_input, expected_output)
    print('x_train.shape: ', x_train.shape)
    print('y_train.shape: ', y_train.shape)
    print('x_test.shape: ', x_test.shape)
    print('y_test.shape: ', y_test.shape)

    print('Input shape:', data_input.shape)
    print('Output shape:', expected_output.shape)
    print('Input head: ')
    print(data_input.head())
    print('Output head: ')
    print(expected_output.head())
    print('Input tail: ')
    print(data_input.tail())
    print('Output tail: ')
    print(expected_output.tail())

    # Create the model
    print('Creating Fully-Connected Model...')
    model_fc = create_fc_model(num_input)

    # Train the model
    print('Training')
    ##### TRAIN YOUR MODEL #####
    # model_fc.fit()
    model_info=model_fc.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=1, epochs=10, verbose=1)

    # Plot and save loss curves of training and test set vs iteration in the same graph
    ##### PLOT AND SAVE LOSS CURVES #####
    plot_model_loss(model_info)
    plot_model_rmse(model_info)

    # Save your model weights with following convention:
    # For example length 1 input sequences model filename
    # fc_model_weights_length_1.h5
    ##### SAVE MODEL WEIGHTS #####
    filename = '/Users/mac/Downloads/CS5242/assignment_03/part1/fc_model_weights_length_'+str(num_input)+'.h5'
    model_fc.save_weights(filename)

    # Predict
    print('Predicting')
    ##### PREDICT #####
    predicted_fc = model_fc.predict(x_test)
    print(predicted_fc.shape," is the predicted_fc")
    print(y_test.shape," is the y_test")


    ##### CALCULATE RMSE #####
    fc_rmse = root_mean_squared_error_layer(y_test,predicted_fc)
    print(fc_rmse , " is the fc_rmse")
    fc_rmse_list.append(fc_rmse)

    # print('tsteps:{}'.format(tsteps))
    #print('length:{}'.format(length))
    #print('Fully-Connected RMSE:{}'.format(fc_rmse))
    #model_fc.predict()


# save your rmse values for different length input sequence models:
filename = 'fc_model_rmse_values.txt'
np.savetxt(filename, np.array(fc_rmse_list), fmt='%.6f', delimiter='\t')
print(fc_rmse_list)

print("#" * 33)
print('Plotting Results')
print("#" * 33)

# Plot and save rmse vs Input Length
plt.figure()
plt.plot(np.arange(min_length, max_length + 1), fc_rmse_list, c='black', label='FC')
plt.title('RMSE vs Input Length in Test Set')
plt.xlabel('length of input sequences')
plt.ylabel('rmse')
plt.legend()
plt.grid()
plt.show()
plt.savefig()


