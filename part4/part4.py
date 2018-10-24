from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN,Activation
from keras import backend as K
from itertools import islice
from keras import optimizers
from sklearn.metrics import mean_squared_error
from math import sqrt
from numpy import zeros, newaxis




# Create model
def create_fc_model(inputLayerSize):
	##### YOUR MODEL GOES HERE #####
	model = Sequential()
	model.add(Dense(20, input_dim=inputLayerSize, kernel_initializer='uniform', activation='relu'))
	model.add(Dense(1, kernel_initializer='uniform'))
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=[root_mean_squared_error])
	return model



def create_rnn_model(stateful,length):
	##### YOUR MODEL GOES HERE #####
	model = Sequential()
	model.add(SimpleRNN(20, return_sequences=False, stateful=stateful, batch_input_shape=(1, length, 1)))
	adam = optimizers.Adam(lr=0.001)
	model.add(Dense(1))
	model.compile(loss='mean_absolute_error', optimizer=adam, metrics=[root_mean_squared_error])
	return model


def create_lstm_model(stateful,length):
	##### YOUR MODEL GOES HERE #####
	model = Sequential()
	model.add(LSTM(20, return_sequences=False, stateful=stateful, batch_input_shape=(1, length, 1)))
	adam = optimizers.Adam(lr=0.001)
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer=adam, metrics=[root_mean_squared_error])
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

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def root_mean_squared_error_layer(y_true, y_pred):
    rms = sqrt(mean_squared_error(np.array(y_true,dtype=np.float), np.array(y_pred,dtype=np.float)))
    return rms

# split train/test data
def split_data(x, y, ratio=0.8,flag=0):
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
	if(flag==1):

	 	x_train = x_train.values[:, :, newaxis]
	 	# y_train=y_train.values.reshape((y_train.shape[0]))
	 	x_test = x_test.values[:, :, newaxis]
	 	# y_test=y_test.values.reshape((y_test.shape[0],))

	return (x_train, y_train), (x_test, y_test)

# predicting parameters passed to "model.predict(...)"
batch_size = 1

# The input sequence min and max length that the model is trained on for each output point
min_length = 1
max_length = 10

# load data from files
noisy_data = np.loadtxt('../filter_data/noisy_data.txt', delimiter='\t', dtype=np.float)
smooth_data = np.loadtxt('../filter_data/smooth_data.txt', delimiter='\t', dtype=np.float)

print('noisy_data shape:{}'.format(noisy_data.shape))
print('smooth_data shape:{}'.format(smooth_data.shape))
print('noisy_data first 5 data points:{}'.format(noisy_data[:5]))
print('smooth_data first 5 data points:{}'.format(smooth_data[:5]))


# List to keep track of root mean square error for different length input sequences
fc_rmse_list=list()
rnn_stateful_rmse_list=list()
rnn_stateless_rmse_list=list()
lstm_stateful_rmse_list=list()
lstm_stateless_rmse_list=list()

for num_input in range(min_length,max_length+1):
	length = num_input

	print("*" * 33)
	print("INPUT DIMENSION:{}".format(length))
	print("*" * 33)

	# convert numpy arrays to pandas dataframe
	data_input = pd.DataFrame(noisy_data)
	expected_output = pd.DataFrame(smooth_data)

	# when length > 1, arrange input sequences
	if length > 1:
		##### ARRANGE YOUR DATA SEQUENCES #####
		data_input, expected_output = returnMovinWindowArray(noisy_data, smooth_data, length)

	print('data_input length:{}'.format(len(data_input.index)) )

	# Split training and test data: use first 80% of data points as training and remaining as test
	(x_train, y_train), (x_test, y_test) = split_data(data_input, expected_output,0.8,1)
	(x_train_fc, y_train_fc), (x_test_fc, y_test_fc) = split_data(data_input, expected_output,0.8,0)

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
	
	# Create the models and load trained weights
	print('Creating Fully-Connected Model and Loading Trained Weights...')
	model_fc = create_fc_model(length)
	##### LOAD MODEL WEIGHTS #####
	filename = '../trained_models/fc_model_weights_length_'+str(length)+'_trained.h5'
	model_fc.load_weights(filename)

	print('Creating Stateful Vanilla RNN Model and Loading Trained Weights...')
	model_rnn_stateful = create_rnn_model(True,length)
	##### LOAD MODEL WEIGHTS #####
	filename = '../trained_models/rnn_stateful_model_weights_length_'+str(length)+'_trained.h5'
	model_rnn_stateful.load_weights(filename)

	print('Creating stateless Vanilla RNN Model and Loading Trained Weights...')
	model_rnn_stateless = create_rnn_model(False,length)
	##### LOAD MODEL WEIGHTS #####
	filename = '../trained_models/rnn_stateless_model_weights_length_'+str(length)+'_trained.h5'
	model_rnn_stateless.load_weights(filename)

	print('Creating Stateful LSTM Model and Loading Trained Weights...')
	model_lstm_stateful = create_lstm_model(True,length)
	##### LOAD MODEL WEIGHTS #####
	filename = '../trained_models/lstm_stateful_model_weights_length_'+str(length)+'_trained.h5'
	model_lstm_stateful.load_weights(filename)

	print('Creating stateless LSTM Model and Loading Trained Weights...')
	model_lstm_stateless = create_lstm_model(False,length)
	##### LOAD MODEL WEIGHTS #####
	filename = '../trained_models/lstm_stateless_model_weights_length_'+str(length)+'_trained.h5'
	model_lstm_stateless.load_weights(filename)

	# Predict 
	print('Predicting')
	##### PREDICT #####
	predicted_fc = model_fc.predict(x_test_fc)
	##### CALCULATE RMSE #####
	fc_rmse = root_mean_squared_error_layer(y_test_fc,predicted_fc)
	fc_rmse_list.append(fc_rmse)

	##### PREDICT #####
	predicted_rnn_stateful = model_rnn_stateful.predict(x_test,batch_size=1)
	##### CALCULATE RMSE #####
	rnn_stateful_rmse = root_mean_squared_error_layer(y_test,predicted_rnn_stateful)
	rnn_stateful_rmse_list.append(rnn_stateful_rmse)

	##### PREDICT #####
	predicted_rnn_stateless = model_rnn_stateless.predict(x_test,batch_size=1)
	##### CALCULATE RMSE #####
	rnn_stateless_rmse = root_mean_squared_error_layer(y_test,predicted_rnn_stateless)
	rnn_stateless_rmse_list.append(rnn_stateless_rmse)

	##### PREDICT #####
	predicted_lstm_stateful = model_lstm_stateful.predict(x_test,batch_size=1)
	##### CALCULATE RMSE #####
	lstm_stateful_rmse = root_mean_squared_error_layer(y_test,predicted_lstm_stateful)
	lstm_stateful_rmse_list.append(lstm_stateful_rmse)

	##### PREDICT #####
	predicted_lstm_stateless = model_lstm_stateless.predict(x_test,batch_size=1)
	##### CALCULATE RMSE #####
	lstm_stateless_rmse = root_mean_squared_error_layer(y_test,predicted_lstm_stateless)
	lstm_stateless_rmse_list.append(lstm_stateless_rmse)

	# print('tsteps:{}'.format(tsteps))
	print('length:{}'.format(length))
	print('Fully-Connected RMSE:{}'.format( fc_rmse ))
	print('Stateful Vanilla RNN RMSE:{}'.format( rnn_stateful_rmse ))
	print('Stateless Vanilla RNN RMSE:{}'.format( rnn_stateless_rmse ))
	print('Stateful LSTM RMSE:{}'.format( lstm_stateful_rmse ))
	print('Stateless LSTM RMSE:{}'.format( lstm_stateless_rmse ))


# Save your rmse values for different length input sequence models:
# This file should have 5 rows (one row per model) and
# 10 columns (one column per input length).
# 1st row: fully-connected model
# 2nd row: vanilla rnn stateful
# 3rd row: vanilla rnn stateless
# 4th row: lstm stateful
# 5th row: lstm stateless
filename = 'all_models_rmse_values.txt'
##### PREPARE RMSE ARRAY THAT WILL BE WRITTEN INTO FILE #####
rmse_arr = np.array([fc_rmse_list,rnn_stateful_rmse_list,rnn_stateless_rmse_list,lstm_stateful_rmse_list,lstm_stateless_rmse_list])

np.savetxt(filename, rmse_arr, fmt='%.6f', delimiter='\t')

print("#" * 33)
print('Plotting Results')
print("#" * 33)

plt.figure()
plt.plot(data_input[0][:100], '.')
plt.plot(expected_output[0][:100], '-')
plt.legend(['Input', 'Expected output'])
plt.title('Input - First 100 data points')

# Plot and save rmse vs Input Length
plt.figure()
plt.plot( np.arange(min_length,max_length+1), fc_rmse_list, c='black', label='FC')
plt.plot( np.arange(min_length,max_length+1), rnn_stateful_rmse_list, c='blue', label='Stateful RNN')
plt.plot( np.arange(min_length,max_length+1), rnn_stateless_rmse_list, c='cyan', label='Stateless RNN')
plt.plot( np.arange(min_length,max_length+1), lstm_stateful_rmse_list, c='red', label='Stateful LSTM')
plt.plot( np.arange(min_length,max_length+1), lstm_stateless_rmse_list, c='magenta', label='Stateless LSTM')
plt.title('RMSE vs Input Length in Test Set')
plt.xlabel('length of input sequences')
plt.ylabel('rmse')
plt.legend()
plt.grid()
plt.savefig('../plots/all_rmse_length_.png')
plt.show()


