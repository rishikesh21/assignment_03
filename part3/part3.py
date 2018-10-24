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
from keras import optimizers
from numpy import zeros, newaxis


# Create model
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

# plot the model history
def plot_model_loss(model_history, length, mode):
    plt.plot(model_history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('../plots/'+mode+'/' + mode + '_loss_length_' + str(length) + '.png')


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
	x_train = x_train.values[:, :, newaxis]
	# y_train=y_train.values.reshape((y_train.shape[0]))
	x_test = x_test.values[:, :, newaxis]
	# y_test=y_test.values.reshape((y_test.shape[0],))

	return (x_train, y_train), (x_test, y_test)

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
# training parameters passed to "model.fit(...)"
batch_size = 1
epochs = 10

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
		print(length)
		data_input, expected_output = returnMovinWindowArray(noisy_data, smooth_data, length)

	print('data_input length:{}'.format(len(data_input.index)) )

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
	
	# Create the stateful model
	print('Creating Stateful LSTM Model...')
	model_lstm_stateful = create_lstm_model(True,length)

	# Train the model
	print('Training')
	for i in range(epochs):
		print('Epoch', i + 1, '/', epochs)
		# Note that the last state for sample i in a batch will
		# be used as initial state for sample i in the next batch.
		
		##### TRAIN YOUR MODEL #####
		model_info_lstm_stateful=model_lstm_stateful.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=1, epochs=10,verbose=1)


		# reset states at the end of each epoch
		model_lstm_stateful.reset_states()


	# Plot and save loss curves of training and test set vs iteration in the same graph
	##### PLOT AND SAVE LOSS CURVES #####
	plot_model_loss(model_info_lstm_stateful, length, "lstm_stateful")

	# Save your model weights with following convention:
	# For example length 1 input sequences model filename
	# lstm_stateful_model_weights_length_1.h5
	##### SAVE MODEL WEIGHTS #####
	filename_lstm_stateful_weight= 'lstm_stateful__model_weights_length_' + str(num_input) + '.h5'
	model_lstm_stateful.save_weights(filename_lstm_stateful_weight)

	# Predict 
	print('Predicting')
	##### PREDICT #####
	predicted_lstm_stateful = model_lstm_stateful.predict(x_test,batch_size=1)

	##### CALCULATE RMSE #####
	lstm_stateful_rmse =root_mean_squared_error_layer(y_test,predicted_lstm_stateful)
	lstm_stateful_rmse_list.append(lstm_stateful_rmse)

	# print('tsteps:{}'.format(tsteps))
	print('length:{}'.format(length))
	print('Stateful LSTM RMSE:{}'.format( lstm_stateful_rmse ))



	# Create the stateless model
	print('Creating stateless LSTM Model...')
	model_lstm_stateless = create_lstm_model(False,length)

	# Train the model
	print('Training')
	##### TRAIN YOUR MODEL #####
	model_info_lstm_stateless=model_lstm_stateless.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=1, epochs=10, verbose=1)

	# Plot and save loss curves of training and test set vs iteration in the same graph
	##### PLOT AND SAVE LOSS CURVES #####
	plot_model_loss(model_info_lstm_stateless, length, "lstm_stateless")


	# Save your model weights with following convention:
	# For example length 1 input sequences model filename
	# lstm_stateless_model_weights_length_1.h5
	##### SAVE MODEL WEIGHTS #####
	filename_lstm_stateless_weight = 'lstm_stateless_model_weights_length_' + str(num_input) + '.h5'
	model_lstm_stateless.save_weights(filename_lstm_stateless_weight)

	# Predict 
	print('Predicting')
	##### PREDICT #####
	predicted_lstm_stateless = model_lstm_stateless.predict(x_test,batch_size=1)

	##### CALCULATE RMSE #####
	lstm_stateless_rmse =root_mean_squared_error_layer(y_test,predicted_lstm_stateless)
	lstm_stateless_rmse_list.append(lstm_stateless_rmse)

	# print('tsteps:{}'.format(tsteps))
	#print('length:{}'.format(length))
	#print('Stateless LSTM RMSE:{}'.format( lstm_stateless_rmse ))


# save your rmse values for different length input sequence models - stateful rnn:
filename_lstm_stateful_rmse = 'lstm_stateful_model_rmse_values.txt'
np.savetxt(filename_lstm_stateful_rmse, np.array(lstm_stateful_rmse_list), fmt='%.6f', delimiter='\t')

# save your rmse values for different length input sequence models - stateless rnn:
filename_lstm_stateless_rmse= 'lstm_stateless_model_rmse_values.txt'
np.savetxt(filename_lstm_stateless_rmse, np.array(lstm_stateless_rmse_list), fmt='%.6f', delimiter='\t')

print("#" * 33)
print('Plotting Results')
print("#" * 33)

# plt.figure()
# plt.plot(data_input[0][:100], '.')
# plt.plot(expected_output[0][:100], '-')
# plt.legend(['Input', 'Expected output'])
# plt.title('Input - First 100 data points')

# Plot and save rmse vs Input Length
plt.figure()
plt.plot( np.arange(min_length,max_length+1), lstm_stateful_rmse_list, c='red', label='Stateful LSTM')
plt.plot( np.arange(min_length,max_length+1), lstm_stateless_rmse_list, c='magenta', label='Stateless LSTM')
plt.title('RMSE vs Input Length in Test Set')
plt.xlabel('length of input sequences')
plt.ylabel('rmse')
plt.legend()
plt.grid()
plt.savefig('../plots/lstm_rmse_length_.png')
plt.show()



