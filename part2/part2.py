from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN

# Create model
def create_rnn_model(stateful):
	##### YOUR MODEL GOES HERE #####
	return model

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
rnn_stateful_rmse_list=list()
rnn_stateless_rmse_list=list()

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
	print('Creating Stateful Vanilla RNN Model...')
	model_rnn_stateful = create_rnn_model(stateful=True)

	# Train the model
	print('Training')
	for i in range(epochs):
		print('Epoch', i + 1, '/', epochs)
		# Note that the last state for sample i in a batch will
		# be used as initial state for sample i in the next batch.
		
		##### TRAIN YOUR MODEL #####
		# model_rnn_stateful.fit()

		# reset states at the end of each epoch
		model_rnn_stateful.reset_states()


	# Plot and save loss curves of training and test set vs iteration in the same graph
	##### PLOT AND SAVE LOSS CURVES #####

	# Save your model weights with following convention:
	# For example length 1 input sequences model filename
	# rnn_stateful_model_weights_length_1.h5
	##### SAVE MODEL WEIGHTS #####
	# filename = ''
	# model_rnn_stateful.save_weights()

	# Predict 
	print('Predicting')
	##### PREDICT #####
	# predicted_rnn_stateful = model_rnn_stateful.predict()

	##### CALCULATE RMSE #####
	# rnn_stateful_rmse = 
	rnn_stateful_rmse_list.append(rnn_stateful_rmse)

	# print('tsteps:{}'.format(tsteps))
	print('length:{}'.format(length))
	print('Stateful Vanilla RNN RMSE:{}'.format( rnn_stateful_rmse ))



	# Create the stateless model
	print('Creating stateless Vanilla RNN Model...')
	model_rnn_stateless = create_rnn_model(stateful=False)

	# Train the model
	print('Training')
	##### TRAIN YOUR MODEL #####
	# model_rnn_stateless.fit()


	# Plot and save loss curves of training and test set vs iteration in the same graph
	##### PLOT AND SAVE LOSS CURVES #####

	# Save your model weights with following convention:
	# For example length 1 input sequences model filename
	# rnn_stateless_model_weights_length_1.h5
	##### SAVE MODEL WEIGHTS #####
	# filename = ''
	# model_rnn_stateless.save_weights()

	# Predict 
	print('Predicting')
	##### PREDICT #####
	# predicted_rnn_stateless = model_rnn_stateless.predict()

	##### CALCULATE RMSE #####
	# rnn_stateless_rmse = 
	rnn_stateless_rmse_list.append(rnn_stateless_rmse)

	# print('tsteps:{}'.format(tsteps))
	print('length:{}'.format(length))
	print('Stateless Vanilla RNN RMSE:{}'.format( rnn_stateless_rmse ))


# save your rmse values for different length input sequence models - stateful rnn:
filename = 'rnn_stateful_model_rmse_values.txt'
np.savetxt(filename, np.array(rnn_stateful_rmse_list), fmt='%.6f', delimiter='\t')

# save your rmse values for different length input sequence models - stateless rnn:
filename = 'rnn_stateless_model_rmse_values.txt'
np.savetxt(filename, np.array(rnn_stateless_rmse_list), fmt='%.6f', delimiter='\t')

print("#" * 33)
print('Plotting Results')
print("#" * 33)

# Plot and save rmse vs Input Length
plt.figure()
plt.plot( np.arange(min_length,max_length+1), rnn_stateful_rmse_list, c='blue', label='Stateful RNN')
plt.plot( np.arange(min_length,max_length+1), rnn_stateless_rmse_list, c='cyan', label='Stateless RNN')
plt.title('RMSE vs Input Length in Test Set')
plt.xlabel('length of input sequences')
plt.ylabel('rmse')
plt.legend()
plt.grid()
plt.show()


