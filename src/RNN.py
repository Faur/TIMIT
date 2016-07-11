print('\n\n * Imporing Libaries')
import os
import time; program_start_time = time.time()
import sys
from six.moves import cPickle

import theano
import theano.tensor as T
import lasagne
import lasagne.layers as L
import numpy as np

from general_tools import *
from RNN_tools import *


##### SCRIPT META VARIABLES #####
print(' * Setting up ...')

comput_confusion = False
	# TODO: ATM this is not implemented


paths = path_reader('path_toke.txt')
output_path = os.path.join('..', 'output')

if 0:
	data_path = os.path.join(paths[0], 'logfbank_39.pkl')
	model_save = os.path.join(output_path, 'best_model_old_data')
	model_load = os.path.join(output_path, 'best_model_old_data.npz')
	INPUT_SIZE 		= 39
elif 0:
	data_path = os.path.join(paths[0], 'std_preprocess_26_ch.pkl')
	model_load = os.path.join(output_path, 'best_model.npz')
	model_save = os.path.join(output_path, 'best_model')
	INPUT_SIZE 		= 26
else:
	data_path = os.path.join(paths[0], 'std_preprocess_26_ch_DEBUG.pkl')
	model_load = os.path.join(output_path, 'best_model_DEBUG.npz')
	model_save = os.path.join(output_path, 'best_model_DEBUG')
	INPUT_SIZE 		= 26
	print('DEBUG MODE ACTIVE: Only a reduced dataset is used.')

##### SCRIPT VARIABLES #####
num_epochs 		= 3

NUM_OUTPUT_UNITS= 61
N_HIDDEN 		= 275

LEARNING_RATE 	= 1e-5
MOMENTUM 		= 0.9
WEIGHT_INIT 	= 0.1
batch_size		= 1


##### IMPORTIN DATA #####
print('\tdata source: '+ data_path)
print('\tmodel target: '+ model_save + '.npz')
dataset = load_dataset(data_path)
X_train, y_train, X_val, y_val, X_test, y_test = dataset

##### BUIDING MODEL #####
print(' * Building network ...')
RNN_network = NeuralNetwork('RNN', batch_size=batch_size, input_size=INPUT_SIZE, n_hidden=N_HIDDEN, 
	num_output_units=NUM_OUTPUT_UNITS, seed=int(time.time()), debug=False)

RNN_network.load_model(model_load)

##### BUIDING FUNCTION #####
print(" * Compiling functions ...")
RNN_network.build_functions(LEARNING_RATE=LEARNING_RATE, MOMENTUM=MOMENTUM, debug=False)


##### TRAINING #####
print(" * Training ...")
RNN_network.train(dataset, model_save, num_epochs=num_epochs, 
	batch_size=batch_size, comput_confusion=False, debug=False)

print()
print(" * Done")
print()
print('Total time: {:.3f}'.format(time.time() - program_start_time))










