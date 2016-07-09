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

debug = True
comput_confusion = False
	# TODO: ATM this is not implemented


paths = path_reader('path_toke.txt')
output_path = os.path.join('..', 'output')
model_save = os.path.join(output_path, 'best_model_old_data')
# model_load = os.path.join(output_path, 'best_model.npz')
model_load = None
# data_path = os.path.join(paths[0], 'std_preprocess_26_ch.pkl')
data_path = os.path.join(paths[0], 'logfbank_39.pkl')
# data_path = os.path.join(paths[0], 'std_preprocess_26_ch_DEBUG.pkl')


##### SCRIPT VARIABLES #####
num_epochs 		= 500

INPUT_SIZE 		= 39
INPUT_SIZE 		= 26
NUM_OUTPUT_UNITS= 61
N_HIDDEN 		= 275

LEARNING_RATE 	= 1e-4
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
network = build_RNN(batch_size=batch_size, input_size=INPUT_SIZE, n_hidden=N_HIDDEN, 
	num_output_units=NUM_OUTPUT_UNITS, seed=int(time.time()), debug=False)

if model_load:
	load_model(model_load, network)

##### BUIDING FUNCTION #####
print(" * Compiling functions ...")
# output_fn, argmax_fn, accuracy_fn, train_fn, validate_fn = build_functions(network)
training_fn = build_functions(network, LEARNING_RATE=LEARNING_RATE, MOMENTUM=MOMENTUM, debug=False)
output_fn, argmax_fn, accuracy_fn, train_fn, validate_fn = training_fn


##### TRAINING #####
print(" * Training ...")
train(network, dataset, training_fn, model_save, num_epochs=num_epochs, 
	batch_size=batch_size, comput_confusion=False, debug=False)

print()
print(" * Done")
print()
print('Total time: {:.3f}'.format(time.time() - program_start_time))










