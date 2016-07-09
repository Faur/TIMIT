import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as L
import time

from six.moves import cPickle


def iterate_minibatches(inputs, targets, batch_size, shuffle=False):
	"""
	Helper function that returns an iterator over the training data of a particular
	size, optionally in a random order.

	For big data sets you can load numpy arrays as memory-mapped files
		(numpy.load(..., mmap_mode='r'))

	This function a slight modification of:
		http://lasagne.readthedocs.org/en/latest/user/tutorial.html
	"""
	assert len(inputs) == len(targets)

	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)

	for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			# excerpt = slice(start_idx, start_idx + batch_size)
			excerpt = range(start_idx, start_idx + batch_size, 1)
		
		input_iter = [inputs[i] for i in excerpt]
		target_iter= [targets[i] for i in excerpt]
		yield input_iter, target_iter
		# yield inputs[excerpt], targets[excerpt]


def load_model(model_name, network):
	with np.load(model_name) as f:
		param_values = [f['arr_%d' % i] for i in range(len(f.files))]
	# param_values[0] = param_values[0].astype('float32')	
	param_values = [param_values[i].astype('float32') for i in range(len(param_values))]
	lasagne.layers.set_all_param_values(network, param_values)

def save_model(model_name, network):
	np.savez(model_name, *L.get_all_param_values(network))

def build_RNN(batch_size=1, input_size=26, n_hidden=275, num_output_units=61,
		weight_init=0.1, activation_fn=lasagne.nonlinearities.sigmoid,
		seed=int(time.time()), debug=False):
	np.random.seed(seed)
		# seed np for weight initialization

	l_in = L.InputLayer(shape=(batch_size, None, input_size))
	# l_in = L.InputLayer(shape=(None, None, input_size))
		# (batch_size, max_time_steps, n_features_1, n_features_2, ...)
		# Only stochastic gradient descent
	if debug:
		get_l_in = theano.function([l_in.input_var], L.get_output(l_in))
		l_in_val = get_l_in(X)
		print('output size:', end='\t');	print(Y.shape)
		print('input size:', end='\t');		print(X[0].shape)
		print('l_in size:', end='\t');		print(l_in_val.shape)

	l_rnn = L.recurrent.RecurrentLayer(
				l_in, num_units=n_hidden, 
				nonlinearity=activation_fn,
				W_in_to_hid=lasagne.init.Uniform(weight_init),
				W_hid_to_hid=lasagne.init.Uniform(weight_init),
				b=lasagne.init.Constant(0.),
				hid_init=lasagne.init.Constant(0.), 
				learn_init=False)
	if debug:
		get_l_rnn = theano.function([l_in.input_var], L.get_output(l_rnn))
		l_rnn_val = get_l_rnn(X)
		print('l_rnn size:', end='\t');	print(l_rnn_val.shape)


	l_reshape = L.ReshapeLayer(l_rnn, (-1, n_hidden))
	if debug:
		get_l_reshape = theano.function([l_in.input_var], L.get_output(l_reshape))
		l_reshape_val = get_l_reshape(X)
		print('l_reshape size:', end='\t');	print(l_reshape_val.shape)


	l_out = L.DenseLayer(l_reshape, num_units=num_output_units, 
		nonlinearity=T.nnet.softmax)
	return l_out



def build_functions(network, LEARNING_RATE=1e-5, MOMENTUM=0.9, debug=False):
	target_var = T.ivector('targets')

	# Get the first layer of the network
	l_in = L.get_all_layers(network)[0]

	network_output = L.get_output(network)

	# Function to get the output of the network
	output_fn = theano.function([l_in.input_var], network_output)
	if debug:
		l_out_val = output_fn(X)
		print('l_out size:', end='\t');	print(l_out_val.shape, end='\t');
		print('min/max: [{:.2f},{:.2f}]'.format(l_out_val.min(), l_out_val.max()))

	# Retrieve all trainable parameters from the network
	all_params = L.get_all_params(network, trainable=True)

	argmax_fn = theano.function([l_in.input_var], [T.argmax(network_output, axis=1)])
	if debug:
		print('argmax_fn')
		print(type(argmax_fn(X)[0]))
		print(argmax_fn(X)[0].shape)

	# loss = T.mean(lasagne.objectives.categorical_crossentropy(network_output, target_var))
	loss = T.sum(lasagne.objectives.categorical_crossentropy(network_output, target_var))

	# use Stochastic Gradient Descent with nesterov momentum to update parameters
	updates = lasagne.updates.momentum(loss, all_params, 
				learning_rate = LEARNING_RATE, 
				momentum = MOMENTUM)

	# Function to determine the number of correct classifications
	accuracy_fn = T.mean(T.eq(T.argmax(network_output, axis=1), target_var),
					dtype=theano.config.floatX)
				  
	# Function implementing one step of gradient descent
	train_fn = theano.function([l_in.input_var, target_var], [loss, accuracy_fn], updates=updates)

	# Function calculating the loss and accuracy
	validate_fn = theano.function([l_in.input_var, target_var], [loss, accuracy_fn])
	if debug:
		print(type(train_fn(X, Y)))
		# print('loss: {:.3f}'.format( float(train_fn(X, Y))))
		# print('accuracy: {:.3f}'.format( float(validate_fn(X, Y)[1]) ))

	return output_fn, argmax_fn, accuracy_fn, train_fn, validate_fn


def create_confusion(X, y, argmax_fn, debug=False):
	y_pred = []
	for X_obs in X:
	    for x in argmax_fn(X_obs):
	        for j in x:
	            y_pred.append(j)

	y_actu = []
	for Y in y:
	    for y in Y:
	        y_actu.append(y)

	conf_img = np.zeros([61, 61])
	assert (len(y_pred) == len(y_actu))

	for i in range(len(y_pred)):
		row_idx = y_actu[i]
		col_idx = y_pred[i]
		conf_img[row_idx, col_idx] += 1

	return conf_img, y_pred, y_actu


def train(network, dataset, training_fn, save_name='Best_model', num_epochs=100, batch_size=1,
	comput_confusion=False, debug=False):
	"""Curently one batch_size=1 is supported"""
	network_train_info = []
		# Train
		# val
		# test

	X_train, y_train, X_val, y_val, X_test, y_test = dataset
	if debug:
		print('X_train', end='\t\t')
		print(type(X_train), end='\t'); 	print(len(X_train))
		print('X_train[0]', end='\t')
		print(type(X_train[0]), end='\t');	print(X_train[0].shape)
		print('X_train[0][0]', end='\t')
		print(type(X_train[0][0]), end='\t');print(X_train[0][0].shape)
		print('X_train[0][0][0]', end='\t')
		print(type(X_train[0][0][0]), end='\t');print(X_train[0][0][0].shape)

		print('y_train', end='\t\t')
		print(type(y_train), end='\t'); 	print(len(X_train))
		print('y_train[0]', end='\t')
		print(type(y_train[0]), end='\t');	print(y_train[0].shape)
		print('y_train[0][0]', end='\t')
		print(type(y_train[0][0]), end='\t');print(y_train[0][0].shape)
		print()

	output_fn, argmax_fn, accuracy_fn, train_fn, validate_fn = training_fn


	# Initiate some vectors used for tracking performance
	train_error 		= np.zeros([num_epochs])
	train_accuracy 		= np.zeros([num_epochs])
	train_batches 		= np.zeros([num_epochs])
	validation_error 	= np.zeros([num_epochs])
	validation_accuracy = np.zeros([num_epochs])
	validation_batches 	= np.zeros([num_epochs])
	test_error 			= np.zeros([num_epochs])
	test_accuracy 		= np.zeros([num_epochs])
	test_batches 		= np.zeros([num_epochs])

	train_epoch_error	= np.zeros([num_epochs])
	val_epoch_error		= np.zeros([num_epochs])
	test_epoch_error	= np.zeros([num_epochs])
	confusion_matrices  = []


	best_validation_error = 100
	for epoch in range(num_epochs):
		epoch_time = time.time()

		# Full pass over the training set
		for inputs, targets in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
			for i in range(len(inputs)):
				# TODO this forloop shouldn't exist!
				if debug:
					print('len(inputs) = {}'.format(len(inputs)))
					print(type(inputs[i]))
				# error, accuracy = train_fn([inputs[i]], targets[i])
				error, accuracy = train_fn(inputs[i], targets[i])

				train_error[epoch] += error
				train_accuracy[epoch] += accuracy
				train_batches[epoch] += 1

		# Full pass over the validation set
		for inputs, targets in iterate_minibatches(X_val, y_val, batch_size, shuffle=False):
			for i in range(len(inputs)):
				# error, accuracy = validate_fn([inputs[i]], targets[i])
				error, accuracy = validate_fn(inputs[i], targets[i])

				validation_error[epoch] += error
				validation_accuracy[epoch] += accuracy
				validation_batches[epoch] += 1

		# Full pass over the test set
		for inputs, targets in iterate_minibatches(X_test, y_test, batch_size, shuffle=False):
			for i in range(len(inputs)):
				# error, accuracy = validate_fn([inputs[i]], targets[i])
				error, accuracy = validate_fn(inputs[i], targets[i])

				test_error[epoch] += error
				test_accuracy[epoch] += accuracy
				test_batches[epoch] += 1


		# Print epoch summary
		train_epoch_error[epoch]= 100 - train_accuracy[epoch] / train_batches[epoch] * 100
		val_epoch_error[epoch]	= 100 - validation_accuracy[epoch] / validation_batches[epoch] * 100
		test_epoch_error[epoch]	= 100 - test_accuracy[epoch] / test_batches[epoch] * 100


		print("Epoch {} of {} took {:.3f}s".format(
			epoch + 1, num_epochs, time.time() - epoch_time))
		if val_epoch_error[epoch] < best_validation_error:
			best_validation_error = val_epoch_error[epoch]
			print("  New best model found!", end=" ")
			if save_name is not None:
				print("Model saved as " + save_name + '.npz')
				save_model(save_name + '.npz', network)
			else:
				print()


		print("  training loss:\t{:.6f}".format(
			train_error[epoch] / train_batches[epoch]), end='\t')
		print("train error:\t\t{:.6f} %".format(train_epoch_error[epoch]))

		print("  validation loss:\t{:.6f}".format(
			validation_error[epoch] / validation_batches[epoch]), end='\t')
		print("validation error:\t{:.6f} %".format(val_epoch_error[epoch]))

		print("  test loss:\t\t{:.6f}".format(test_error[epoch] / test_batches[epoch]), end='\t')
		print("test error:\t\t{:.6f} %".format(test_epoch_error[epoch]))

		# if comput_confusion:
		# 	confusion_matrices.append(create_confusion(X_val, y_val)[0])
		# 	print('  Confusion matrix computed')
		print()


	network_train_info.append(train_epoch_error)
	network_train_info.append(val_epoch_error)
	network_train_info.append(test_epoch_error)



	with open(save_name + '_var.pkl', 'wb') as cPickle_file:
		cPickle.dump(
		[network_train_info], 
		cPickle_file, 
		protocol=cPickle.HIGHEST_PROTOCOL)


	if comput_confusion:
		with open(save_name + '_conf.pkl', 'wb') as cPickle_file:
			cPickle.dump(
			[confusion_matrices], 
			cPickle_file, 
			protocol=cPickle.HIGHEST_PROTOCOL)

	return best_validation_error

