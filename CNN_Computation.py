import numpy as np
import DNN_Computation as DCN
import CNN_Test_Model as CTM

def build_model(layers, parameters):
	"""
	CNN Test Model Configuration with given parameters
	Input: layers-a list containing each layer information as dictionary
	       parameters-list of training weights(Matrix Type)
	"""

	layer_count = 0 # index of current layer in processing
	parameter_count = 0 # index of parameter to be extracted
	num_layers = len(layers)
	CNN_Model = CTM.CNN_Test_Model() # Model covers the computation from Conv layer to Flatten one(excluding)
	
	Activations = {'linear':0, 'relu':1, 'sigmoid':2, 'tanh':3, 'softmax':4}
	while layer_count < num_layers: 
		layer = layers[layer_count]
		layer_type = layer['class_name'] # string type with four possible values: 'Conv2D', 'MaxPooling2D', 'Flatten', 'Dense'
		# signal word 'Flatten' to indicate the end of Convolutional Model building
		if layer_type == 'Flatten':
			break

		config = layer['config']
		strides, padding = config['strides'], config['padding']	# strides: a typle consisting of 2 ints; padding: string of 2 possible values('same', 'valid')
		if layer_type == 'Conv2D':
			FILTER = parameters[parameter_count]
			parameter_count += 1
			
			BIAS = np.array([0] * config['filters']) # default bias equal 0 
			if config['use_bias']:
				BIAS = parameters[parameter_count]
				parameter_count += 1

			activation = config['activation'] # string type with 4 possible cases('linear', 'relu', 'sigmoid', 'tanh')
			if activation not in ('linear', 'relu', 'sigmoid', 'tanh'):
				print("activation not suitable for Conv Computation")
				return None

			CNN_Model.add_conv2D_Layer(FILTER, BIAS, Activations[activation], strides, padding)

		elif layer_type == 'MaxPooling2D':
			pooling_size = config['pool_size']	
			pooling_method = 0 # 0: maxPooling, 1: averagePooling
			CNN_Model.add_pooling_Layer(pooling_size, strides, padding, pooling_method)

		else:
			print("undefined layer type")
			return None
		layer_count += 1

	# DNN Model Building
	layer_count += 1
	weights_DNN = []  # weights to feed into the DNN build_model fcn
	activations = []  # activations to feed into the DNN build_model fcn
	while layer_count < num_layers:
		layer = layers[layer_count]
		layer_type = layer['class_name']
		if layer_type == 'Flatten':
			layer_count += 1
			continue
		elif layer_type == 'Dense':
			config = layer['config']
			weight = parameters[parameter_count]
			parameter_count += 1
			bias = np.array([0] * config['units']) 
			if config['use_bias']:
				bias = parameters[parameter_count]
				parameter_count += 1
			activation = config['activation'] # string type with 5 possible cases('linear', 'relu', 'sigmoid', 'tanh', 'softmax')
			if activation not in Activations.keys():
				print("activation not defined")
				return None
			weights_DNN.append(weight)
			weights_DNN.append(bias)
			activations.append(Activations[activation])
			layer_count += 1
		else:
			print("layer type not suitable for DNN Computation")
			return None
	#build DNN Model with the extracted parameters and activations	
	DNN_Model = DCN.build_model(weights_DNN, activations, True)

	return (CNN_Model, DNN_Model)

