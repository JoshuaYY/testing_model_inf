import numpy as np
import Layer as Ly

class Conv2D_Layer(Ly.Layer):
	"""
	Design a convolutional layer(filter) to store Conv2D filter.
	Initialization Parameter: F-3D Matrix; B-1D array with length same as the channel of F; activation-non linear function(0:'linear', 1:'relu', 2:'sigmoid', 3:'tanh', 4:'softmax'); strides-a 2-element array specify the sliding steps horizontally and vertically; padding-two possible strings choice('valid' vs 'same') with 'valid' to shrink the size of input and 'same' to keep the size of it
	"""
	def __init__(self, F, B, activation, strides, padding, Noise=None):
		super(Conv2D_Layer, self).__init__(F, B, activation, Noise)
		if not self._padding_validating(padding):
			return
		self._strides = strides
		self._padding = padding

	def get_strides(self):
		return self._strides

	def get_padding(self):
		return self._padding

	def change_strides(self, strides):
		self._strides = strides

	def change_padding(self, padding):
		if not self._padding_validating(padding):
			return
		self._padding = padding

	'''check whether the given padding in defined two choices'''
	def _padding_validating(self, padding):
		allowed_padding = ('valid', 'same')
		if  padding not in allowed_padding:
			print("padding is not defined")
			return False
		else:
			return True

	def get_name(self):
		return 'conv2D'
	
	# with given input, to calculate output with the parameters
	# input must be in numpy type
	def computing(self, inp):
		size = inp.shape
		for i in range(len(size)):
			if size[i] <= 0:
				print('invalid input dimension')
				return None
		in_size, in_height, in_width, in_channel = inp[0], inp[1], inp[2], inp[3]
		stride_height, stride_width = self._strides[0], self._strides[1]
		filter_size = self._weights.shape
		filter_height, filter_width = filter_size[0], filter_size[1]
		if self._padding == 'same':
			if in_height % stride_height == 0:
				pad_along_height = max(filter_height - stride_height, 0)
			else:
				pad_along_height = max(filter_height - (in_height % stride_height), 0)
			if in_width % stride_width == 0:
				pad_along_width = max(filter_width - stride_width, 0)
			else:
				pad_along_width = max(filter_width - (in_width % stride_width), 0)




#Test Cases:
FIL = np.array([[[1, 2, 4], [2, 5, 6]], [[2, 5, 6], [8, 10, 17]]])
bias = np.array([1, 4, 5])
activation = 2
strides = (2, 4)
padding = 'valid'
#padding = 'Geneious'
convLayer = Conv2D_Layer(FIL, bias, activation, strides, padding) 

convLayer.change_strides(np.random.rand(5, 2, 3))
print(convLayer.get_strides())
print(convLayer.get_padding())
convLayer.change_padding('same')
print(convLayer.get_padding())
print(convLayer.get_weight())
print(convLayer.get_bias())
convLayer.change_padding('non_valid')
convLayer.change_activation(8)
print(convLayer.show_activation())
print(convLayer.get_name())
