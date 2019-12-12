import numpy as np
import Layer as Ly
import Activations as Act
import useful_functions as USF

class Conv2D_Layer(Ly.Layer):
	"""
	Design a convolutional layer(filter) to store Conv2D filter.
	Initialization Parameter: F-4D Matrix([height, width, channel, number]); B-1D array with length same as the channel of F; activation-non linear function(0:'linear', 1:'relu', 2:'sigmoid', 3:'tanh', 4:'softmax'); strides-a 2-element array specify the sliding steps horizontally and vertically; padding-two possible strings choice('valid' vs 'same') with 'valid' to shrink the size of input and 'same' to keep the size of it
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
		size = inp.shape  # size shoud be like (num, height, width, channle)
		
		# the physical dimension of input must be non-negative	
		for i in range(len(size)):
			if size[i] <= 0:
				print('invalid input dimension')
				return None
		
		in_size, in_height, in_width, in_channel = size[0], size[1], size[2], size[3] # get the 4 dimensions of input individually
		stride_height, stride_width = self._strides[0], self._strides[1] # get the move step along height and width separately
		filter_size = self._weights.shape # 4 dimensions of filter
		filter_height, filter_width, filter_channel, filter_num = filter_size[0], filter_size[1], filter_size[2], filter_size[3] # get the height and width of filter separately
		if in_channel != filter_channel:
			print("channel does not match")
			return None
		
		# padding method to add additional 0 columns and arrays on the input
		pad_top, pad_bottom, pad_left, pad_right = USF.padding(self._padding, in_height, in_width, stride_height, stride_width, filter_height, filter_width)
     	
		# dimensions of the output
		height_after_padding = in_height + pad_top + pad_bottom
		out_height = ((height_after_padding - filter_height) // stride_height) + 1

		width_after_padding = in_width + pad_left + pad_right
		out_width = ((width_after_padding - filter_width) // stride_width) + 1

		output = np.array([[[[0]*filter_num for _ in range(out_width)] for _ in range(out_height)] for _ in range(in_size)])
		for i in range(in_size):
			image = inp[i]	# individual image in size [height, width, channel]
			temp = np.array([[[0]*filter_num for _ in range(out_width)] for _ in range(out_height)]) # Temporary list to store the output of individual image

			# index of result convolution matrix
			move_height = 0 # height direction movement
			
			ceil = 0 - pad_top
			bottom = filter_height - pad_top - 1
			while bottom < in_height + pad_bottom:
				height_s = max(0, ceil)
				height_e = min(in_height-1, bottom)

				move_width = 0 # width direction movement
				left = 0 - pad_left
				right = filter_width - pad_left - 1

				while right < in_width + pad_right:
					width_s = max(0, left)
					width_e = min(in_width-1, right)
					result = np.array([[0]*filter_num]) # filter result of each point 
					for h in range(height_s, height_e + 1):
						for w in range(width_s, width_e + 1):
							target_point = image[h][w] # individual point channel of each image imput
							filter_corresponded = self._weights[h-height_s][w-width_s] # the corresponding filter in shape[channel, size]
							result = np.add(np.dot(target_point, filter_corresponded), result)
					result = np.add(result, self._biases)
					if self._activation == 1:
						result = Act.act_relu(result)
					elif self._activation == 2:
						result = Act.act_sigm(result)
					elif self._activation == 3:
						result = Act.act_tanh(result)
					elif self._activation == 0:
						result = result
					else:
						print('undefined activation or activation not applicabel to Conv Layer')
						return None
					temp[move_height][move_width] = result
					left += stride_width
					right += stride_width
					move_width += 1
				ceil += stride_height
				bottom += stride_height
				move_height += 1
			output[i] = temp
		
		return output

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

# cases to test computing func only
inp = np.array([[[[1, 2, -53], [2, 3, -55], [2, 7, -109]], [[3, 6, -88], [1, 7, 0], [0, 3, 1]], [[3, 5, 7], [9, 0, -100], [3, 5, 10]]], [[[1, 2, 3], [2, 3, 5], [2, 7, 9]], [[3, 6, 8], [1, 7, 0], [0, 3, 1]], [[3, 5, 7], [9, 0, 10], [3, 5, 10]]]])
fil = np.array([[[[1, 2], [6, 0], [2, 5]]], [[[2, 7], [1, 0], [3, 10]]]])
#fil = np.array([[[[1, 1], [1, 1], [1, 1]]], [[[1, 1], [1, 1], [1, 1]]]])

strides = (1, 2)
padding = 'same'
Conv2D = Conv2D_Layer(fil, np.array([2, 1]), 1, strides, padding) 
res = Conv2D.computing(inp)
print(res.shape)
print(res)
