import numpy as np
import useful_functions as USF

class pooling2D(object):
	"""
	Pooling Method to define the pooling way for the function;
	Initialization parameters: pooling_size-2D matrix defining the filter size; strides-2D matrix defining the steps horizontally and vertically; padding-can be difined on 2 valid choices('same', 'valid'); method(int)-2 methods are available(0: maxPooling, 1: avePooling)
	"""
	global allowed_type
	allowed_type = ('MaxPooling2D', 'AveragePooling2D')
	def __init__(self, pooling_size, strides, padding, method):
		self._pooling_size = pooling_size
		self._strides = strides
		if not self._padding_validating(padding):
			return
		self._padding = padding
		if not self._type_validating(method):
			return
		self._type = method
		self.next = None
		self.pre = None
		
	def get_pooling_size(self):
		return self._pooling_size

	def get_strides(self):
		return self._strides

	def get_padding(self):
		return self._padding

	def get_type(self):
		return self._type

	def show_type(self):
		return allowed_type[self._type]

	def get_name(self):
		return "pooling"

	def change_pooling_size(self, pooling_size):
		self._pooling_size = pooling_size

	def change_strides(self, strides):
		self._strides = strides

	def change_padding(self, padding):
		if not self._padding_validating(padding):
			return
		self._padding = padding

	def change_type(self, method):
		if not self._type_validating(method):
			return
		self._type = method

	def _type_validating(self, method):
		if 0 <= method < len(allowed_type):
			return True
		else:
			print("pooling method not defined")
			return False

	def _padding_validating(self, padding):
		allowed_padding = ('valid', 'same')
		if  padding not in allowed_padding:
			print("padding is not defined")
			return False
		else:
			return True
	
	def computing(self, inp):
		"""using pooling method to treat the inputs
		   input: in shape[size, height, width, channel]
		   output: in shape[size, mod_height, mod_width, channel]"""
		size = inp.shape  # size shoud be like (num, height, width, channle)
		
		# the physical dimension of input must be non-negative	
		for i in range(len(size)):
			if size[i] <= 0:
				print('invalid input dimension')
				return None
		
		in_size, in_height, in_width, in_channel = size[0], size[1], size[2], size[3] # get the 4 dimensions of input individually
		stride_height, stride_width = self._strides[0], self._strides[1] # get the move step along height and width separately
		pooling_height, pooling_width = self._pooling_size[0], self._pooling_size[1]

		pad_top, pad_bottom, pad_left, pad_right = USF.padding(self._padding, in_height, in_width, stride_height, stride_width, pooling_height, pooling_width)

		# dimensions of the output
		height_after_padding = in_height + pad_top + pad_bottom
		out_height = ((height_after_padding - pooling_height) // stride_height) + 1

		width_after_padding = in_width + pad_left + pad_right
		out_width = ((width_after_padding - pooling_width) // stride_width) + 1

		output = np.array([[[[0.]*in_channel for _ in range(out_width)] for _ in range(out_height)] for _ in range(in_size)])
		for i in range(in_size):
			image = inp[i]	# individual image in size [height, width, channel]
			temp = np.array([[[0.]*in_channel for _ in range(out_width)] for _ in range(out_height)]) # Temporary list to store the output of individual image

			# index of result convolution matrix
			move_height = 0 # height direction movement
			
			ceil = 0 - pad_top
			bottom = pooling_height - pad_top - 1
			while bottom < in_height + pad_bottom:
				height_s = max(0, ceil)
				height_e = min(in_height-1, bottom)

				move_width = 0 # width direction movement
				left = 0 - pad_left
				right = pooling_width - pad_left - 1

				
				while right < in_width + pad_right:
					width_s = max(0, left)
					width_e = min(in_width-1, right)
					result = np.array([[0.]*in_channel]) # filter result of each point 
					
					if self._type == 0: # maxpooling implementation 
						# check whether or not padding method influence the current maximum method
						if (bottom - ceil) - (height_e - height_s) != 0 or (right - left) - (width_e - width_s) != 0:
							result = np.maximum(image[height_s][width_s], result)
						else:
							result = image[height_s][width_s]
						for h in range(height_s, height_e + 1):
							for w in range(width_s, width_e + 1):
								result = np.maximum(result, image[h][w])
					else:  # average pooling implementation
						"""to be continued"""
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

#test case:
"""
#pooling_size=np.random.randn(2, 2)
strides = np.random.randn(2, 2)
method = 0
padding = 'valid'
max_pooling = pooling2D(pooling_size, strides, padding, method)
print(max_pooling.get_pooling_size())
print(max_pooling.show_type())
print(max_pooling.get_strides())
print(max_pooling.get_padding())
max_pooling.change_padding('non_linear')
max_pooling.change_type(10)

# cases to test computing func only
#inp = np.array([[[[1, 2, -53], [2, 3, -55], [2, 7, -109]], [[3, 6, -88], [1, 7, 0], [0, 3, 1]], [[3, 5, 7], [9, 0, -100], [3, 5, 10]]], [[[1, 2, 3], [2, 3, 5], [2, 7, 9]], [[3, 6, 8], [1, 7, 0], [0, 3, 1]], [[3, 5, 7], [9, 0, 10], [3, 5, 10]]]])
inp = np.array([[[[1, 2, 3], [2, 4, 6], [2, 7, 90]], [[34, 5, 0], [4, 5, 9], [5, 0, 3]], [[2, 6, 45], [9, 3, 10], [5, 67, 4]]]])
strides = (2, 1)
padding = 'same'
pooling_size = (11, 2)

pooling_layer = pooling2D(pooling_size, strides, padding, 0)
print(pooling_layer.computing(inp))
"""
