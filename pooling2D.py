import numpy as np

class pooling2D(object):
	"""
	Pooling Method to define the pooling way for the function;
	Initialization parameters: pooling_size-2D matrix defining the filter size; strides-2D matrix defining the steps horizontally and vertically; padding-can be difined on 2 valid choices('same', 'valid'); method-2 methods are available(0: maxPooling, 1: avePooling)
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

#test case:
pooling_size=np.random.randn(2, 2)
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
