import Test_Model as TM
import numpy as np
import Conv2D_Layer as CLY
import pooling2D as PL2

class CNN_Test_Model(TM.Test_Model):
	
	global allowed_padding
	allowed_padding = ('valid', 'same')
	def __init(self):
		super(CNN_Test_Model, self).__init__()


	def add_conv2D_Layer(self, F, B, activation, strides, padding, index=None):
		if index == None:
			index = self._size
		if index > self._size or index < 0:
			print("out of boundary")
			return
		if activation < 0 or activation >= 5:
			print("activation func not defined")
			return
		if padding not in allowed_padding:
			print("padding type not defined")
			return
		target_layer = self._traverse(index)
		new_Conv2D_Layer = CLY.Conv2D_Layer(F, B, activation, strides, padding)
		target_layer.pre.next = new_Conv2D_Layer
		new_Conv2D_Layer.pre = target_layer.pre
		target_layer.pre = new_layer
		new_Conv2D_Layer.next = target_layer
		self._size += 1

	def add_pooling_Layer(self, pooling_size, strides, padding, method, index=None):
		if index == None:
			index = self._size
		if index > self._size or index < 0:
			print("out of boundary")
			return
		if padding not in allowed_padding:
			print("padding type not defined")
			return
		if method < 0 or method > 1:
			print("pooling method not defined")
			return
		target_layer = self._traverse(index)
		new_Pooling_Layer = PL2.pooling2D(pooling_size, strides, padding, method)
		target_layer.pre.next = new_Pooling_Layer
		new_Pooling_Layer.pre = target_layer.pre
		target_layer.pre = new_Pooing_Layer
		new_Pooling_Layer.next = target_layer
		self._size += 1
		
		
	def computing(self, samples):
		
		current_input = np.array(samples)
		current_layer = self._head.next
		while current_layer.next is not None:
			name = current_layer.get_name()
			if name == 'conv2D':
				output = self._conv2D_computing(current_layer, current_input)
			elif name == 'pooling'
				output = self._pooling2D_computing(current_layer, current_input)
			else:
				print("undefined layer type")
				return
			current_input = output
			current_layer = current_layer.next

		






