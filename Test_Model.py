import Layer
import numpy as np
import Activations as Act
class Test_Model(object):
	"""This model will build the neural net for further process
	   The key point is that I will use double linkedlist to store layer parameters
	"""
	def __init__(self):
		self._head = Layer.Layer(None, None, 0)
		self._tail = Layer.Layer(None, None, 0)
		self._head.next = self._tail
		self._tail.pre = self._head
		self._size = 0
	
	#Input:W-weights matrix; B-bias matrix; activation-nonlinear function(0:'linear', 1:'relu', 2:'sigmoid', 3:'tanh', 4:'softmax') caution: activation is integer; default to add layer in the end
	def add_layer(self, W, B, activation, index=None):
		if index == None:
			index = self._size
		if index > self._size or index < 0:
			print("out of boundary")
			return
		if activation < 0 or activation >= 5:
			print("activation func not defined")
			return
		target_layer = self._traverse(index)
		new_layer = Layer.Layer(W, B, activation)
		target_layer.pre.next = new_layer
		new_layer.pre = target_layer.pre
		target_layer.pre = new_layer
		new_layer.next = target_layer
		self._size += 1
	
	# delete the designated layer and return it(default to remove the last layer)
	def remove_layer(self, index=None):
		if index is None:
			index = self._size - 1
		target_layer = self.get_layer(index)
		if target_layer is not None:
			target_layer.pre.next = target_layer.next
			target_layer.next.pre = target_layer.pre
			target_layer.pre = None
			target_layer.next = None
			self._size -= 1
		return target_layer
	
	def get_layer(self, index):
		if self._size == 0:
			print("empty model")
			return None
		if index >= self._size or index < 0:
			print("out of boundary")
			return None
		return self._traverse(index)
		
	# move the point to the target layer		
	def _traverse(self, index):
		move = self._head.next
		for _ in range(index):
			move = move.next	
		return move

	# Go through the whole layers with given samples
	# Samples in the size n by m meaning m samples with n dimension each
	def computing(self, samples):

		# checking func to make sure the dim of input, weigh and bias is consistent
		def dimension_match(weight, bias, inp):
			dim_inp = inp.shape[0]
			dim_height_weight = weight.shape[0]
			dim_width_weight = weight.shape[1]
			dim_bias = bias.shape[0]
			if dim_inp != dim_width_weight:
				print("dim conflict between input and weight")
				return False
			if dim_height_weight != dim_bias:
				print("dim conflict between weight and bias")
				return False
			return True

		current_input = np.array(samples) # transfer the samples into numpy version
		current_layer = self._head.next		
		while current_layer.next is not None:
			current_weight = np.array(current_layer.get_weight())
			current_bias = np.array(current_layer.get_bias())
			current_activation = current_layer.get_activation()
			if not dimension_match(current_weight, current_bias, current_input):
				return None				
			output = np.add(np.dot(current_weight, current_input), current_bias)
			if current_activation == 1:
				output = Act.act_relu(output)
			elif current_activation == 2:
				output = Act.act_sigm(output)
			elif current_activation == 3:
				output = Act.act_tanh(output)
			elif current_activation == 4:
				output = Act.act_softmax(output)
			current_input = output
			current_layer = current_layer.next
		return output


	def __len__(self):
		return self._size
