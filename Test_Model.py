import Layer
import numpy as np
class Test_Model(object):
	"""This model will build the neural net for further process
	   The key point is that I will use double linkedlist to store layer parameters
	"""
	def __init__(self):
		self._head = Layer.Layer(None, None, None)
		self._tail = Layer.Layer(None, None, None)
		self._head.next = self._tail
		self._tail.pre = self._head
		self._size = 0
	
	#Input:W-weights matrix; B-bias matrix; activation-nonlinear function(0:'linear', 1:'relu', 2:'sigmoid', 3:'tanh', 4:'softmax') caution: activation is integer
	def add_layer(self, W, B, activation, index=None):
		if index == None:
			index = self._size
		if index > self._size or index < 0:
			print("out of boundary")
			return
		if index == self._size:
			target_layer = self._tail
		else:
			target_layer = self._travese(index)
		new_layer = Layer.Layer(W, B, activation)
		target_layer.pre.next = new_layer
		new_layer.pre = target_layer.pre
		target_layer.pre = new_layer
		new_layer.next = target_layer
		self._size += 1
	
	def remove_layer(self, index=None):
		if index == None:
			index = self._size
		target_layer = self.get_layer(index)
		if not target_layer: return
		layer_removed = target_layer.pre
		layer_removed.next.pre = layer_removed.pre
		layer_removed.pre.next = layer_removed.next
		layer_removed.next = None
		layer_removed.pre = None
		self._size -= 1
	
	def get_layer(self, index):
		if self._size == 0: return None
		if index > self._size or index < 0:
			print("out of boundary")
			return None
		elif index < self._size:
			target_layer = self._traverse(index)
		else: target_layer = self._tail
		return target_layer
		
	# move the point to the target layer		
	def _traverse(self, index):
		move = self._head.next
		for _ in range(index):
			move = move.next	
		return move

	# Go through the whole layers with given samples
	# Samples in the size n by m meaning m samples with n dimension each
	def computing(self, samples):
		current_input = np.array(samples) # transfer the samples into numpy version
		current_layer = self._head.next		
		while current_layer.next is not None:
			current_weight = np.array(current_layer.get_weight())
			current_bias = np.array(current_layer.get_bias())
			current_activation = current_layer.get_activation()
			z = np.add(np.dot(current_weight, current_input), current_bias)






	def __len__(self):
		return self._size
