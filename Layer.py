import numpy as np

class Layer(object):
	"""Design a container to store the parameters of each hidden layer.
   Initialization Parameters: W-weights matrix; B-bias matrix; activation-non linear function(0:'linear', 1:'relu', 2:'sigmoid', 3:'tanh', 4:'softmax'); Noise-to be defined in the future
	"""
	global Activations
	Activations = ["linear", "relu", "sigmoid", "tanh", "softmax"]
	def __init__(self, W, B, activation, Noise=None):
		if not self._activation_validate(activation):
			return
		self._weights = W
		self._biases = B
		self._activation = activation
		self.next = None  # pointer to the next layer
		self.pre = None   # pointer to the previous layer
		
	def get_weight(self):
		return self._weights

	def get_bias(self):
		return self._biases

	def get_activation(self):
		return self._activation
	
	# activation version easily understand
	def show_activation(self):
		return Activations[self._activation]

	def change_weight(self, W):
		self._weights = W

	def change_bias(self, B):
		self._biases = B

	def change_activation(self, A):
		if self._activation_validate(A):
			self._activation = A
		else:
			return

	def _activation_validate(self, A):
		if 0 <= A < len(Activations):
			return True
		else:
			print("activation function not defined")
			return False
		
