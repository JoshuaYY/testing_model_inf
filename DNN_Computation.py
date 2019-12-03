import numpy as np
import Test_Model as TM

def build_model(P, A):
	"""test model configuring with given parameters
	   Input: P-matrix in this form:[W, B, W, B,...], size must be even number, pay attention to dimension match
	   		  A-list collection of activation function type
	   Output: a test model consisting of all hidden layers with training parameters
	"""
	test = TM.Test_Model()
	L_P = len(P) # length of parameters
	L_A = len(A) # length of activations
	if (L_P & 1) or (L_A != L_P / 2):    # parameter length must be even due to pair of weight and bias
		print("missing parameters")
		return test
	for i in range(L_A):
		test.add_layer(P[2*i], P[2*i + 1], A[i])
	return test
  

def act_relu(res):
	"""treat matrix res element wise to mute negatives
	"""
	res[res<0] = 0
	return res

#test case:
t = np.array([[-1.02, 2.94], [0., -4.0]])
assert np.array_equal(act_relu(t), np.array([[0, 2.94], [0, 0]]))
assert np.array_equal(act_relu(t), np.array([[1.02, 2.94], [0, 0]])) == False


def act_sigm(res):
	"""fix each element of res to range(0, 1)
	"""
	activation_func = lambda x: 1 /(1 + np.exp(-x))
	return activation_func(res)
#test case:
t = np.array([[0, 5], [3, -5]])
assert np.array_equal(act_sigm(t), np.array([[1 / 2, 1 / (1 + np.exp(-5))], [1 / (1 + np.exp(-3)), 1 / (1 + np.exp(5))]]))


def act_tanh(res):
	"""fix each element of res to range(-1, 1)"""
	activation_func = lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
	return activation_func(res)

#test cases:
t = np.array([[0.04, -2.4], [3.9, -9]])
T = act_tanh(t)
A = lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
for i in range(2):
	for j in range(2):
		t[i][j] = A(t[i][j])
assert np.array_equal(T, t)


def act_softmax(res):
	"""fix each element of res to range(0, 1) with the sum of each column to be 1 showing the probability of each class"""
	res = np.exp(res)
	summy = np.sum(res, axis = 0)
	return res / summy

#test cases:
t = np.array([[1, 1], [4, 4]])
transfer = np.exp(t)
summy = np.sum(transfer, axis=0)
assert np.array_equal(act_softmax(t), transfer / summy)


def computing(test_samples, model):
	"""running the test samples through model to get result
	   Input: test_samples-Sample Matrix with each column as single sample(sample dimension must be flattened into a column vector; model is with training parameters(be caution about the dimension of first layer paramter to be matched with input sample's)
	   Output: Labels list containing probability of each class
	"""
	Activations = ["linear", "relu", "sigmoid", "tanh", "softmax"]
	current_input = np.array(test_samples)  # transfer the test samples into numpy version
	num_layers = len(model) # the num of layers in the model
	for i in range(num_layers):
		layer = model.get_layer(i)
		current_weight = np.array(layer.get_weight())
		current_bias = np.array(layer.get_bias())
		current_activation = layer.get_activation()
		'''
		if (current_activation == "unknown") or (current_input.shape[0] != current_weight.shape[1]) or (current_bias.shape[0] != current_weight.shape[0]):
			print("unmatched dimension or unknown activation")
			return None
		'''
		z = np.add(np.dot(current_weight, current_input),  current_bias)
		if current_activation == "linear":
			output = z
		elif current_activation == "relu":
			output = act_relu(z)	
		elif current_activation == "sigmoid":
			output = act_sigm(z)	
		elif current_activation == "tanh":
			output = act_tanh(z)	
		elif current_activation == "softmax":
			output = act_softmax(z)
		current_input = output
	return output 

#test cases:
activations = [1, 2, 4]
inp = [[1, 2.0, 3.5, 6.04], [-0.5, 6.4, 8.1, -2.3]]
W1 = np.random.normal(size = [3, 2])
B1 = np.random.normal(size = [3, 1])
W2 = np.random.normal(size = [2, 3])
B2 = np.random.normal(size = [2, 1])
W3 = np.random.normal(size = [5, 2])
B3 = np.random.normal(size = [5, 1])
P = [W1, B1, W2, B2, W3, B3]
model = build_model(P, activations)
result = computing(inp, model)
