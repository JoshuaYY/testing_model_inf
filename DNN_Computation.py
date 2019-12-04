import numpy as np
import Test_Model as TM

def build_model(P, A):
	"""test model configuration with given parameters
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
result = model.computing(inp)
print(result)
