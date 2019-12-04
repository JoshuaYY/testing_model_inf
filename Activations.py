import numpy as np

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
