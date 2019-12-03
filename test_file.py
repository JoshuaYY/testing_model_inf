import numpy as np
import Layer
import Test_Model as TM
if __name__ == "__main__":
	M = [[3, 4, 5], [4, 5, 6]]
	B = [[4], [6]]
	activation = 2
	layer = Layer.Layer(M, B, activation)
	test1 = TM.Test_Model()
	for _ in range(10):
		test1.add_layer(M, B, activation)
	print(len(test1))
	test1.remove_layer()
	print(len(test1))
	test1.remove_layer(4)
	lay = test1.get_layer(len(test1))
