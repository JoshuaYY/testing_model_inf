import numpy as np
import Layer as Ly
import Test_Model as TM
if __name__ == "__main__":
	M = [[3, 4, 5], [4, 5, 6]]
	B = [[4], [6]]
	M_new = [[3, 4], [6, 7]]
	B_new = [[4], [9]]
	activation = 2
	# Layer Class Test Cases:
	layer = Ly.Layer(M, B, activation)
	print(layer.get_weight())
	print(layer.get_bias())
	print(layer.get_activation())
	print(layer.show_activation())
	layer.change_weight(M_new)
	layer.change_bias(B_new)
	layer.change_activation(3)
	print(layer.get_weight())
	print(layer.get_bias())
	print(layer.get_activation())
	print(layer.show_activation())
	layer.change_activation(5)

	# Test_Model Class Test Cases:
	model1 = TM.Test_Model()
	model1.add_layer(M, B, 4, 0)
	layer0 = model1.get_layer(0)
	layer_fail = model1.get_layer(10)
	layer1 = model1.remove_layer()
	print(layer0.get_weight())
	print(layer1.get_weight())
	print(layer0.get_bias())
	print(layer1.get_bias())
	print(layer0.get_activation())
	print(layer1.get_activation())
	
	act = [0, 1, 4, 6, 3, 5, 7]
	for _ in range(7):
		model1.add_layer(M, B, act[_])
	print(len(model1))
	layer3 = model1.get_layer(2)
	print(layer3.get_activation())
	layer4 = model1.get_layer(0)
	print(layer4.get_activation())
	layer5 = model1.get_layer(3)
	print(layer5.get_activation())
	layer6 = model1.get_layer(5)
	print(layer6)
	model1.remove_layer()
	print(len(model1))
	model1.remove_layer(4)
	print(len(model1))
