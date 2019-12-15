import numpy as np

def padding(padding_type, in_height, in_width, stride_height, stride_width, filter_height, filter_width):
	if padding_type == 'valid':
		return (0, 0, 0, 0)

	if in_height % stride_height == 0:
		pad_along_height = max(filter_height - stride_height, 0)
	else:
		pad_along_height = max(filter_height - (in_height % stride_height), 0)
	if in_width % stride_width == 0:
		pad_along_width = max(filter_width - stride_width, 0)
	else:
		pad_along_width = max(filter_width - (in_width % stride_width), 0)
	
	pad_top = pad_along_height // 2
	pad_bottom = pad_along_height - pad_top
	pad_left = pad_along_width // 2
	pad_right = pad_along_width - pad_left
	return (pad_top, pad_bottom, pad_left, pad_right)

def Flatten(conv_result):
	"""Convert 3D data into 1D Data"""
	
	dimension = conv_result.shape

	size, height, width, channel = dimension[0], dimension[1], dimension[2], dimension[3]

	dnn_result = np.array([[0] * (height * width * channel) for _ in range(size)])

	# func to concatenate the whole 3D data
	def combine(elements):
		if len(elements.shape) == 1:
			return elements
		result = []
		for i in range(len(elements)):
			element = combine(elements[i])
			result = np.concatenate((result, element), axis=0)
		return result

	for i in range(size):
		target = conv_result[i]
		dnn_result[i] = combine(target)
	
	return dnn_result
"""
#test case:
C = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24], [25, 26, 27]]]])
print(Flatten(C))
"""
