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
"""
def computing(inp, pooling_size, strides, padding):
	size = inp.shape  # size shoud be like (num, height, width, channle)
	
	# the physical dimension of input must be non-negative	
	for i in range(len(size)):
		if size[i] <= 0:
			print('invalid input dimension')
			return None
	
	in_size, in_height, in_width, in_channel = size[0], size[1], size[2], size[3] # get the 4 dimensions of input individually
	stride_height, stride_width = strides[0], strides[1] # get the move step along height and width separately
	pooling_height, pooling_width = pooling_size[0], pooling_size[1]
	pad_top = pad_bottom = pad_left = pad_right = 0

	pad_top, pad_bottom, pad_left, pad_right = padding(padding, in_height, in_width, stride_height, stride_width, pooling_height, pooling_width)

	# dimensions of the output
	height_after_padding = in_height + pad_top + pad_bottom
	out_height = ((height_after_padding - pooling_height) // stride_height) + 1

	width_after_padding = in_width + pad_left + pad_right
	out_width = ((width_after_padding - pooling_width) // stride_width) + 1
	output = np.array([[[[0]*in_channel for _ in range(out_width)] for _ in range(out_height)] for _ in range(in_size)])
	return output


inp = np.array([[[[1, 2, 3], [2, 4, 6], [2, 7, 90]], [[34, 5, 0], [4, 5, 9], [5, 0, 3]], [[2, 6, 45], [9, 3, 10], [5, 67, 4]]]])
strides = (2, 1)
padding = 'same'
polling_size = (1, 2)
print(computing(inp, pooling_size, strides, padding))
"""
