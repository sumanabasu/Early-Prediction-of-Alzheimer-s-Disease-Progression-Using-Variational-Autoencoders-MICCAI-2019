'''
Spatial Pyramid Pooling
'''
import math
import torch
from torch import nn

def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
	'''
	previous_conv: a tensor vector of previous convolution layer
	num_sample: an int number of image in the batch
	previous_conv_size: an int vector [depth, height, width] of the matrix features size of previous convolution
	layer
	out_pool_size: a int vector of expected output size of max pooling layer

	returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
	'''
	# print(previous_conv.size())
	for i in range(len(out_pool_size)):
		# print(previous_conv_size)
		d_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
		h_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
		w_wid = int(math.ceil(previous_conv_size[2] / out_pool_size[i]))
		d_pad = (h_wid * out_pool_size[i] - previous_conv_size[0] + 1) / 2
		h_pad = (w_wid * out_pool_size[i] - previous_conv_size[1] + 1) / 2
		w_pad = (d_wid * out_pool_size[i] - previous_conv_size[2] + 1) / 2
		maxpool = nn.MaxPool3d((d_wid, h_wid, w_wid), stride=(d_wid, h_wid, w_wid), padding=(d_pad, h_pad, w_pad))
		x = maxpool(previous_conv)
		if (i == 0):
			spp = x.view(num_sample, -1)
		# print("spp size:",spp.size())
		else:
			# print("size:",spp.size())
			spp = torch.cat((spp, x.view(num_sample, -1)), 1)
	return spp