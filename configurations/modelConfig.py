import numpy as np

# section for binary or 3-way classification
num_classes = 3
name_classes = np.asarray(['NL', 'MCI', 'AD'])
class_weight = [0.3, 0.75, 1]


num_conv = 4
img_shape = np.array([213, 197, 189])

for i in range(3):
	for _ in range(num_conv):
		if i == 0 and _ == 0:
			img_shape[0] = img_shape[0] - 2
		else:
			img_shape[i] /= 2
			img_shape[i]-= 1


layer_config = {
	'conv1': {
		'in_channels': 1,
		'out_channels':16,
		'kernel_size': 3,
		'stride': 1,
		'padding': 1
	},
	'conv2': {
		'in_channels':16,
		'out_channels':16,
		'kernel_size': 3,
		'stride': 1,
		'padding': 1
	},
	'conv3': {
		'in_channels':16,
		'out_channels': 16,
		'kernel_size': 3,
		'stride': 1,
		'padding': 1
	},
	'conv4': {
		'in_channels': 16,
		'out_channels': 16,
		'kernel_size': 3,
		'stride': 1,
		'padding': 1
	},
	'conv5': {
		'in_channels': 16,
		'out_channels': 16,
		'kernel_size': 3,
		'stride': 1,
		'padding': 1
	},
	'conv6': {
		'in_channels': 16,
		'out_channels': 16,
		'kernel_size': 3,
		'stride': 1,
		'padding': 1
	},
	'conv7': {
		'in_channels': 16,
		'out_channels': 16,
		'kernel_size': 3,
		'stride': 1,
		'padding': 1
	},
	'conv8': {
		'in_channels': 16,
		'out_channels': 16,
		'kernel_size': 3,
		'stride': 1,
		'padding': 1
	},
	'conv9': {
		'in_channels': 16,
		'out_channels': 16,
		'kernel_size': 3,
		'stride': 1,
		'padding': 1
	},
	'conv10': {
		'in_channels': 16,
		'out_channels': 16,
		'kernel_size': 3,
		'stride': 1,
		'padding': 1
	},
	'conv11': {
		'in_channels': 16,
		'out_channels': 16,
		'kernel_size': 3,
		'stride': 1,
		'padding': 1
	},
	'conv12': {
		'in_channels': 16,
		'out_channels': 16,
		'kernel_size': 3,
		'stride': 1,
		'padding': 1
	},
	'conv13': {
		'in_channels': 16,
		'out_channels': 16,
		'kernel_size': 3,
		'stride': 1,
		'padding': 1
	},
	'conv14': {
		'in_channels': 16,
		'out_channels': 16,
		'kernel_size': 3,
		'stride': 1,
		'padding': 1
	},
	'conv15': {
		'in_channels': 16,
		'out_channels': 16,
		'kernel_size': 3,
		'stride': 1,
		'padding': 1
	},
	'conv16': {
		'in_channels': 16,
		'out_channels': 16,
		'kernel_size': 3,
		'stride': 1,
		'padding': 1
	},
		
	'fc1'	:	{
		'in'	: 	16 * 13 * 12 * 11,	# 1980, #11 * img_shape[0] * img_shape[1] * img_shape[2],
		'out'	:	4096
	},
	'fc2'	:	{
		'in'	: 	4096,
		'out'	:	4096
	},
	'fc3'	:	{
		'in'	: 	4096,
		'out'	:	1000
	},
	'final'	:	{
		'in'	: 	1000,
		'out'	:	num_classes
	},
	
	'maxpool3d'	:	{
		'l1'	:	{						#to preserve temporal information in  the early phase
			'kernel'	:	(1, 2, 2),
			'stride'	:	(1, 2, 2)
		},
		'ln'	:	{
			'kernel'	:	2,
			'stride'	:	2
		},
		
		'adaptive'	:	1
	}
}

params	=	{
	'model'	:	{
		'conv_drop_prob'	: 0.2,
		'fcc_drop_prob'		: 0.0
	},
	
	'train'	:	{
		'learning_rate' 	: 0.0001,
		'num_epochs' 		: 100,
		'batch_size' 		: 2,
		'label_weights' 	: class_weight
	}
}

# data augmentation
data_aug = {
	'horizontal_flip': 0.5,
	'vertical_flip': 0.5,
	#'spline_warp': True,
	#'warp_sigma': 0.1,
	#'warp_grid_size': 3,
	## 'crop_size': (100, 100),
	#'channel_shift_range': 5.
}