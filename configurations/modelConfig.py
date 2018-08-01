import numpy as np

num_classes = 2

if num_classes == 3:
	name_classes = np.asarray(['NL', 'MCI', 'AD'])
	class_weight = [1, 1, 1]
else:
	name_classes = np.asarray(['NL', 'Diseased'])
	class_weight = [1, 1]

num_conv = 4
img_shape = np.array([213, 197, 189])

for _ in range(num_conv):
	img_shape = np.ceil(img_shape / 2)
	
#img_shape.astype(np.int32)

layer_config = {
	'conv1': {
		'in_channels': 1,
		'out_channels': 11,
		'kernel_size': 3,
		'stride': 1,
		'padding': 1
	},
	'conv2': {
		'in_channels': 11,
		'out_channels': 11,
		'kernel_size': 3,
		'stride': 1,
		'padding': 1
	},
	'conv3': {
		'in_channels': 11,
		'out_channels': 11,
		'kernel_size': 3,
		'stride': 1,
		'padding': 1
	},
	'conv4': {
		'in_channels': 11,
		'out_channels': 11,
		'kernel_size': 3,
		'stride': 1,
		'padding': 1
	},
	
	'fc1': {
		'in': 11 * int(np.prod(img_shape[:])), #14 * 13 * 12,
		'out': 4096
	},
	'fc2': {
		'in': 4096,
		'out': num_classes
	},
	
	'tconv1': {
		'in_channels': 11,
		'out_channels': 11,
		'kernel_size': 3,
		'stride': 1,
		'padding': 1
		#'output_padding' : 1	#(0, 0, 1)
	},
	'tconv2': {
		'in_channels': 11,
		'out_channels': 11,
		'kernel_size': 3,
		'stride': 1,
		'padding': 1
		#'output_padding' : 1
	},
	'tconv3': {
		'in_channels': 11,
		'out_channels': 11,
		'kernel_size': 3,
		'stride': 1,
		'padding': 1
		#'output_padding' : 1	#(0, 0, 0)
	},
	'tconv4': {
		'in_channels': 11,
		'out_channels': 1,
		'kernel_size': 3,
		'stride': 1,
		'padding': 1
		#'output_padding' : 1	#(0, 0, 0)
	},
	
	'maxpool3d': {
		'ln': {
			'kernel': 2,
			'stride': 2
		},
		
		'adaptive': 1
	}
}

params = {
	'model': {
		'conv_drop_prob': 0.2,
		'fcc_drop_prob'	: 0.0
	},
	
	'train'	:	{
		'model'				: 'AutoEncoder',
		'timestamp'			: 'NextLabel',	#'CurrLabel'
		'learning_rate' 	: 0.0001,
		'num_epochs' 		: 100,
		'batch_size' 		: 4,
		'label_weights' 	: class_weight,
		'lambda'			: 2.5,
		'lr_schedule'		: [15, 25, 35]
	}
}

# data augmentation
data_aug = {
	'horizontal_flip': 0.5,
	'vertical_flip': 0.5,
	# 'spline_warp': True,
	# 'warp_sigma': 0.1,
	# 'warp_grid_size': 3,
	## 'crop_size': (100, 100),
	#'channel_shift_range': 5.
}

'''
'conv5': {
		'in_channels': 11,
		'out_channels': 11,
		'kernel_size': 3,
		'stride': 1,
		'padding': 1
	},
	
'tconv5': {
		'in_channels': 11,
		'out_channels': 1,
		'kernel_size': 3,
		'stride': 1,
		'padding': 1
		#'output_padding' : 1	#(0, 0, 0)
	},
	
	'''