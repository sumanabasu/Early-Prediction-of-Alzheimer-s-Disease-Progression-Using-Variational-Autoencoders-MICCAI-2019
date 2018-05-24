layer_config = {
	'conv1': {
		'in_channels': 1,
		'out_channels': 16,
		'kernel_size': 7,
		'stride': 1,
		'padding': 0
	},
	'conv2': {
		'in_channels': 16,
		'out_channels': 32,
		'kernel_size': 5,
		'stride': 1,
		'padding': 0
	},
	'conv3': {
		'in_channels': 32,
		'out_channels': 64,
		'kernel_size': 3,
		'stride': 1,
		'padding': 0
	},
	'conv4': {
		'in_channels': 64,
		'out_channels': 128,
		'kernel_size': 3,
		'stride': 1,
		'padding': 0
	},
	'fc1'	:	{
		'in'	: 	128,
		'out'	:	3
	},
	
	'maxpool3d'	:	{
		'layer1'	:	{
			'kernel'	:	7,
			'stride'		:	3
		},
		'layer2'	:	{
			'kernel'	:	5,
			'stride'		:	3
		},
		'layer3'	:	{
			'kernel'	:	3,
			'stride'		:	2
		},
		'adaptive'	:	1
	}
}

params	=	{
	'model'	:	{
		'conv_drop_prob'	: 0.2,
		'fcc_drop_prob'		: 0.2
	},
	
	'train'	:	{
		'learning_rate' 	: 0.0001,
		'num_epochs' 		: 100,
		'batch_size' 		: 8,
		'label_weights' 	: [0.3, 0.75, 1]
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