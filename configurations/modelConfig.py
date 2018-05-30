num_classes = 2

layer_config = {
	'conv1': {
		'in_channels': 1,
		'out_channels': 5,
		'kernel_size': 3,
		'stride': 1,
		'padding': 0
	},
	'conv2': {
		'in_channels': 5,
		'out_channels': 5,
		'kernel_size': 3,
		'stride': 1,
		'padding': 0
	},
	'conv3': {
		'in_channels': 5,
		'out_channels': 5,
		'kernel_size': 3,
		'stride': 1,
		'padding': 0
	},
	'conv4': {
		'in_channels': 5,
		'out_channels': 5,
		'kernel_size': 3,
		'stride': 1,
		'padding': 0
	},
	
	'fc1'	:	{
		'in'	: 	5 * 24 * 10 * 9,
		'out'	:	4096
	},
	'fc2'	:	{
		'in'	: 	4096,
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
		'batch_size' 		: 8,
		'label_weights' 	: [1, 1]	#[0.3, 0.75, 1]
	}
}

# data augmentation
data_aug = {
	'horizontal_flip': 0.5,
	'vertical_flip': 0.5,
	'spline_warp': True,
	'warp_sigma': 0.1,
	'warp_grid_size': 3,
	## 'crop_size': (100, 100),
	'channel_shift_range': 5.
}