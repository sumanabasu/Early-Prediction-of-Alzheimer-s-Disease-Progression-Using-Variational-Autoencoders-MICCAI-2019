layer_config = {
	'conv1': {
		'in_channels': 1,
		'out_channels': 16,
		'kernel_size': 3,
		'stride': 1,
		'padding': 1
	},
	'conv2': {
		'in_channels': 16,
		'out_channels': 32,
		'kernel_size': 3,
		'stride': 2,
		'padding': 0,
		'dilation': 1
	},
	'conv3': {
		'in_channels': 32,
		'out_channels': 64,
		'kernel_size': 3,
		'stride': 2,
		'padding': 0,
		'dilation': 1
	},
	'conv4': {
		'in_channels': 64,
		'out_channels': 128,
		'kernel_size': 3,
		'stride': 2,
		'padding': 0,
		'dilation': 1
	},
	'conv5': {
		'in_channels': 128,
		'out_channels': 256,
		'kernel_size': 3,
		'stride': 2,
		'padding': 0,
		'dilation': 1
	},
	'conv6': {
		'in_channels': 256,
		'out_channels': 512,
		'kernel_size': 3,
		'stride': 2,
		'padding': 0,
		'dilation': 1
	},
	'fc1'	:	{
		'in'	:	512 * 5 * 5 * 4,
		'out'	:	4096
	},
	'fc2'	:	{
		'in'	: 	4096,
		'out'	: 	2048
	},
	'fc3'	:	{
		'in'	:	2048,
		'out'	: 	1024
	},
	'fc4'	:	{
		'in'	: 	1024,
		'out'	:	3
	}
}

params	=	{
	'model'	:	{
		'conv_drop_prob'	: 0.2,
		'fcc_drop_prob'		: 0.0
	},
	
	'train'	:	{
		'learning_rate' 	: 0.0002,
		'num_epochs' 		: 100,
		'batch_size' 		: 3,
		'label_weights' 	: [0.7, 1, 1] #[0.63, 0.43, 1]
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