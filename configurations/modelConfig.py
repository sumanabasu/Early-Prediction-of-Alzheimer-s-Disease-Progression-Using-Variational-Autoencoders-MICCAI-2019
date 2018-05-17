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
		'out_channels': 128,
		'kernel_size': 3,
		'stride': 2,
		'padding': 0,
		'dilation': 1
	},
	'fc1'	:	{
		'in'	:	128 * 13 * 11 * 10,
		'out'	:	1024
	},
	'fc2'	:	{
		'in'	: 	1024,
		'out'	: 	512
	},
	'fc3'	:	{
		'in'	:	512,
		'out'	: 	256
	},
	'fc4'	:	{
		'in'	: 	256,
		'out'	:	3
	}
}

params	=	{
	'model'	:	{
		'conv_drop_prob'	: 0.5,
		'fcc_drop_prob'		: 0.5
	},
	
	'train'	:	{
		'learning_rate' 	: 0.0001,
		'num_epochs' 		: 100,
		'batch_size' 		: 3,
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