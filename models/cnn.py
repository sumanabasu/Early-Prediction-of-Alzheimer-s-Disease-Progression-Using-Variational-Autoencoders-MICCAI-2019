'''
Convolutional Neural Network
'''
import torch.nn as nn
from  configurations.modelConfig import layer_config, params

class CnnVanilla(nn.Module):
	"""Cnn with simple architecture of 6 stacked convolution layers followed by a fully connected layer"""
	
	def __init__(self):
		super(CnnVanilla, self).__init__()
		self.conv1 = nn.Conv3d(in_channels=layer_config['conv1']['in_channels'], out_channels=layer_config['conv1']['out_channels'],
							   kernel_size=layer_config['conv1']['kernel_size'], stride=layer_config['conv1']['stride'],
							   padding=layer_config['conv1']['padding'])
		self.conv2 = nn.Conv3d(in_channels=layer_config['conv2']['in_channels'], out_channels=layer_config['conv2']['out_channels'],
							   kernel_size=layer_config['conv2']['kernel_size'], stride=layer_config['conv2']['stride'],
							   padding=layer_config['conv2']['padding'])
		self.conv3 = nn.Conv3d(in_channels=layer_config['conv3']['in_channels'], out_channels=layer_config['conv3']['out_channels'],
							   kernel_size=layer_config['conv3']['kernel_size'], stride=layer_config['conv3']['stride'],
							   padding=layer_config['conv3']['padding'])
		self.conv4 = nn.Conv3d(in_channels=layer_config['conv4']['in_channels'], out_channels=layer_config['conv4']['out_channels'],
							   kernel_size=layer_config['conv4']['kernel_size'], stride=layer_config['conv4']['stride'],
							   padding=layer_config['conv4']['padding'])
	
		
		self.fc1 = nn.Linear(layer_config['fc1']['in'] , layer_config['fc1']['out'])
		
		self.bn1 = nn.BatchNorm3d(layer_config['conv1']['out_channels'])
		self.bn2 = nn.BatchNorm3d(layer_config['conv2']['out_channels'])
		self.bn3 = nn.BatchNorm3d(layer_config['conv3']['out_channels'])
		self.bn4 = nn.BatchNorm3d(layer_config['conv4']['out_channels'])
		
		self.dropout3d = nn.Dropout3d(p=params['model']['conv_drop_prob'])
		self.dropout = nn.Dropout(params['model']['fcc_drop_prob'])
		
		self.mp3dL1 = nn.MaxPool3d(layer_config['maxpool3d']['layer1']['kernel'], layer_config['maxpool3d'][
			'layer1']['stride'])
		self.mp3dL2 = nn.MaxPool3d(layer_config['maxpool3d']['layer2']['kernel'], layer_config['maxpool3d'][
			'layer2']['stride'])
		self.mp3dL3 = nn.MaxPool3d(layer_config['maxpool3d']['layer3']['kernel'], layer_config['maxpool3d'][
			'layer3']['stride'])
		self.adaptiveMp3d = nn.AdaptiveMaxPool3d(layer_config['maxpool3d']['adaptive'])
		
		self.relu = nn.ReLU()
		self.logsoftmax = nn.LogSoftmax(dim=0)
		
		#self.maxpool3d = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(3, 1, 1))
				
		for m in self.modules():
			if isinstance(m, nn.Conv3d):
				#nn.init.kaiming_uniform(m.weight.data, mode='fan_in')
				nn.init.kaiming_normal(m.weight.data, mode='fan_in')
				#nn.init.xavier_uniform(m.weight.data)
				nn.init.constant(m.bias.data, 0.01)
			elif isinstance(m, nn.Linear):
				#nn.init.kaiming_uniform(m.weight.data, mode='fan_in')
				nn.init.kaiming_normal(m.weight.data, mode='fan_in')
				#nn.init.xavier_uniform(m.weight.data)
				nn.init.constant(m.bias.data, 0.01)
			elif isinstance(m, nn.BatchNorm3d):
				nn.init.constant(m.weight.data, 1)
				nn.init.constant(m.bias.data, 0.01)
	
	def forward(self, x):
		# print(x.size())
		out16 = self.relu(self.bn1(self.mp3dL1(self.conv1(x))))
		out16 = self.dropout3d(out16)
		# print(out16.size())
		out32 = self.relu(self.bn2(self.mp3dL2(self.conv2(out16))))
		out32 = self.dropout3d(out32)
		# print(out32.size())
		out64 = self.relu(self.bn3(self.mp3dL3(self.conv3(out32))))
		out64 = self.dropout3d(out64)
		# print(out64.size())
		out128 = self.relu(self.bn4(self.adaptiveMp3d(self.conv4(out64))))
		out128 = self.dropout3d(out128)
		# print(out128.size())
		flat = out128.view(out128.size(0), -1)
		# print(flat.size())
		fcc = self.logsoftmax(self.fc1(flat))
		return fcc