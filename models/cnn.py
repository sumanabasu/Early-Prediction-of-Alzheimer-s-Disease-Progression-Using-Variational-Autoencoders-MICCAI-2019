'''
Convolutional Neural Network
'''
import torch.nn as nn
from  modelConfig import layer_config, params

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
		self.conv5 = nn.Conv3d(in_channels=layer_config['conv5']['in_channels'], out_channels=layer_config['conv5']['out_channels'],
							   kernel_size=layer_config['conv5']['kernel_size'], stride=layer_config['conv5']['stride'],
							   padding=layer_config['conv5']['padding'])
		
		self.fc1 = nn.Linear(layer_config['fc1']['in'] , layer_config['fc1']['out'])
		self.fc2 = nn.Linear(layer_config['fc2']['in'], layer_config['fc2']['out'])
		self.fc3 = nn.Linear(layer_config['fc3']['in'], layer_config['fc3']['out'])
		self.fc4 = nn.Linear(layer_config['fc4']['in'], layer_config['fc4']['out'])
		
		self.bn1 = nn.BatchNorm3d(layer_config['conv1']['out_channels'])
		self.bn2 = nn.BatchNorm3d(layer_config['conv2']['out_channels'])
		self.bn3 = nn.BatchNorm3d(layer_config['conv3']['out_channels'])
		self.bn4 = nn.BatchNorm3d(layer_config['conv4']['out_channels'])
		self.bn5 = nn.BatchNorm3d(layer_config['conv5']['out_channels'])
		
		self.dropout3d = nn.Dropout3d(p=params['model']['conv_drop_prob'])
		self.dropout = nn.Dropout(params['model']['fcc_drop_prob'])
		
		self.relu = nn.ReLU()
		self.logsoftmax = nn.LogSoftmax()
		
		#self.maxpool3d = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(3, 1, 1))
		
		for m in self.modules():
			if isinstance(m, nn.Conv3d):
				nn.init.xavier_uniform(m.weight.data)
				nn.init.constant(m.bias.data, 0.01)
			elif isinstance(m, nn.Linear):
				nn.init.xavier_uniform(m.weight.data)
				nn.init.constant(m.bias.data, 0.01)
			elif isinstance(m, nn.BatchNorm3d):
				nn.init.constant(m.weight.data, 1)
				nn.init.constant(m.bias.data, 0.01)
	
	def forward(self, x):
		# reduce depth
		#x = self.maxpool3d(x)
		# print(x.size())
		# print "inside forward"
		out16 = self.relu(self.bn1(self.conv1(x)))
		out16 = self.dropout3d(out16)
		# print(out16.size())
		out32 = self.relu(self.bn2(self.conv2(out16)))
		out32 = self.dropout3d(out32)
		# print(out32.size())
		out64 = self.relu(self.bn3(self.conv3(out32)))
		out64 = self.dropout3d(out64)
		# print(out64.size())
		out128 = self.relu(self.bn4(self.conv4(out64)))
		out128 = self.dropout3d(out128)
		# print(out128.size())
		out256 = self.relu(self.bn5(self.conv5(out128)))
		out256 = self.dropout3d(out256)
		# print(out256.size())
		flat = out256.view(out256.size(0), -1)
		# print(flat.size())
		fcc1 = self.dropout(self.relu(self.fc1(flat)))
		# print(fcc1.size())
		fcc2 = self.dropout(self.relu(self.fc2(fcc1)))
		# print fcc.size()
		fcc3 = self.dropout(self.relu(self.fc3(fcc2)))
		fcc4 = self.logsoftmax(self.fc4(fcc3))
		return fcc4