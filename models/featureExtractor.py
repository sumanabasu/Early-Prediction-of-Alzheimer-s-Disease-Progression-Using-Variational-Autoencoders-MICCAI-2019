import torch.nn as nn
from configurations.modelConfig import layer_config

class FeatureExtractor(nn.Module):
	def __init__(self):
		super(FeatureExtractor, self).__init__()
		
		self.conv1 = nn.Conv3d(in_channels=layer_config['conv1']['in_channels'],
							   out_channels=layer_config['conv1']['out_channels'],
							   kernel_size=layer_config['conv1']['kernel_size'], stride=layer_config['conv1']['stride'],
							   padding=layer_config['conv1']['padding'])
		self.conv2 = nn.Conv3d(in_channels=layer_config['conv2']['in_channels'],
							   out_channels=layer_config['conv2']['out_channels'],
							   kernel_size=layer_config['conv2']['kernel_size'], stride=layer_config['conv2']['stride'],
							   padding=layer_config['conv2']['padding'])
		self.conv3 = nn.Conv3d(in_channels=layer_config['conv3']['in_channels'],
							   out_channels=layer_config['conv3']['out_channels'],
							   kernel_size=layer_config['conv3']['kernel_size'], stride=layer_config['conv3']['stride'],
							   padding=layer_config['conv3']['padding'])
		self.conv4 = nn.Conv3d(in_channels=layer_config['conv4']['in_channels'],
							   out_channels=layer_config['conv4']['out_channels'],
							   kernel_size=layer_config['conv4']['kernel_size'], stride=layer_config['conv4']['stride'],
							   padding=layer_config['conv4']['padding'])
		
		self.bn1 = nn.BatchNorm3d(layer_config['conv1']['out_channels'])
		self.bn2 = nn.BatchNorm3d(layer_config['conv2']['out_channels'])
		self.bn3 = nn.BatchNorm3d(layer_config['conv3']['out_channels'])
		self.bn4 = nn.BatchNorm3d(layer_config['conv4']['out_channels'])
		
		self.fc = nn.Linear(layer_config['fc']['in'], layer_config['fc']['out'])
		
		self.maxpool = nn.MaxPool3d(layer_config['maxpool3d']['ln']['kernel'], layer_config['maxpool3d'][
			'ln']['stride'], ceil_mode=True)
		
		self.relu = nn.ReLU()
		
		for m in self.modules():
			if isinstance(m, nn.Conv3d):
				nn.init.kaiming_uniform(m.weight.data, mode='fan_in')
				# nn.init.kaiming_normal(m.weight.data, mode='fan_in')
				nn.init.constant(m.bias.data, 0.01)
			
			elif isinstance(m, nn.Linear):
				nn.init.kaiming_uniform(m.weight.data, mode='fan_in')
				# nn.init.kaiming_normal(m.weight.data, mode='fan_in')
				nn.init.constant(m.bias.data, 0.01)
			
			elif isinstance(m, nn.BatchNorm3d):
				nn.init.constant(m.weight.data, 1)
				nn.init.constant(m.bias.data, 0.01)
		
	def forward(self, image):
		# print('encoder : ', x.size())
		#self.shapes.append(image.size()[-3:])
		
		x = (self.relu(self.bn1(self.maxpool(self.conv1(image)))))
		# self.shapes.append(x.size()[-3:])
		# print(x.size())
		
		x = (self.relu(self.bn2(self.maxpool(self.conv2(x)))))
		# self.shapes.append(x.size()[-3:])
		# print(x.size())
		
		x = (self.relu(self.bn3(self.maxpool(self.conv3(x)))))
		# self.shapes.append(x.size()[-3:])
		# print(x.size())
		
		x = (self.relu(self.bn4(self.maxpool(self.conv4(x)))))
		# self.shapes.append(x.size()[-3:])
		# print('encoder : ', self.shapes)
		# print(x.size())
		
		x = x.contiguous().view(x.size(0), -1)
		
		x = self.relu(self.fc(x))
		# print(x.size())
	
		return x
		
		