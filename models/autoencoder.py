'''
Convolutional Neural Network
'''
import torch.nn as nn
from configurations.modelConfig import layer_config, params, num_classes


class AutoEncoder(nn.Module):
	"""Cnn with simple architecture of 6 stacked convolution layers followed by a fully connected layer"""
	
	def __init__(self):
		super(AutoEncoder, self).__init__()
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
		
		self.tconv1 = nn.ConvTranspose3d(in_channels=layer_config['tconv1']['in_channels'],
								out_channels=layer_config['tconv1']['out_channels'],
								kernel_size=layer_config['tconv1']['kernel_size'],
								stride=layer_config['tconv1']['stride'],
								padding=layer_config['tconv1']['padding'],
								output_padding = layer_config['tconv1']['output_padding'])
		self.tconv2 = nn.ConvTranspose3d(in_channels=layer_config['tconv2']['in_channels'],
								out_channels=layer_config['tconv2']['out_channels'],
								kernel_size=layer_config['tconv2']['kernel_size'],
								stride=layer_config['tconv2']['stride'],
								padding=layer_config['tconv2']['padding'],
								output_padding=layer_config['tconv2']['output_padding'])
		self.tconv3 = nn.ConvTranspose3d(in_channels=layer_config['tconv3']['in_channels'],
								out_channels=layer_config['tconv3']['out_channels'],
								kernel_size=layer_config['tconv3']['kernel_size'],
								stride=layer_config['tconv3']['stride'],
								padding=layer_config['tconv3']['padding'],
								output_padding=layer_config['tconv3']['output_padding'])
		self.tconv4 = nn.ConvTranspose3d(in_channels=layer_config['tconv4']['in_channels'],
								out_channels=layer_config['tconv4']['out_channels'],
								kernel_size=layer_config['tconv4']['kernel_size'],
								stride=layer_config['tconv4']['stride'],
								padding=layer_config['tconv4']['padding'],
								output_padding=layer_config['tconv4']['output_padding'])
		
		self.fc1 = nn.Linear(layer_config['fc1']['in'], layer_config['fc1']['out'])
		self.fc2 = nn.Linear(layer_config['fc2']['in'], layer_config['fc2']['out'])
		# self.fc3 = nn.Linear(layer_config['fc3']['in'], layer_config['fc3']['out'])
		
		self.bn1 = nn.BatchNorm3d(layer_config['conv1']['out_channels'])
		self.bn2 = nn.BatchNorm3d(layer_config['conv2']['out_channels'])
		self.bn3 = nn.BatchNorm3d(layer_config['conv3']['out_channels'])
		self.bn4 = nn.BatchNorm3d(layer_config['conv4']['out_channels'])
		
		self.tbn1 = nn.BatchNorm3d(layer_config['tconv1']['out_channels'])
		self.tbn2 = nn.BatchNorm3d(layer_config['tconv2']['out_channels'])
		self.tbn3 = nn.BatchNorm3d(layer_config['tconv3']['out_channels'])
		self.tbn4 = nn.BatchNorm3d(layer_config['tconv4']['out_channels'])
		
		self.dropout3d = nn.Dropout3d(p=params['model']['conv_drop_prob'])
		self.dropout = nn.Dropout(params['model']['fcc_drop_prob'])
		
		self.maxpool = nn.MaxPool3d(layer_config['maxpool3d']['ln']['kernel'], layer_config['maxpool3d'][
			'ln']['stride'], padding=1)
		
		self.relu = nn.ReLU()
		self.logsoftmax = nn.LogSoftmax(dim=0)
		
		
		for m in self.modules():
			if isinstance(m, nn.Conv3d):
				nn.init.kaiming_uniform(m.weight.data, mode='fan_in')
				# nn.init.kaiming_normal(m.weight.data, mode='fan_in')
				nn.init.constant(m.bias.data, 0.01)
				
			if isinstance(m, nn.ConvTranspose3d):
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
	
	def encoder(self, x):
		#print('encoder : ', x.size())
		
		x = self.dropout3d(self.relu(self.bn1(self.conv1(x))))
		#print(x.size())
		
		x = self.dropout3d(self.relu(self.bn2(self.conv2(x))))
		#print(x.size())
		
		x = self.dropout3d(self.relu(self.bn3(self.conv3(x))))
		#print(x.size())
		
		x = self.dropout3d(self.relu(self.bn4(self.conv4(x))))
		#print(x.size())
		
		return x
	
	def decoder(self, x):
		#print('decoder : ', x.size())
		
		x = self.dropout3d(self.relu(self.tbn1(self.tconv1(x))))
		#print(x.size())
		
		x = self.dropout3d(self.relu(self.tbn2(self.tconv2(x))))
		#print(x.size())
		
		x = self.dropout3d(self.relu(self.tbn3(self.tconv3(x))))
		#print(x.size())
		
		x = self.dropout3d(self.relu(self.tbn4(self.tconv4(x))))
		#print(x.size())
		
		return x
		
	def classifier(self, x):
		x = x.view(x.size(0), -1)
		# print(x.size())
		
		x = self.dropout(self.relu(self.fc1(x)))
		# print(x.size())
		
		x = self.logsoftmax(self.fc2(x))
		
		return x
	
	# print(output.size())
	
	def forward(self, x):
		# encoder
		enc_x = self.encoder(x)
		
		# decoder
		x_hat = self.decoder(enc_x)
		
		# classifier
		class_prob = self.classifier(enc_x)
		
		return x_hat, class_prob