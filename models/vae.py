'''
Convolutional Neural Network with reconstrcution loss of euto-encoder as regularization
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from configurations.modelConfig import layer_config, params, num_classes
from configurations.modelConfig import img_shape


def crop(layer, target_size):
	#print('inside crop :', layer.size(), target_size)
	dif = [(layer.size()[2] - target_size[0]) // 2, (layer.size()[3] - target_size[1]) // 2, (layer.size()[4] -
																							target_size[2]) // 2]
	cs = target_size
	#print(cs, dif)
	
	layer = layer[:, :, dif[0]:dif[0] + cs[0], dif[1]:dif[1] + cs[1], dif[2]:dif[2] + cs[2]]
	#print(layer.size())
	
	return layer


class VAE(nn.Module):
	"""Cnn with simple architecture of 6 stacked convolution layers followed by a fully connected layer"""
	
	def __init__(self):
		super(VAE, self).__init__()
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
		
		
		self.tconv1 = nn.Conv3d(in_channels=layer_config['tconv1']['in_channels'],
										 out_channels=layer_config['tconv1']['out_channels'],
										 kernel_size=layer_config['tconv1']['kernel_size'],
										 stride=layer_config['tconv1']['stride'],
										 padding=layer_config['tconv1']['padding'])
		self.tconv2 = nn.Conv3d(in_channels=layer_config['tconv2']['in_channels'],
										 out_channels=layer_config['tconv2']['out_channels'],
										 kernel_size=layer_config['tconv2']['kernel_size'],
										 stride=layer_config['tconv2']['stride'],
										 padding=layer_config['tconv2']['padding'])
		self.tconv3 = nn.Conv3d(in_channels=layer_config['tconv3']['in_channels'],
										 out_channels=layer_config['tconv3']['out_channels'],
										 kernel_size=layer_config['tconv3']['kernel_size'],
										 stride=layer_config['tconv3']['stride'],
										 padding=layer_config['tconv3']['padding'])
		self.tconv4 = nn.Conv3d(in_channels=layer_config['tconv4']['in_channels'],
										 out_channels=layer_config['tconv4']['out_channels'],
										 kernel_size=layer_config['tconv4']['kernel_size'],
										 stride=layer_config['tconv4']['stride'],
										 padding=layer_config['tconv4']['padding'])
		
		self.fc_mean = nn.Linear(layer_config['gaussian'], layer_config['z_dim'])
		self.fc_logvar = nn.Linear(layer_config['gaussian'], layer_config['z_dim'])
		self.lineard = nn.Linear(layer_config['z_dim'], layer_config['gaussian'])

		self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
	
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
			'ln']['stride'], ceil_mode=True)
		
		self.relu = nn.ReLU()
		self.logsoftmax = nn.LogSoftmax(dim=0)
		
		self.shapes = []
		
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
		# print('encoder : ', x.size())
		self.shapes.append(x.size()[-3:])
		
		x = self.dropout3d(self.relu(self.bn1(self.maxpool(self.conv1(x)))))
		self.shapes.append(x.size()[-3:])
		#print(x.size())
		
		x = self.dropout3d(self.relu(self.bn2(self.maxpool(self.conv2(x)))))
		self.shapes.append(x.size()[-3:])
		#print(x.size())
		
		x = self.dropout3d(self.relu(self.bn3(self.maxpool(self.conv3(x)))))
		self.shapes.append(x.size()[-3:])
		#print(x.size())
		
		x = self.dropout3d(self.relu(self.bn4(self.maxpool(self.conv4(x)))))
		#self.shapes.append(x.size()[-3:])
		#print('encoder : ', self.shapes)
		#print(x.size())
		
		# flatten
		h = x.view(x.size(0), -1)
		#print(h.size())
		
		mu = self.dropout(self.relu(self.fc_mean(h)))
		logvar = self.dropout(self.relu(self.fc_logvar(h)))
		
		return mu, logvar
	
	def reparametrize(self, mu, logvar):
		sigma = torch.exp(0.5 * logvar)
		eps = Variable(torch.randn(mu.size())).cuda()
		z = mu + sigma * eps
		return z
	
	def decoder(self, z):
		#print('decoder : ', x.size())
		x_hat = self.dropout(self.relu(self.lineard(z)))
		#print(x_hat.size())
		x_hat = x_hat.view(x_hat.size(0), -1, int(img_shape[0]), int(img_shape[1]), int(img_shape[2]))
		#print(x_hat.size())
		
		x_hat = self.dropout3d(self.relu(self.tbn1(self.tconv1(self.upsample(x_hat)))))
		x_hat = crop(x_hat, self.shapes[-1])
		self.shapes.pop()
		#print(x_hat.size())
		
		x_hat = self.dropout3d(self.relu(self.tbn2(self.tconv2(self.upsample(x_hat)))))
		x_hat = crop(x_hat, self.shapes[-1])
		self.shapes.pop()
		#print(x_hat.size())
		
		x_hat = self.dropout3d(self.relu(self.tbn3(self.tconv3(self.upsample(x_hat)))))
		x_hat = crop(x_hat, self.shapes[-1])
		self.shapes.pop()
		#print(x_hat.size())
		
		x_hat = self.dropout3d(self.relu(self.tbn4(self.tconv4(self.upsample(x_hat)))))
		x_hat = crop(x_hat, self.shapes[-1])
		self.shapes.pop()
		#print(x_hat.size())
		
		return x_hat
	
	def classifier(self, z):
		
		# print(x.size())
		
		z = self.dropout(self.relu(self.fc1(z)))
		# print(x.size())
		
		prob = self.logsoftmax(self.fc2(z))
		
		return prob
	
	# print(output.size())
	
	def forward(self, x):
		# encoder
		mu, logvar = self.encoder(x)
		#print(mu.size(), logvar.size())
		
		#reparameterize
		z = self.reparametrize(mu, logvar)
		#print(z.size())
		
		# decoder
		x_hat = self.decoder(z)
		
		# classifier
		class_prob = self.classifier(z)
		
		return z, mu, logvar, x_hat, class_prob