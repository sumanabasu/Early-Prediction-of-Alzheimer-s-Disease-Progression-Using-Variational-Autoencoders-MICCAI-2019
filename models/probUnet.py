import  torch
import torch.nn as nn
from configurations.modelConfig import posterior_layer_config, prior_layer_config, generator_layer_config
from torch.distributions.normal import Normal
from torch.autograd import Variable

class posterior(nn.Module):
	'''
	Input : (X_{t+1}, y_{t+1})
	X_{t+1} is the MRI at timestamp t+1
	y_{t+1} is the one hot encoded label at timestamp t+1
	output : z
	'''
	def __init__(self):
		super(posterior, self).__init__()
		self.conv1 = nn.Conv3d(in_channels=posterior_layer_config['conv1']['in_channels'],
							   out_channels=posterior_layer_config['conv1']['out_channels'],
							   kernel_size=posterior_layer_config['conv1']['kernel_size'], stride=posterior_layer_config['conv1']['stride'],
							   padding=posterior_layer_config['conv1']['padding'])
		self.conv2 = nn.Conv3d(in_channels=posterior_layer_config['conv2']['in_channels'],
							   out_channels=posterior_layer_config['conv2']['out_channels'],
							   kernel_size=posterior_layer_config['conv2']['kernel_size'], stride=posterior_layer_config['conv2']['stride'],
							   padding=posterior_layer_config['conv2']['padding'])
		self.conv3 = nn.Conv3d(in_channels=posterior_layer_config['conv3']['in_channels'],
							   out_channels=posterior_layer_config['conv3']['out_channels'],
							   kernel_size=posterior_layer_config['conv3']['kernel_size'], stride=posterior_layer_config['conv3']['stride'],
							   padding=posterior_layer_config['conv3']['padding'])
		self.conv4 = nn.Conv3d(in_channels=posterior_layer_config['conv4']['in_channels'],
							   out_channels=posterior_layer_config['conv4']['out_channels'],
							   kernel_size=posterior_layer_config['conv4']['kernel_size'], stride=posterior_layer_config['conv4']['stride'],
							   padding=posterior_layer_config['conv4']['padding'])
		
		self.bn1 = nn.BatchNorm3d(posterior_layer_config['conv1']['out_channels'])
		self.bn2 = nn.BatchNorm3d(posterior_layer_config['conv2']['out_channels'])
		self.bn3 = nn.BatchNorm3d(posterior_layer_config['conv3']['out_channels'])
		self.bn4 = nn.BatchNorm3d(posterior_layer_config['conv4']['out_channels'])
		
		self.fc_mean = nn.Linear(posterior_layer_config['gaussian'], posterior_layer_config['z_dim'])
		self.fc_var = nn.Linear(posterior_layer_config['gaussian'], posterior_layer_config['z_dim'])
		
		self.maxpool = nn.MaxPool3d(posterior_layer_config['maxpool3d']['ln']['kernel'], posterior_layer_config['maxpool3d'][
			'ln']['stride'], ceil_mode=True)
		
		self.relu = nn.ReLU()
		
	def forward(self, x_t_plus_1, y_t_plus_1):
		
		#extract feature vector from 3D MRI of next timestamp
		feature = (self.relu(self.bn1(self.maxpool(self.conv1(x_t_plus_1)))))
		feature = (self.relu(self.bn2(self.maxpool(self.conv2(feature)))))
		feature = (self.relu(self.bn3(self.maxpool(self.conv3(feature)))))
		feature = (self.relu(self.bn4(self.maxpool(self.conv4(feature)))))
		# flatten
		flat_feature = feature.view(feature.size(0), -1)
		
		#append one hot encoded label to the extracted feature vector
		z_concatenated = torch.cat([flat_feature, y_t_plus_1], dim=0)
		
		mu = self.fc_mean(z_concatenated)
		var = self.fc_var(z_concatenated)
		
		'''
		# to return a Normal distribution object
		posterior_distribution = Normal(mu, var)
		return  posterior_distribution
		'''
		
		return mu, var


class prior(nn.Module):
	'''
	Input : X_t
	X_t is the MRI at timestamp t
	output : z
	'''
	def __init__(self):
		super(prior, self).__init__()
		self.conv1 = nn.Conv3d(in_channels=prior_layer_config['conv1']['in_channels'],
							   out_channels=prior_layer_config['conv1']['out_channels'],
							   kernel_size=prior_layer_config['conv1']['kernel_size'],
							   stride=prior_layer_config['conv1']['stride'],
							   padding=prior_layer_config['conv1']['padding'])
		self.conv2 = nn.Conv3d(in_channels=prior_layer_config['conv2']['in_channels'],
							   out_channels=prior_layer_config['conv2']['out_channels'],
							   kernel_size=prior_layer_config['conv2']['kernel_size'],
							   stride=prior_layer_config['conv2']['stride'],
							   padding=prior_layer_config['conv2']['padding'])
		self.conv3 = nn.Conv3d(in_channels=prior_layer_config['conv3']['in_channels'],
							   out_channels=prior_layer_config['conv3']['out_channels'],
							   kernel_size=prior_layer_config['conv3']['kernel_size'],
							   stride=prior_layer_config['conv3']['stride'],
							   padding=prior_layer_config['conv3']['padding'])
		self.conv4 = nn.Conv3d(in_channels=prior_layer_config['conv4']['in_channels'],
							   out_channels=prior_layer_config['conv4']['out_channels'],
							   kernel_size=prior_layer_config['conv4']['kernel_size'],
							   stride=prior_layer_config['conv4']['stride'],
							   padding=prior_layer_config['conv4']['padding'])
		
		self.bn1 = nn.BatchNorm3d(prior_layer_config['conv1']['out_channels'])
		self.bn2 = nn.BatchNorm3d(prior_layer_config['conv2']['out_channels'])
		self.bn3 = nn.BatchNorm3d(prior_layer_config['conv3']['out_channels'])
		self.bn4 = nn.BatchNorm3d(prior_layer_config['conv4']['out_channels'])
		
		self.fc_mean = nn.Linear(prior_layer_config['gaussian'], prior_layer_config['z_dim'])
		self.fc_var = nn.Linear(prior_layer_config['gaussian'], prior_layer_config['z_dim'])
		
		self.maxpool = nn.MaxPool3d(prior_layer_config['maxpool3d']['ln']['kernel'],
									prior_layer_config['maxpool3d'][
										'ln']['stride'], ceil_mode=True)
		
		self.relu = nn.ReLU()
	
	def forward(self, x_t):
		# extract feature vector from 3D MRI of next timestamp
		feature = (self.relu(self.bn1(self.maxpool(self.conv1(x_t)))))
		feature = (self.relu(self.bn2(self.maxpool(self.conv2(feature)))))
		feature = (self.relu(self.bn3(self.maxpool(self.conv3(feature)))))
		feature = (self.relu(self.bn4(self.maxpool(self.conv4(feature)))))
		# flatten
		z_flat = feature.view(feature.size(0), -1)
		
		mu = self.fc_mean(z_flat)
		var = self.fc_var(z_flat)
		
		'''
		# to return a Normal distribution object
		prior_distribution = Normal(mu, var)
		return  posterior_distribution
		'''
		
		return mu, var
	
class generator(nn.Module):
	'''
	reconstructs labels
	'''
	def __init__(self):
		super(generator, self).__init__()
		self.fc1 = nn.Linear(generator_layer_config['fc1']['in'], generator_layer_config['fc1']['out'])
		self.fc2 = nn.Linear(generator_layer_config['fc2']['in'], generator_layer_config['fc2']['out'])
		self.relu = nn.ReLU()
		self.logsoftmax = nn.LogSoftmax(dim=1)
	
	def reparametrize(self, mu, logvar):
		sigma = torch.exp(0.5 * logvar)
		eps = Variable(torch.randn(mu.size())).cuda()
		z = mu + sigma * eps
		return z
	
	def forward(self, z):
		z = (self.relu(self.fc1(z)))
		prob = self.logsoftmax(self.fc2(z))
		return prob
	
class probCNN(nn.Module, prior, posterior, generator):
	def __init__(self):
		super(probCNN, self).__init__()
		self.prior = prior
		self.posterior = posterior
		self.generator = generator
		
	def forward(self, inference, x_t = None, x_t_plus_1 = None, y_t_plus_1 = None):
		if inference:
			mu, var = self.prior(x_t)
		else:
			mu, var = self.posterior(x_t_plus_1, y_t_plus_1)
			
		z = self.generator.reparametrize(mu, var)
		y_hat_t_plus_1 = self.generator(z)
		
		return y_hat_t_plus_1
		
		
		
		
	
	
	
		
		