import torch
import torch.nn as nn
from torch.autograd import Variable
from featureExtractor import FeatureExtractor
from configurations.modelConfig import num_classes, layer_config
import numpy as np

def oneHot(label):
	label = label.view(label.size(0), 1)
	onehot = torch.zeros(label.size(0), num_classes)
	onehot.scatter_(1, label.data.cpu(), 1)
	onehot = Variable(onehot).cuda()
	return onehot

class CVAE(nn.Module):
	def __init__(self):
		super(CVAE, self).__init__()
		self.extractFeatures = FeatureExtractor()
		
		self.fc_post = nn.Linear(layer_config['fc_post']['in'], layer_config['fc_post']['out'])
		self.fc_mu = nn.Linear(layer_config['fc_mu']['in'], layer_config['fc_mu']['out'])
		self.fc_logvar = nn.Linear(layer_config['fc_logvar']['in'], layer_config['fc_logvar']['out'])
		
		self.fc_gen1 = nn.Linear(layer_config['fc_gen1']['in'], layer_config['fc_gen1']['out'])
		self.fc_gen2 = nn.Linear(layer_config['fc_gen2']['in'], layer_config['fc_gen2']['out'])
		
		self.relu = nn.ReLU()
		self.logsoftmax = nn.LogSoftmax(dim=1)
		
	def priorNetwork(self, feature):
		# TODO : implement prior network
		pass
	
	def posteriorNetwork(self, feature, label):
		'''
		Learns parameters of Gaussian Distribution
		:param feature, label:
		:return mu, logvar:
		'''
		
		feature_label = torch.cat((feature, label), dim=1)
		
		post = self.relu(self.fc_post(feature_label))
		mu = self.fc_mu(post)
		logvar = self.fc_logvar(post)
		
		return mu, logvar
	
	def reparameterize(self, mu, logvar):
		sigma = torch.exp(0.5 * logvar)
		eps = Variable(torch.randn(mu.size())).cuda()
		z = mu + sigma * eps
		return z
	
	def generateLabels(self, z):
		gen = self.relu(self.fc_gen1(z))
		generated_class_prob = self.logsoftmax(self.fc_gen2(gen))
		
		return  generated_class_prob
	
		
	def forward(self, image, label):
		# extract features from MRI
		feature = self.extractFeatures(image)
		
		# one hot encoding of label
		label_onehot = oneHot(label)
		
		mu, logvar = self.posteriorNetwork(feature, label_onehot)
		
		z = self.reparameterize(mu, logvar)
		
		generated_class_prob = self.generateLabels(z)
		
		return generated_class_prob, mu, logvar
		
		
		
		