'''
Convolutional Neural Network
'''
import torch.nn as nn
from configurations.modelConfig import layer_config, params, num_classes


class VGG(nn.Module):
	"""Cnn with simple architecture of 6 stacked convolution layers followed by a fully connected layer"""
	
	def __init__(self):
		super(VGG, self).__init__()
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
		
		self.conv5 = nn.Conv3d(in_channels=layer_config['conv5']['in_channels'],
							   out_channels=layer_config['conv5']['out_channels'],
							   kernel_size=layer_config['conv5']['kernel_size'], stride=layer_config['conv5']['stride'],
							   padding=layer_config['conv5']['padding'])
		self.conv6 = nn.Conv3d(in_channels=layer_config['conv6']['in_channels'],
							   out_channels=layer_config['conv6']['out_channels'],
							   kernel_size=layer_config['conv6']['kernel_size'], stride=layer_config['conv6']['stride'],
							   padding=layer_config['conv6']['padding'])
		self.conv7 = nn.Conv3d(in_channels=layer_config['conv7']['in_channels'],
							   out_channels=layer_config['conv7']['out_channels'],
							   kernel_size=layer_config['conv7']['kernel_size'], stride=layer_config['conv7']['stride'],
							   padding=layer_config['conv7']['padding'])
		self.conv8 = nn.Conv3d(in_channels=layer_config['conv8']['in_channels'],
							   out_channels=layer_config['conv8']['out_channels'],
							   kernel_size=layer_config['conv8']['kernel_size'], stride=layer_config['conv8']['stride'],
							   padding=layer_config['conv8']['padding'])
		self.conv9 = nn.Conv3d(in_channels=layer_config['conv9']['in_channels'],
							   out_channels=layer_config['conv9']['out_channels'],
							   kernel_size=layer_config['conv9']['kernel_size'], stride=layer_config['conv9']['stride'],
							   padding=layer_config['conv9']['padding'])
		self.conv10 = nn.Conv3d(in_channels=layer_config['conv10']['in_channels'],
								out_channels=layer_config['conv10']['out_channels'],
								kernel_size=layer_config['conv10']['kernel_size'],
								stride=layer_config['conv10']['stride'],
								padding=layer_config['conv10']['padding'])
		self.conv11 = nn.Conv3d(in_channels=layer_config['conv11']['in_channels'],
								out_channels=layer_config['conv11']['out_channels'],
								kernel_size=layer_config['conv11']['kernel_size'],
								stride=layer_config['conv11']['stride'],
								padding=layer_config['conv11']['padding'])
		self.conv12 = nn.Conv3d(in_channels=layer_config['conv12']['in_channels'],
								out_channels=layer_config['conv12']['out_channels'],
								kernel_size=layer_config['conv12']['kernel_size'],
								stride=layer_config['conv12']['stride'],
								padding=layer_config['conv12']['padding'])
		self.conv13 = nn.Conv3d(in_channels=layer_config['conv13']['in_channels'],
								out_channels=layer_config['conv13']['out_channels'],
								kernel_size=layer_config['conv13']['kernel_size'],
								stride=layer_config['conv13']['stride'],
								padding=layer_config['conv13']['padding'])
		self.conv14 = nn.Conv3d(in_channels=layer_config['conv14']['in_channels'],
								out_channels=layer_config['conv14']['out_channels'],
								kernel_size=layer_config['conv14']['kernel_size'],
								stride=layer_config['conv14']['stride'],
								padding=layer_config['conv14']['padding'])
		self.conv15 = nn.Conv3d(in_channels=layer_config['conv15']['in_channels'],
								out_channels=layer_config['conv15']['out_channels'],
								kernel_size=layer_config['conv15']['kernel_size'],
								stride=layer_config['conv15']['stride'],
								padding=layer_config['conv15']['padding'])
		self.conv16 = nn.Conv3d(in_channels=layer_config['conv16']['in_channels'],
								out_channels=layer_config['conv16']['out_channels'],
								kernel_size=layer_config['conv16']['kernel_size'],
								stride=layer_config['conv16']['stride'],
								padding=layer_config['conv16']['padding'])
		
		self.fc1 = nn.Linear(layer_config['fc1']['in'], layer_config['fc1']['out'])
		self.fc2 = nn.Linear(layer_config['fc2']['in'], layer_config['fc2']['out'])
		self.fc3 = nn.Linear(layer_config['fc3']['in'], layer_config['fc3']['out'])
		self.final = nn.Linear(layer_config['final']['in'], layer_config['final']['out'])
		
		self.bn1 = nn.BatchNorm3d(layer_config['conv1']['out_channels'])
		self.bn2 = nn.BatchNorm3d(layer_config['conv2']['out_channels'])
		self.bn3 = nn.BatchNorm3d(layer_config['conv3']['out_channels'])
		self.bn4 = nn.BatchNorm3d(layer_config['conv4']['out_channels'])
		self.bn5 = nn.BatchNorm3d(layer_config['conv5']['out_channels'])
		self.bn6 = nn.BatchNorm3d(layer_config['conv6']['out_channels'])
		self.bn7 = nn.BatchNorm3d(layer_config['conv7']['out_channels'])
		self.bn8 = nn.BatchNorm3d(layer_config['conv8']['out_channels'])
		self.bn9 = nn.BatchNorm3d(layer_config['conv9']['out_channels'])
		self.bn10 = nn.BatchNorm3d(layer_config['conv10']['out_channels'])
		self.bn11 = nn.BatchNorm3d(layer_config['conv11']['out_channels'])
		self.bn12 = nn.BatchNorm3d(layer_config['conv12']['out_channels'])
		self.bn13 = nn.BatchNorm3d(layer_config['conv13']['out_channels'])
		self.bn14 = nn.BatchNorm3d(layer_config['conv14']['out_channels'])
		self.bn15 = nn.BatchNorm3d(layer_config['conv15']['out_channels'])
		self.bn16 = nn.BatchNorm3d(layer_config['conv16']['out_channels'])
		
		self.dropout3d = nn.Dropout3d(p=params['model']['conv_drop_prob'])
		self.dropout = nn.Dropout(params['model']['fcc_drop_prob'])
		
		self.maxpool1 = nn.MaxPool3d(layer_config['maxpool3d']['l1']['kernel'], layer_config['maxpool3d'][
			'l1']['stride'])
		self.maxpool = nn.MaxPool3d(layer_config['maxpool3d']['ln']['kernel'], layer_config['maxpool3d'][
			'ln']['stride'])
		
		self.adaptiveMp3d = nn.AdaptiveMaxPool3d(layer_config['maxpool3d']['adaptive'])
		
		self.relu = nn.ReLU()
		self.logsoftmax = nn.LogSoftmax(dim=0)
		
		# self.maxpool3d = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(3, 1, 1))
		
		for m in self.modules():
			if isinstance(m, nn.Conv3d):
				nn.init.kaiming_uniform(m.weight.data, mode='fan_in')
				# nn.init.kaiming_normal(m.weight.data, mode='fan_in')
				# nn.init.xavier_uniform(m.weight.data)
				nn.init.constant(m.bias.data, 0.01)
			elif isinstance(m, nn.Linear):
				nn.init.kaiming_uniform(m.weight.data, mode='fan_in')
				# nn.init.kaiming_normal(m.weight.data, mode='fan_in')
				# nn.init.xavier_uniform(m.weight.data)
				nn.init.constant(m.bias.data, 0.01)
			elif isinstance(m, nn.BatchNorm3d):
				nn.init.constant(m.weight.data, 1)
				nn.init.constant(m.bias.data, 0.01)
	
	def forward(self, x):
		# print(x.size())
		
		out1 = self.dropout3d(self.relu(self.bn1(self.conv1(x))))
		# print('1 : ', out1.size())
		
		out2 = self.dropout3d(self.maxpool(self.relu(self.bn2(self.conv2(out1)))))
		# print('2 : ', out2.size())
		
		out3 = self.dropout3d(self.relu(self.bn3(self.conv3(out2))))
		# print('3 : ', out3.size())
		
		out4 = self.dropout3d(self.maxpool(self.relu(self.bn4(self.conv4(out3)))))
		# print('4 : ', out4.size())
		
		out5 = self.dropout3d(self.relu(self.bn5(self.conv5(out4))))
		# print('5 : ', out5.size())
		
		out6 = self.dropout3d(self.relu(self.bn6(self.conv6(out5))))
		# print('6 : ', out6.size())
		
		out7 = self.dropout3d(self.relu(self.bn7(self.conv7(out6))))
		# print('7 : ', out7.size())
		
		out8 = self.dropout3d(self.maxpool(self.relu(self.bn8(self.conv8(out7)))))
		# print('8 : ', out8.size())
		
		out9 = self.dropout3d(self.relu(self.bn9(self.conv9(out8))))
		# print('9 : ', out9.size())
		
		out10 = self.dropout3d(self.relu(self.bn10(self.conv10(out9))))
		# print('10 : ', out10.size())
		
		out11 = self.dropout3d(self.relu(self.bn11(self.conv11(out10))))
		# print('11 : ', out11.size())
		
		out12 = self.dropout3d(self.maxpool(self.relu(self.bn12(self.conv12(out11)))))
		# print('12 : ', out12.size())
		
		out13 = self.dropout3d(self.relu(self.bn13(self.conv13(out12))))
		# print('13 : ', out13.size())
		
		out14 = self.dropout3d(self.relu(self.bn14(self.conv14(out13))))
		# print('14 : ', out14.size())
		
		out15 = self.dropout3d(self.relu(self.bn15(self.conv15(out14))))
		# print('15 : ', out15.size())
		
		out16 = self.dropout3d(self.relu(self.bn16(self.conv16(out15))))
		# print('16 : ', out16.size())
		
		flat = out16.view(out16.size(0), -1)
		# print(flat.size())
		
		fcc1 = self.dropout(self.relu(self.fc1(flat)))
		# print(fcc1.size())
		
		fcc2 = self.dropout(self.relu(self.fc2(fcc1)))
		fcc3 = self.dropout(self.relu(self.fc3(fcc2)))
		
		# output = self.dropout(self.relu(self.fc2(fcc1)))
		# print(fcc2.size())
		
		output = self.logsoftmax(self.final(fcc3))
		# print(output.size())
		
		return output