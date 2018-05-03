'''
train model
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from modelConfig import params
import math
from tqdm import tqdm

class Trainer():
	def __init__(self, model, train_loader, valid_loader, config):
		super(Trainer, self).__init__()
		
		if torch.cuda.is_available():
			self.model = model.cuda()
			
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.optimizer = torch.optim.Adam(model.parameters(),
										  lr=params['train']['learning_rate'])
		self.criterion = nn.NLLLoss(weight=torch.FloatTensor(params['train']['label_weights']).cuda())
		self.curr_epoch = 0
	
	def train(self):
		for _ in range(params['train']['num_epochs']):
			print('Training...\nEpoch : '+str(_))
			self.curr_epoch += 1
			
			# Train Model
			self.trainEpoch()
			
			# Validate Model
			print ('Validation...')
			self.model.eval()
			self.validate()
	
	def trainEpoch(self):
		pbt = tqdm(total=int(math.ceil((1.0 * len(self.train_loader)) / params['train']['batch_size'])))
		
		for batch_idx, (images, labels) in enumerate(self.train_loader):
			accuracy, loss = self.trainBatch(batch_idx, images, labels)
			pbt.update(1)
		
		pbt.close()
	
	def trainBatch(self, batch_idx, images, labels):
		images = Variable(images).cuda()
		labels = Variable(labels).cuda()
		labels = labels.view(-1, )
		
		# Forward + Backward + Optimize
		self.optimizer.zero_grad()
		outputs = self.model(images)
		
		loss = self.criterion(outputs, labels)
		loss.backward()
		
		self.optimizer.step()
		
		# Compute accuracy
		_, argmax = torch.max(outputs, 1)
		accuracy = (labels == argmax).float().mean()
		
		# Print metrics
		if batch_idx % 100 == 0:
			print('Epoch [%d/%d], Batch [%d/%d] Loss: %.4f Accuracy: %0.2f'
				  % (self.curr_epoch, params['train']['num_epochs'], batch_idx,
					 math.ceil(len(self.train_loader) / params['train']['batch_size']),
					 loss.data[0], accuracy))
		
		# clean GPU
		del images, labels, outputs
		
		return accuracy, loss.data[0]
	
	def validate(self):
		# TODO : Implement Validation
		pass
	
	def test(self):
		# TODO : implement test
		pass