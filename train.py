'''
train model
'''
import torch

class Trainer():
	def __init__(self, model, train_data, config):
		super(Train, self).__init__()
		if torch.cuda.is_available():
			self.model = model.cuda()
		
	def trainEpoch(self):
		pass
	
	def trainStep(self):
		pass