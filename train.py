'''
train model
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from modelConfig import params
import math
from tqdm import tqdm
import numpy as np
from visualizations import plot_confusion_matrix, plot_accuracy
from tensorboardX import SummaryWriter

class Trainer(object):
	def __init__(self, model, train_loader, valid_loader, expt_folder):
		super(Trainer, self).__init__()
		
		if torch.cuda.is_available():
			self.model = model.cuda()
			
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.optimizer = torch.optim.Adam(model.parameters(),
										  lr=params['train']['learning_rate'])
		self.criterion = nn.NLLLoss(weight=torch.FloatTensor(params['train']['label_weights']).cuda())
		self.curr_epoch = 0
		
		self.expt_folder = expt_folder
		self.writer = SummaryWriter(log_dir=expt_folder)
	
	def train(self):
		train_losses = []
		train_accuracy = []
		
		for _ in range(params['train']['num_epochs']):
			print('Training...\nEpoch : '+str(_))
			self.curr_epoch += 1
			
			# Train Model
			accuracy, loss = self.trainEpoch()
			
			train_losses.append(loss)
			train_accuracy.append(accuracy)
			
			# Validate Model
			print ('Validation...')
			self.model.eval()
			self.validate()
			
			# TODO : Save model
		
			# TODO : Save accuracy and loss to disk
	
	def trainEpoch(self):
		pbt = tqdm(total=int(math.ceil((1.0 * len(self.train_loader)) / params['train']['batch_size'])))
		
		minibacth_losses, minibacth_accuracy, actual_labels, predicted_labels = ([] for i in range(4))
		
		for batch_idx, (images, labels) in enumerate(self.train_loader):
			accuracy, loss, pred_labels = self.trainBatch(batch_idx, images, labels)
			
			minibacth_losses.append(loss)
			minibacth_accuracy.append(accuracy)
			
			actual_labels.extend(labels.data.cpu().numpy())
			predicted_labels.extend(pred_labels)
			
			pbt.update(1)
		
		pbt.close()
			
		# Plot losses
		self.writer.add_scalar('train_loss', np.mean(minibacth_losses), self.curr_epoch)
		self.writer.add_scalar('train_accuracy', np.mean(minibacth_accuracy), self.curr_epoch)
		
		# Plot confusion matrices
		plot_confusion_matrix(actual_labels, predicted_labels, title='Confusion matrix, without normalization (Train)')
		plot_confusion_matrix(actual_labels, predicted_labels, normalize=True, title='Normalized confusion matrix ('
																					 'Train)')
		
		return (np.mean(minibacth_accuracy), np.mean(minibacth_losses))
	
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
		_, pred_labels = torch.max(outputs, 1)
		accuracy = (labels == pred_labels).float().mean()
		
		# Print metrics
		if batch_idx % 100 == 0:
			print('Epoch [%d/%d], Batch [%d/%d] Loss: %.4f Accuracy: %0.2f'
				  % (self.curr_epoch, params['train']['num_epochs'], batch_idx,
					 math.ceil(len(self.train_loader) / params['train']['batch_size']),
					 loss.data[0], accuracy))
		
		# clean GPU
		del images, labels, outputs
		
		return accuracy, loss.data[0], pred_labels.data.cpu().numpy()
	
	def validate(self):
		# TODO : Implement Validation
		pass
	
	def test(self):
		# TODO : implement test
		pass