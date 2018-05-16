''''
train model
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from modelConfig import params
from tqdm import tqdm
import numpy as np
from visualizations import plot_confusion_matrix, plot_accuracy
from tensorboardX import SummaryWriter
from save import saveModelandMetrics

class Trainer(object):
	def __init__(self, model, train_loader, valid_loader, expt_folder):
		super(Trainer, self).__init__()
		
		if torch.cuda.is_available():
			self.model = model.cuda()
			
		self.train_loader = train_loader # TODO: change to train_loader
		self.valid_loader = valid_loader # TODO: change to valid_loader
		self.optimizer = torch.optim.Adam(model.parameters(),
										  lr=params['train']['learning_rate'])
		self.criterion = nn.NLLLoss(weight=torch.FloatTensor(params['train']['label_weights']).cuda())
		self.curr_epoch = 0
		self.batchstep = 0
		
		self.expt_folder = expt_folder
		self.writer = SummaryWriter(log_dir=expt_folder)
		
		self.train_losses, self.valid_losses, self.train_accuracy, self.valid_accuracy = ([] for i in range(4))
	
	def train(self):
		
		for _ in range(params['train']['num_epochs']):
			print('Training...\nEpoch : '+str(self.curr_epoch))
			
			# Train Model
			accuracy, loss = self.trainEpoch()
			
			self.train_losses.append(loss)
			self.train_accuracy.append(accuracy)
			
			# Validate Model
			print ('Validation...')
			self.model.eval()
			self.validate()
			
			# Save model
			saveModelandMetrics(self)
			# TODO : Stop learning if model doesn't improve for 10 epochs --> Save best model
			
			self.curr_epoch += 1
	
	def trainEpoch(self):
		pbt = tqdm(total=len(self.train_loader))
		
		actual_labels, predicted_labels = ([] for i in range(2))
		
		# TODO : shuffle dataloader after every epoch
		
		for batch_idx, (images, labels) in enumerate(self.train_loader):
			minibatch_losses = 0
			minibatch_accuracy = 0
			
			accuracy, loss, pred_labels = self.trainBatch(batch_idx, images, labels)
			
			minibatch_losses += loss
			minibatch_accuracy += accuracy
			
			actual_labels.extend(labels)
			predicted_labels.extend(pred_labels)
			
			pbt.update(1)
		
		pbt.close()
		
		minibatch_losses /= len(self.train_loader)
		minibatch_accuracy /= len(self.train_loader)
			
		# Plot losses
		self.writer.add_scalar('train_loss', minibatch_losses , self.curr_epoch)
		self.writer.add_scalar('train_accuracy', minibatch_accuracy, self.curr_epoch)
		
		# Plot confusion matrices
		plot_confusion_matrix(actual_labels, predicted_labels, location=self.expt_folder, title='Confusion matrix, ' \
																			  'without normalization (Train)')
		
		return (minibatch_accuracy, minibatch_losses)
	
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
					 len(self.train_loader),
					 loss.data[0], accuracy))
		
		# clean GPU
		del images, labels, outputs
		
		self.writer.add_scalar('minibatch_loss', np.mean(loss.data[0]), self.batchstep)
		self.batchstep += 1
		
		return accuracy, loss.data[0], pred_labels.data.cpu().numpy()
	
	def validate(self):
		correct, actual_labels, predicted_labels = ([] for i in range(3))
		
		pb = tqdm(total=len(self.valid_loader))
		
		for i, (images, labels) in enumerate(self.valid_loader):
			img = Variable(images, volatile=True).cuda()
			outputs = self.model(img)
			_, predicted = torch.max(outputs.data, 1)
			labels = labels.view(-1, )
			correct.append(((predicted.cpu() == labels).float().mean()))
			pb.update(1)
			
			actual_labels.extend(labels.numpy())
			predicted_labels.extend(predicted.cpu().numpy())
			
			loss = self.criterion(outputs, Variable(labels).cuda())
			self.valid_losses.append(loss.data[0])
			
			del img
			
		pb.close()
		
		print('Validation Accuracy : %0.6f' % np.mean(correct))
		
		self.valid_accuracy.append(np.mean(correct))
		
		# Plot loss and accuracy
		self.writer.add_scalar('validation_accuracy', np.mean(correct), self.curr_epoch)
		self.writer.add_scalar('validation_loss', np.mean(self.valid_losses), self.curr_epoch)
		
		# Plot confusion matrices
		plot_confusion_matrix(actual_labels, predicted_labels, location=self.expt_folder, title='Confusion matrix, ' \
																			  'without normalization (Valid)')
	
	def test(self, test_loader):
		correct, actual_labels, predicted_labels, test_losses = ([] for i in range(4))
		
		pb = tqdm(total=len(self.valid_loader))
		
		for i, (images, labels) in enumerate(test_loader):
			img = Variable(images, volatile=True).cuda()
			outputs = self.model(img)
			_, predicted = torch.max(outputs.data, 1)
			labels = labels.view(-1, )
			correct.append(((predicted.cpu() == labels).float().mean()))
			pb.update(1)
			
			actual_labels.extend(labels.numpy())
			predicted_labels.extend(predicted.cpu().numpy())
			
			loss = self.criterion(outputs, Variable(labels).cuda())
			test_losses.append(loss.data[0])
			
			del img
		
		pb.close()
		
		print('Test Accuracy : %0.6f' % np.mean(correct))
		print('Test Losses : %0.6f' % np.mean(test_losses))
		
		# Plot confusion matrices
		plot_confusion_matrix(actual_labels, predicted_labels, location=self.expt_folder, title='Confusion matrix, ' \
																			  'without normalization (Test)')
		