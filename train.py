''''
train model
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from configurations.modelConfig import params, num_classes
from tqdm import tqdm
import numpy as np
from utils.visualizations import plot_confusion_matrix, plot_embedding
from utils.metrics import updateConfusionMatrix, calculateF1Score
from tensorboardX import SummaryWriter
from utils.save import saveModelandMetrics
from torch.optim.lr_scheduler import MultiStepLR
from utils.visualizations import plotROC
import pickle
from configurations.paths import paths, file_names
import os

train_indices = pickle.load(open(os.path.join(paths['data']['Input_to_Training_Model'],
											  file_names['data']['Train_set_indices']), 'r'))

valid_indices = pickle.load(open(os.path.join(paths['data']['Input_to_Training_Model'],
											  file_names['data']['Valid_set_indices']), 'r'))

test_indices = pickle.load(open(os.path.join(paths['data']['Input_to_Training_Model'],
											 file_names['data']['Test_set_indices']), 'r'))

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
		self.batchstep = 0
		
		self.expt_folder = expt_folder
		self.writer = SummaryWriter(log_dir=expt_folder)
		
		self.train_losses, self.train_f1_Score, self.valid_losses, self.valid_f1_Score, self.train_accuracy, \
		self.valid_accuracy = ([] for i in range(6))
	
	def train(self):
		scheduler = MultiStepLR(self.optimizer, milestones=params['train']['lr_schedule'], gamma=0.1)	# [40, 60] earlier
		
		for _ in range(params['train']['num_epochs']):
			print('Training...\nEpoch : '+str(self.curr_epoch))
			scheduler.step()
			
			# Train Model
			accuracy, loss, f1_score = self.trainEpoch()
			
			self.train_losses.append(loss)
			self.train_accuracy.append(accuracy)
			self.train_f1_Score.append(f1_score)
			
			# Validate Model
			print ('Validation...')
			self.validate()
			
			# Save model
			saveModelandMetrics(self)
			
			# TODO : Stop learning if model doesn't improve for 10 epochs --> Save best model
			
			self.curr_epoch += 1
	
	def trainEpoch(self):
		self.model.train(True)
		
		pbt = tqdm(total=len(self.train_loader))
		
		cm = np.zeros((num_classes, num_classes), int)
		
		minibatch_losses = 0
		minibatch_accuracy = 0
		
		for batch_idx, (images, labels) in enumerate(self.train_loader):
			
			torch.cuda.empty_cache()
			accuracy, loss, conf_mat = self.trainBatch(batch_idx, images, labels)
			
			minibatch_losses += loss
			minibatch_accuracy += accuracy
			cm += conf_mat
			
			pbt.update(1)
		
		pbt.close()
		
		minibatch_losses /= len(train_indices)
		minibatch_accuracy /= len(train_indices)
			
		# Plot losses
		self.writer.add_scalar('train_loss', minibatch_losses , self.curr_epoch)
		self.writer.add_scalar('train_accuracy', minibatch_accuracy, self.curr_epoch)
		
		# Plot confusion matrices
		plot_confusion_matrix(cm, location=self.expt_folder, title='CNN on Train Set')
		
		# F1 Score
		f1_score = calculateF1Score(cm)
		self.writer.add_scalar('train_f1_score', f1_score, self.curr_epoch)
		print('F1 Score : ', f1_score)
		
		# plot ROC curve
		#plotROC(cm, location=self.expt_folder, title='ROC Curve(Train)')
		
		return (minibatch_accuracy, minibatch_losses, f1_score)
	
	def trainBatch(self, batch_idx, images, labels):
		images = Variable(images).cuda()
		labels = Variable(labels).cuda()
		labels = labels.view(-1, )
		
		# Forward + Backward + Optimize
		self.optimizer.zero_grad()
		outputs, _ = self.model(images)
		
		loss = self.criterion(outputs, labels)
		loss.backward()
		
		self.optimizer.step()
		
		# Compute accuracy
		_, pred_labels = torch.max(outputs, 1)
		accuracy = (labels == pred_labels).float().sum()
		
		# Print metrics
		if batch_idx % 100 == 0:
			print('Epoch [%d/%d], Batch [%d/%d] Loss: %.4f Accuracy: %0.2f'
				  % (self.curr_epoch, params['train']['num_epochs'], batch_idx,
					 len(self.train_loader),
					 loss.data[0], accuracy))
		cm = updateConfusionMatrix(labels.data.cpu().numpy(), pred_labels.data.cpu().numpy())
		
		# clean GPU
		del images, labels, outputs, _, pred_labels
		
		self.writer.add_scalar('minibatch_loss', np.mean(loss.data[0]), self.batchstep)
		self.batchstep += 1
		
		return accuracy, loss.data[0], cm
	
	def validate(self):
		self.model.eval()
		correct = 0
		cm = np.zeros((num_classes, num_classes), int)
		
		pb = tqdm(total=len(self.valid_loader))
		
		for i, (images, labels) in enumerate(self.valid_loader):
			img = Variable(images, volatile=True).cuda()
			outputs, _ = self.model(img)
			_, predicted = torch.max(outputs.data, 1)
			labels = labels.view(-1, )
			correct += ((predicted.cpu() == labels).float().sum())
			
			cm += updateConfusionMatrix(labels.numpy(), predicted.cpu().numpy())
			
			loss = self.criterion(outputs, Variable(labels).cuda())
			self.valid_losses.append(loss.data[0])
			
			del img
			pb.update(1)
			
		pb.close()
		
		correct /= len(valid_indices)
		
		print('Validation Accuracy : %0.6f' % correct)
		
		self.valid_accuracy.append(correct)
		
		# Plot loss and accuracy
		self.writer.add_scalar('validation_accuracy', correct, self.curr_epoch)
		self.writer.add_scalar('validation_loss', np.mean(self.valid_losses), self.curr_epoch)
		
		# Plot confusion matrices
		plot_confusion_matrix(cm, location=self.expt_folder, title='CNN on Validation Set')
		
		# F1 Score
		f1_score = calculateF1Score(cm)
		self.writer.add_scalar('valid_f1_score', f1_score, self.curr_epoch)
		print('F1 Score : ',f1_score)
		self.valid_f1_Score.append(f1_score)
		
		# plot ROC curve
		#plotROC(cm, location=self.expt_folder, title='ROC Curve(Valid)')
	
	def test(self, test_loader):
		self.model.eval()
		print ('Test...')
		
		correct =0
		test_losses = 0
		cm = np.zeros((num_classes, num_classes), int)
		embedding = []
		pred_labels = []
		act_labels = []
		
		pb = tqdm(total=len(self.valid_loader))
		
		for i, (images, labels) in enumerate(test_loader):
			img = Variable(images, volatile=True).cuda()
			outputs, features = self.model(img)
			#outputs = outputs.exp()
			#print(outputs.size(), features.size())
			_, predicted = torch.max(outputs.data, 1)
			labels = labels.view(-1, )
			correct += ((predicted.cpu() == labels).float().sum())
			
			cm += updateConfusionMatrix(labels.numpy(), predicted.cpu().numpy())
			
			loss = self.criterion(outputs, Variable(labels).cuda())
			test_losses += loss.data[0]
			
			del img
			pb.update(1)
		
			embedding.extend(np.array(features.data.cpu().numpy()))
			pred_labels.extend(np.array(predicted.cpu().numpy()))
			act_labels.extend(np.array(labels.numpy()))
		
		pb.close()
		
		correct /= len(test_indices)
		test_losses /= len(test_indices)
		
		print('Test Accuracy : %0.6f' % correct)
		print('Test Losses : %0.6f' % test_losses)
		
		# Plot confusion matrices
		plot_confusion_matrix(cm, location=self.expt_folder, title='CNN on Test Set')
		
		# F1 Score
		print('F1 Score : ', calculateF1Score(cm))
		
		# plot PCA or tSNE
		embedding = np.array(embedding)
		pred_labels = np.array(pred_labels)
		act_labels = np.array(act_labels)
		
		plot_embedding(embedding, act_labels, pred_labels, mode='tsne', location=self.expt_folder)
		plot_embedding(embedding, act_labels, pred_labels, mode='pca', location = self.expt_folder)
		
		# plot ROC curve
		#plotROC(cm, location=self.expt_folder, title='ROC Curve(Test)')
		