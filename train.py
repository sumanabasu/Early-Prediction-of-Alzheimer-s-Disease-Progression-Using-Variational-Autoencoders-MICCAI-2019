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
		
		minibatch_losses /= len(self.train_loader)
		minibatch_accuracy /= len(self.train_loader)
			
		# Plot losses
		self.writer.add_scalar('train_loss', minibatch_losses , self.curr_epoch)
		self.writer.add_scalar('train_accuracy', minibatch_accuracy, self.curr_epoch)
		
		# Plot confusion matrices
		plot_confusion_matrix(cm, location=self.expt_folder, title='Confusion matrix, ' \
																			  '(Train)')
		
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
			outputs = self.model(img)
			_, predicted = torch.max(outputs.data, 1)
			labels = labels.view(-1, )
			correct += ((predicted.cpu() == labels).float().mean())
			
			cm += updateConfusionMatrix(labels.numpy(), predicted.cpu().numpy())
			
			loss = self.criterion(outputs, Variable(labels).cuda())
			self.valid_losses.append(loss.data[0])
			
			del img
			pb.update(1)
			
		pb.close()
		
		correct /= len(self.valid_loader)
		
		print('Validation Accuracy : %0.6f' % correct)
		
		self.valid_accuracy.append(correct)
		
		# Plot loss and accuracy
		self.writer.add_scalar('validation_accuracy', correct, self.curr_epoch)
		self.writer.add_scalar('validation_loss', np.mean(self.valid_losses), self.curr_epoch)
		
		# Plot confusion matrices
		plot_confusion_matrix(cm, location=self.expt_folder, title='Confusion matrix, ' \
																			  'without normalization (Valid)')
		
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
		softmax = []
		xx = []
		outt1 = []
		outt2 = []
		outt3 = []
		flatt = []
		fccc1 = []
		fccc2 = []
		
		pb = tqdm(total=len(self.valid_loader))
		
		for i, (images, labels) in enumerate(test_loader):
			img = Variable(images, volatile=True).cuda()
			outputs, x, out1, out2, out3, flat, fcc1, fcc2 = self.model(img)
			#outputs = outputs.exp()
			#print(outputs.size(), features.size())
			#import pdb; pdb.set_trace()
			_, predicted = torch.max(outputs.data, 1)
			labels = labels.view(-1, )
			correct += ((predicted.cpu() == labels).float().mean())
			
			cm += updateConfusionMatrix(labels.numpy(), predicted.cpu().numpy())
			
			loss = self.criterion(outputs, Variable(labels).cuda())
			test_losses += loss.data[0]
			
			del img
			pb.update(1)
			'''
			embedding.extend(np.array(features.data.cpu().numpy()))
			pred_labels.extend(np.array(predicted.cpu().numpy()))
			act_labels.extend(np.array(labels.numpy()))
			'''
			softmax.extend(outputs.data.cpu().numpy())
			xx.extend(x.data.cpu().numpy())
			outt1.extend(out1.data.cpu().numpy())
			outt2.extend(out2.data.cpu().numpy())
			outt3.extend(out3.data.cpu().numpy())
			flatt.extend(flat.data.cpu().numpy())
			fccc1.extend(fcc1.data.cpu().numpy())
			fccc2.extend(fcc2.data.cpu().numpy())
			
			if i == 3:
				import pdb;
				pdb.set_trace()
				np.savez(self.expt_folder+'/softmax.pkl', softmax=softmax, xx=xx, outt1=outt1, outt2=outt2,
						 outt3=outt3, flatt=flatt, fccc1=fccc1, fccc2=fccc2)
				exit(1)
		
		pb.close()
		
		correct /= len(test_loader)
		test_losses /= len(test_loader)
		
		print('Test Accuracy : %0.6f' % correct)
		print('Test Losses : %0.6f' % test_losses)
		
		# Plot confusion matrices
		plot_confusion_matrix(cm, location=self.expt_folder, title='Confusion matrix, ' \
																			  'without normalization (Test)')
		
		# F1 Score
		print('F1 Score : ', calculateF1Score(cm))
		
		# plot PCA or tSNE
		embedding = np.array(embedding)
		pred_labels = np.array(pred_labels)
		act_labels = np.array(act_labels)
		softmax = np.array(softmax)
		print('softmax : ', np.array(softmax))
		from utils.save import savePickle
		savePickle(self.expt_folder, 'softmax.pkl', softmax)
		import pdb;
		pdb.set_trace()
		
		'''
		plot_embedding(embedding, act_labels, pred_labels, mode='tsne', location=self.expt_folder)
		plot_embedding(embedding, act_labels, pred_labels, mode='pca', location = self.expt_folder)
		'''
		
		# plot ROC curve
		#plotROC(cm, location=self.expt_folder, title='ROC Curve(Test)')
		