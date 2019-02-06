''''
train model
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from configurations.modelConfig import params, num_classes, latent_dim
from tqdm import tqdm
import numpy as np
from utils.visualizations import plot_confusion_matrix, plot_embedding
from utils.metrics import updateConfusionMatrix, calculateF1Score
from tensorboardX import SummaryWriter
from utils.save import saveModelandMetrics
from torch.optim.lr_scheduler import MultiStepLR
from data.splitDataset import getIndicesTrainValidTest
from utils.visualizations import plotROC
import pickle
from configurations.paths import paths, file_names
import os
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
import pdb

train_indices = pickle.load(open(os.path.join(paths['data']['Input_to_Training_Model'],
											  file_names['data']['Train_set_indices']), 'r'))

valid_indices = pickle.load(open(os.path.join(paths['data']['Input_to_Training_Model'],
											  file_names['data']['Valid_set_indices']), 'r'))

test_indices = pickle.load(open(os.path.join(paths['data']['Input_to_Training_Model'],
											 file_names['data']['Test_set_indices']), 'r'))


class Trainer(object):
	def __init__(self, prior, posterior, generator, model, train_loader, valid_loader, expt_folder):
		super(Trainer, self).__init__()
		
		if torch.cuda.is_available():
			self.prior = prior.cuda()
			self.posterior = posterior.cuda()
			self.generator = generator.cuda()
			self.model = model.cuda()
		
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.optimizer = torch.optim.Adam(model.parameters(),
										  lr=params['train']['learning_rate'])
		self.classification_criterion = nn.NLLLoss().cuda()
		#self.crossentropy_loss = nn.CrossEntropyLoss()
		
		self.curr_epoch = 0
		self.batchstep = 0
		
		self.expt_folder = expt_folder
		self.writer = SummaryWriter(log_dir=expt_folder)
		
		self.train_losses_class, self.train_losses_vae, self.train_mse, self.train_kld, \
		self.valid_losses, self.valid_mse, self.valid_kld, \
		self.train_f1_Score, self.valid_f1_Score, \
		self.train_accuracy, self.valid_accuracy = ([] for i in range(11))
		
		self.trainset_size, self.validset_size, self.testset_size = getIndicesTrainValidTest(requireslen=True)
	
	# def klDivergence(self, mu_prior, var_prior, mu_posterior, var_posterior):
	# 	# D_KL(Q(z|X) || P(z|X))
	# 	# P(z|X) is the real distribution, Q(z|X) is the distribution we are trying to approximate P(z|X) with
	# 	# calculate in closed form
	#
	# 	# from torch.distributions.kl import kl_divergence not available in torch 0.3.1
	# 	# pdb.set_trace()
	# 	var_prior = torch.exp(var_prior)
	# 	var_posterior = torch.exp(var_posterior)
	# 	# prior_distribution = MultivariateNormal(mu_prior, torch.diag_embed(var_prior))
	# 	# posterior_distribution = MultivariateNormal(mu_posterior, torch.diag_embed(var_posterior))
	# 	prior_distribution = Normal(mu_prior, torch.sqrt(var_prior))
	# 	posterior_distribution = Normal(mu_posterior, torch.sqrt(var_posterior))
	# 	return kl_divergence(prior_distribution, posterior_distribution)
		# Implemented as in : https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
		# {D_{\text{KL}}({\mathcal {N}}_{0}\parallel {\mathcal {N}}_{1})={\frac {1}{2}}\left(\operatorname {tr} \left(\Sigma _{1}^{-1}\Sigma _{0}\right)+(\mu _{1}-\mu _{0})^{\mathsf {T}}\Sigma _{1}^{-1}(\mu _{1}-\mu _{0})-k+\ln \left({\frac {\det \Sigma _{1}}{\det \Sigma _{0}}}\right)\right).}
		
		#
		# print(var_prior.size(), var_posterior.size())
		# term1 = torch.trace(var_prior/var_posterior)
		# term2 = torch.sum((mu_posterior - mu_prior).pow(2) / var_posterior)
		# term31 = torch.prod(var_posterior)
		# term31 = torch.clamp(term31, min=-100, max = 100) + params['train']['epsilon']
		# term32 = torch.prod(var_prior)  + params['train'][ 'epsilon']
		# print('term31 :', term31, 'term32 :', term32)
		# term3 = torch.log(term31  / term32)
		# term3 = torch.clamp(term3, min=-100, max = 100)
		# # print('term 1 :', term1, 'term 2 :', term2, 'term 3:', term3, 'term 3 part 1:', torch.log(torch.prod(
		# #  	var_posterior) + params['train']['epsilon']), 'term 3 part 2: ', torch.log(torch.prod(var_prior) +
		# # 																		   params['train']['epsilon']),
		# # 	  'latent :', latent_dim)
		# kl = 0.5 * (term1 + term2 - latent_dim + term3)
		# return  kl
		# # return 0.5 * (term1 + term2 - latent_dim + term3) # fails because of term3, which is infinite
		# # return 0.5 * (term1 + term2 - latent_dim)
	
	# self.lambda_ = 1	#hyper-parameter to control regularizer by reconstruction loss
	def loss(self, p_hat_t_plus_1, y_t_plus_1, prior_distribution, posterior_distribution):
		# print('inside loss : ', p_hat_t_plus_1, y_t_plus_1, mu_prior, std_prior, mu_posterior, std_posterior)
		NLL = self.classification_criterion(p_hat_t_plus_1, y_t_plus_1)
		# print('NLL :', NLL)
		KLD = torch.sum(kl_divergence(prior_distribution, posterior_distribution))
		print('KLD :', KLD.item())
		
		return NLL + KLD, NLL, KLD
	
	def train(self):
		scheduler = MultiStepLR(self.optimizer, milestones=params['train']['lr_schedule'], gamma=0.1)  # [40,
		# 60] earlier

		self.train_losses, self.train_f1_Score, self.valid_losses, self.valid_f1_Score, self.train_accuracy, \
		self.valid_accuracy = ([] for i in range(6))
	
	# def train(self):
	# 	scheduler = MultiStepLR(self.optimizer, milestones=params['train']['lr_schedule'], gamma=0.1)	# [40, 60] earlier
		
		for _ in range(params['train']['num_epochs']):
			print('Training...\nEpoch : ' + str(self.curr_epoch))
			scheduler.step()
			
			# Train Model
			accuracy, classification_loss, vae_loss, kld, f1_score = self.trainEpoch()
			
			self.train_losses_class.append(classification_loss)
			self.train_losses_vae.append(vae_loss)
			self.train_kld.append(kld)
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
		
		minibatch_losses_classification = 0
		minibatch_losses = 0
		minibatch_accuracy = 0
		minibatch_kld = 0
		
		for batch_idx, (images_curr, images_next, labels_next) in enumerate(self.train_loader):
			# print('batch id :', batch_idx)
			# print('label :', labels_next)
			torch.cuda.empty_cache()
			accuracy, nll_loss, conf_mat, loss, kld = self.trainBatch(batch_idx, images_curr, images_next, labels_next)
			
			minibatch_losses_classification += nll_loss
			minibatch_losses += loss
			minibatch_kld += kld
			minibatch_accuracy += accuracy
			cm += conf_mat
			
			pbt.update(1)
		
		pbt.close()
		
		minibatch_losses_classification /= self.trainset_size
		minibatch_losses /= self.trainset_size
		minibatch_kld /= self.trainset_size
		minibatch_accuracy /= self.trainset_size
		
		minibatch_losses /= len(train_indices)
		minibatch_accuracy /= len(train_indices)
		
		# Plot losses
		self.writer.add_scalar('train_classification_loss', minibatch_losses_classification, self.curr_epoch)
		self.writer.add_scalar('train_loss', minibatch_losses, self.curr_epoch)
		self.writer.add_scalar('train_KL_divergence', minibatch_kld, self.curr_epoch)
		self.writer.add_scalar('train_accuracy', minibatch_accuracy, self.curr_epoch)
		
		# Plot confusion matrices
		plot_confusion_matrix(cm, location=self.expt_folder, title='probCNN on Train Set')
		
		# F1 Score
		f1_score = calculateF1Score(cm)
		self.writer.add_scalar('train_f1_score', f1_score, self.curr_epoch)
		print('F1 Score : ', f1_score)
		
		# plot ROC curve
		# plotROC(cm, location=self.expt_folder, title='ROC Curve(Train)')
		
		return minibatch_accuracy, minibatch_losses_classification, minibatch_losses, minibatch_kld, f1_score
		#plotROC(cm, location=self.expt_folder, title='ROC Curve(Train)')
		
		#return (minibatch_accuracy, minibatch_losses, f1_score)
	
	def trainBatch(self, batch_idx, images_curr, images_next, labels_next):
		images_curr = Variable(images_curr).cuda()
		images_next = Variable(images_next).cuda()
		labels_next = Variable(labels_next).cuda()
		labels_next = labels_next.contiguous().view(-1, )
		# print('label (inside trainBatch:', labels_next)
		
		# Forward + Backward + Optimize
		self.optimizer.zero_grad()
		prior_distribution = self.prior(images_curr)
		posterior_distribution = self.posterior(images_next, labels_next)
		labels_prob_next_pred = self.model(False, images_curr, images_next, labels_next)

		loss, nll, kld = self.loss(labels_prob_next_pred, labels_next, prior_distribution, posterior_distribution)
		
		self.optimizer.zero_grad()
		# classification_loss.backward(retain_graph=True)
		#import pdb; pdb.set_trace()
		loss.backward()
		
		self.optimizer.step()
		
		# Compute accuracy
		_, pred_labels_next = torch.max(labels_prob_next_pred, 1)
		# print('pred :', pred_labels_next)
		accuracy = (labels_next == pred_labels_next).float().sum()
		
		# Print metrics
		if batch_idx % 100 == 0:
			print('Epoch [%d/%d], Batch [%d/%d] Classification Loss: %.4f KL Divergence: %.4f Accuracy: %0.2f '
				  % (self.curr_epoch, params['train']['num_epochs'],
					 batch_idx, self.trainset_size,
					 loss.item(),
					 kld.item(),
					 accuracy * 1.0 / params['train']['batch_size']))
		
		cm = updateConfusionMatrix(labels_next.data.cpu().numpy(), pred_labels_next.data.cpu().numpy())
		
		# clean GPU
		del images_curr, images_next, labels_next, labels_prob_next_pred, pred_labels_next, prior_distribution, posterior_distribution
		
		self.writer.add_scalar('minibatch_NLL', np.mean(nll.item()), self.batchstep)
		self.writer.add_scalar('minibatch_KLD', np.mean(kld.item()), self.batchstep)
		self.writer.add_scalar('minibatch_loss', np.mean(loss.item()), self.batchstep)
		self.batchstep += 1
		
		return accuracy, nll.item(), cm, loss.item(), kld.item()
	
	def validate(self):
		self.model.eval()
		correct = 0
		mse = 0
		kld = 0
		cm = np.zeros((num_classes, num_classes), int)
		loss = 0
		
		pb = tqdm(total=len(self.valid_loader))
		
		for i, (images, labels) in enumerate(self.valid_loader):
			img = Variable(images, volatile=True).cuda()
			_, _, mu, logvar, x_hat, p_hat = self.model(img)
			_, predicted = torch.max(p_hat.data, 1)
			outputs, _ = self.model(img)
			_, predicted = torch.max(outputs.data, 1)
			labels = labels.contiguous().view(-1, )
			correct += ((predicted.cpu() == labels).float().sum())
			
			cm += updateConfusionMatrix(labels.numpy(), predicted.cpu().numpy())
			
			loss += self.classification_criterion(p_hat, Variable(labels).cuda()).data
			mse += self.reconstruction_loss(x_hat, img).data
			kld += self.klDivergence(mu, logvar).data
			
			del img, x_hat, p_hat, _, predicted, mu, logvar
			pb.update(1)
		
		pb.close()
		
		correct /= self.validset_size
		mse /= self.validset_size
		kld /= self.validset_size
		loss /= self.validset_size
		correct /= len(valid_indices)
		
		print('Validation Accuracy : %0.6f' % correct)
		
		self.valid_accuracy.append(correct)
		self.valid_mse.append(mse)
		self.valid_kld.append(kld)
		self.valid_losses.append(loss)
		
		# Plot loss and accuracy
		self.writer.add_scalar('validation_accuracy', correct, self.curr_epoch)
		self.writer.add_scalar('validation_loss', loss * 1.0 / self.validset_size, self.curr_epoch)
		self.writer.add_scalar('validation_mse', mse, self.curr_epoch)
		self.writer.add_scalar('validation KLD', kld, self.curr_epoch)
		# print('MSE : ', mse)
		# print('KLD : ', kld)
		
		# Plot confusion matrices
		plot_confusion_matrix(cm, location=self.expt_folder, title='VAE on Validation Set')
		plot_confusion_matrix(cm, location=self.expt_folder, title='CNN on Validation Set')
		
		# F1 Score
		f1_score = calculateF1Score(cm)
		self.writer.add_scalar('valid_f1_score', f1_score, self.curr_epoch)
		print('F1 Score : ', calculateF1Score(cm))
		self.valid_f1_Score.append(f1_score)
	
	# plot ROC curve
	# plotROC(cm, location=self.expt_folder, title='ROC Curve(Valid)')
		print('F1 Score : ',f1_score)
		self.valid_f1_Score.append(f1_score)
		
		# plot ROC curve
		#plotROC(cm, location=self.expt_folder, title='ROC Curve(Valid)')
	
	def test(self, test_loader):
		self.model.eval()
		print ('Test...')
		
		correct = 0
		test_losses = 0
		cm = np.zeros((num_classes, num_classes), int)
		embedding = []
		pred_labels = []
		act_labels = []
		class_prob = []
		
		encoder_embedding = []
		classifier_embedding = []
		pred_labels = []
		act_labels = []
		class_prob = []
		
		pb = tqdm(total=len(self.valid_loader))
		
		for i, (images, labels) in enumerate(test_loader):
			img = Variable(images, volatile=True).cuda()
			enc_emb, cls_emb, _, _, _, p_hat = self.model(img)
			_, predicted = torch.max(p_hat.data, 1)
			outputs, features = self.model(img)
			#outputs = outputs.exp()
			#print(outputs.size(), features.size())
			_, predicted = torch.max(outputs.data, 1)
			labels = labels.contiguous().view(-1, )
			correct += ((predicted.cpu() == labels).float().sum())
			
			cm += updateConfusionMatrix(labels.numpy(), predicted.cpu().numpy())
			
			loss = self.classification_criterion(p_hat, Variable(labels).cuda())
			test_losses += loss.data[0]
			
			del img
			pb.update(1)
			p_hat = torch.exp(p_hat)
			
			encoder_embedding.extend(np.array(enc_emb.data.cpu().numpy()))
			classifier_embedding.extend(np.array(cls_emb.data.cpu().numpy()))
			pred_labels.extend(np.array(predicted.cpu().numpy()))
			act_labels.extend(np.array(labels.numpy()))
			class_prob.extend(np.array(p_hat.data.cpu().numpy()))
		
		pb.close()
		
		correct /= self.testset_size
		test_losses /= self.testset_size
		outputs = torch.exp(outputs)
		
		embedding.extend(np.array(features.data.cpu().numpy()))
		pred_labels.extend(np.array(predicted.cpu().numpy()))
		act_labels.extend(np.array(labels.numpy()))
		class_prob.extend(np.array(outputs.data.cpu().numpy()))
		
		pb.close()
		
		correct /= len(test_indices)
		test_losses /= len(test_indices)
		
		print('Test Accuracy : %0.6f' % correct)
		print('Test Losses : %0.6f' % test_losses)
		
		# Plot confusion matrices
		plot_confusion_matrix(cm, location=self.expt_folder, title='VAE on Test Set')
		plot_confusion_matrix(cm, location=self.expt_folder, title='CNN on Test Set')
		
		# F1 Score
		print('F1 Score : ', calculateF1Score(cm))
		
		# plot PCA or tSNE
		encoder_embedding = np.array(encoder_embedding)
		classifier_embedding = np.array(classifier_embedding)
		embedding = np.array(embedding)
		pred_labels = np.array(pred_labels)
		act_labels = np.array(act_labels)
		class_prob = np.array(class_prob)
		plot_embedding(encoder_embedding, act_labels, pred_labels, mode='tsne', location=self.expt_folder,
					   title='encoder_embedding_test')
		plot_embedding(encoder_embedding, act_labels, pred_labels, mode='pca', location=self.expt_folder,
					   title='encoder_embedding_test')
		
		plot_embedding(classifier_embedding, act_labels, pred_labels, mode='tsne', location=self.expt_folder,
					   title='classifier_embedding_test')
		plot_embedding(classifier_embedding, act_labels, pred_labels, mode='pca', location=self.expt_folder,
					   title='classifier_embedding_test')
		
		# plot ROC curve
		plotROC(act_labels, class_prob, location=self.expt_folder, title='ROC (VAE on Test Set)')
	
	'''
	def test(self, test_loader):
		self.model.eval()
		print ('Test...')
		
		encoder_embedding = []
		classifier_embedding = []
		pred_labels = []
		act_labels = []
		
		pb = tqdm(total=len(self.valid_loader))
		
		for i, (images, labels) in enumerate(test_loader):
			for itr in range(20):
				print('iteration:',itr)
				print('label:',labels.cpu())
				
				img = Variable(images, volatile=True).cuda()
				enc_emb, cls_emb, _, _, _, p_hat = self.model(img)
				_, predicted = torch.max(p_hat.data, 1)
				labels = labels.view(-1, )
				
				del img
				pb.update(1)
				
				encoder_embedding.extend(np.array(enc_emb.data.cpu().numpy()))
				classifier_embedding.extend(np.array(cls_emb.data.cpu().numpy()))
				pred_labels.extend(np.array(predicted.cpu().numpy()))
				act_labels.extend(np.array(labels.numpy()))
				
			if i == 0:
				break
		
		pb.close()
		
		
		# plot PCA or tSNE
		encoder_embedding = np.array(encoder_embedding)
		classifier_embedding = np.array(classifier_embedding)
		pred_labels = np.array(pred_labels)
		act_labels = np.array(act_labels)
		
		plot_embedding(encoder_embedding, act_labels, pred_labels, mode='tsne', location=self.expt_folder,
					   title='encoder_embedding_test')
		plot_embedding(encoder_embedding, act_labels, pred_labels, mode='pca', location=self.expt_folder,
					   title='encoder_embedding_test')
		
		plot_embedding(classifier_embedding, act_labels, pred_labels, mode='tsne', location=self.expt_folder,
					   title='classifier_embedding_test')
		plot_embedding(classifier_embedding, act_labels, pred_labels, mode='pca', location=self.expt_folder,
					   title='classifier_embedding_test')
	
		plot_embedding(embedding, act_labels, pred_labels, mode='tsne', location=self.expt_folder)
		plot_embedding(embedding, act_labels, pred_labels, mode='pca', location = self.expt_folder)
		
		# plot ROC curve
		#plotROC(cm, location=self.expt_folder, title='ROC Curve(Test)')
		plotROC(act_labels, class_prob, location=self.expt_folder, title='ROC (CNN on Test Set)')
	'''
