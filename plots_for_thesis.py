import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pickle
import itertools
import os
from configurations.modelConfig import  name_classes, num_classes
import numpy as np
from sklearn.metrics import roc_curve, auc


def plot_confusion_matrix(cm,
						  location,
						  classes=name_classes,  # np.asarray(['NL', 'Diseased']),
						  title='Confusion matrix',
						  cmap=plt.cm.Blues):
	"""
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
	
	plt.clf()
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)
	
	fmt = 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else
		"black")
	
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	
	plt.savefig(os.path.join(location, title), bbox_inches="tight")
	
	
def plot1():
	# plot saved confusion matrix
	
	'''cm_pkl = '/home/ml/sbasu11/Documents/ADNI Project/ADNI_data/CNN/Outputs/20180713-173351/Confusion matrix, ' \
			 '(Train).pkl'
	'''
	location = '/home/ml/sbasu11/Dropbox/Thesis Results/Confusion Matrix'	#'/home/ml/sbasu11/Documents/ADNI Project/ADNI_data/CNN/Outputs/20180719-210645/'
	title = 'AutoEncoder on Train Set'
	
	#cm = pickle.load(open(cm_pkl, 'r'))
	cm = np.array([[1620, 307],[224,1106]])
	plot_confusion_matrix(cm, location=location,
						  title=title)

def plot_accuracy(cnn, ae, vae, location, title):
	plt.clf()
	plt.plot(cnn, label='CNN', color='g')
	plt.plot(ae, label='AE', color='r')
	plt.plot(vae, label='VAE', color='b')
	
	plt.tight_layout()
	
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.title(title)
	plt.legend(loc="lower right")
	plt.savefig(os.path.join(location, title), bbox_inches="tight")
	
def plot2():
	# plot accuracy comparison of models
	location = '/home/ml/sbasu11/Dropbox/Thesis Results/Accuracy/'
	
	cnn_train = pickle.load(open('/home/ml/sbasu11/Dropbox/Thesis Results/Accuracy/CNN_train_accuracy.pkl', 'r'))
	cnn_train = [d.data.cpu().tolist()[0] for d in cnn_train]

	ae_train = pickle.load(open('/home/ml/sbasu11/Dropbox/Thesis Results/Accuracy/AE_train_accuracy.pkl', 'r'))
	ae_train = [d.data.cpu().tolist()[0] for d in ae_train]
	
	vae_train = pickle.load(open('/home/ml/sbasu11/Dropbox/Thesis Results/Accuracy/VAE_train_accuracy.pkl', 'r'))
	vae_train = [d.data.cpu().tolist()[0] for d in vae_train]
	
	
	plot_accuracy(cnn_train, ae_train, vae_train, location, title='Accuracy on Train Set')
	
	cnn_valid = pickle.load(open('/home/ml/sbasu11/Dropbox/Thesis Results/Accuracy/CNN_valid_accuracy.pkl', 'r'))
	ae_valid = pickle.load(open('/home/ml/sbasu11/Dropbox/Thesis Results/Accuracy/AE_valid_accuracy.pkl', 'r'))
	vae_valid = pickle.load(open('/home/ml/sbasu11/Dropbox/Thesis Results/Accuracy/VAE_valid_accuracy.pkl', 'r'))
	
	plot_accuracy(cnn_valid, ae_valid, vae_valid, location, title='Accuracy on Validation Set')


#plot1()
plot2()
	
	
