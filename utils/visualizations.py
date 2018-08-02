'''
Collection of visualization functions
'''
import matplotlib
from numpy.core.multiarray import ndarray

matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import itertools
from sklearn.metrics import confusion_matrix
import nibabel as nib
import numpy as np
from scipy.misc import imsave
from save import savePickle
import torch
from configurations.modelConfig import name_classes
from sklearn.metrics import auc
from sklearn import decomposition
from sklearn.manifold import TSNE

def visualizeSlices(mri, mri_flag, location, file_name):
	'''
	accepts MRI file/3D numpy array corresponding to an MRI

	Args :
		mri_flag = 1 for MRI file
		mri_flag = 0 for array
	
		mri = file name as string if mri_flag =1
		mri = numpy array if mri_flag = 0
	
		file_name = preferred file name to save the visualized slices

	return:
	saves 2D visualization of MRI slices
	'''
	if mri_flag:
		nib_img = nib.load(mri)
		img = nib_img.get_data()
	
	else:
		img = mri
	
	if img.ndim == 3:
		img = np.moveaxis(img, 0, 2)
	elif img.ndim == 4:
		# img = np.moveaxis(img, 1, 3)
		img = np.squeeze(img)
	
	depth, height, width = img.shape
	viz = np.hstack((img[d, ::]).reshape(-1, width) for d in range(20, depth - 10, 10))
	print(viz.shape)
	imsave(os.path.join(location, file_name) + '.png', viz)
	

def plot_confusion_matrix(cm,
						  location,
						  classes=name_classes, #np.asarray(['NL', 'Diseased']),
						  title='Confusion matrix',
						  cmap=plt.cm.Blues):
	"""
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
	# normalize
	cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	print("Normalized confusion matrix")
	print(cmn)
	
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
	savePickle(location, title+'.pkl', cm)
	savePickle(location, title+'(normalized)'+'.pkl', cmn)


def plot_embedding(embedding, labels_actual, labels_predicted, mode, location, title):
	plt.clf()
	colors = ['green', 'red']
	
	if mode == 'pca':
		# PCA
		pca = decomposition.PCA(n_components=2)
		pca.fit(embedding)
		x_embedded = pca.transform(embedding)
		'''
		axes = plt.gca()
		axes.set_xlim([-50, 50])
		axes.set_ylim([-40, 20])
		'''
	
	elif mode == 'tsne':
		# tSNE
		tsne = TSNE(n_components=2, init='random', random_state=0)
		x_embedded = tsne.fit_transform(embedding)
	else:
		print('wrong mode')
		pass
	
	plt.subplot(1, 2, 1)
	plt.title('actuals labels')
	
	plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=labels_actual, cmap=matplotlib.colors.ListedColormap(colors),
				alpha=0.5, edgecolors='none')
	
	plt.subplot(1, 2, 2)
	plt.title('predicted labels')
	
	plt.scatter(x_embedded[:, 0], x_embedded[:, 1], c=labels_predicted, cmap=matplotlib.colors.ListedColormap(colors),
				alpha=0.5, edgecolors='none')
	
	plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
	cb = plt.colorbar()
	loc = np.arange(0, max(labels_actual), max(labels_actual) / float(len(colors)))
	cb.set_ticks(loc)
	cb.set_ticklabels(['NL', 'Diseased'])
	cb.set_label('Disease Label')
	plt.savefig(os.path.join(location, title + '_' + mode + '.png'), additional_artists=cb, bbox_inches="tight")
	
def plotROC(cm, location, title):
	fpr = cm[0,1] * 1. / np.sum(cm[0,:])
	tpr = cm[1,1] * 1. /np.sum(cm[1,:])
	
	auc_ = auc(fpr, tpr)
	plt.plot(fpr, tpr, color='b',
			 label=r'ROC (AUC = %0.2f)' % (auc_),
			 lw=2, alpha=.8)
	
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	#plt.legend(loc="lower right")
	plt.savefig(os.path.join(location, title))


def plot_accuracy(train_acc, test_acc, location, title='Accuracy'):
	"""
    This function plots accuracy over epochs.
    """
	
	plt.clf()
	
	plt.plot(train_acc, label='train accuracy', color='g')
	plt.plot(test_acc, label='test accuracy', color='r')
	
	plt.tight_layout()
	plt.ylabel('Accuracy')
	plt.xlabel('Epochs')
	plt.title(title)
	plt.legend()
	
	plt.savefig(os.path.join(location, title))
	
def visualizeFilters(model_weights):
	plt.figure(figsize=(10, 10))
	for idx, filt in enumerate(model_weights['conv1.weight']):
		print(filt[0,:,:,:])
		#print(filt[0, :, :])
		plt.subplot(3, 3, idx + 1)
		plt.imshow(filt[0, :, :], cmap="gray")
		plt.axis('off')
		
		plt.savefig('filters.png')


def run_test():
	model_weights = torch.load('/home/ml/sbasu11/Documents/ADNI Project/ADNI_data/CNN/model.pkl')
	#model_weights = torch.load('/home/ml/sbasu11/Documents/ADNI Project/ADNI_data/CNN/model.pkl',
	# map_location=lambda storage, loc: storage)
	#print(model_weights.keys())
	
	#print(model_weights['conv1.weight'])
	visualizeFilters(model_weights)
	
#run_test()