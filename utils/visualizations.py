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
	
	
'''
def plot_confusion_matrix(actual_labels,
						  predicted_labels,
						  location,
						  classes = np.asarray(['NL', 'MCI', 'AD']),
						  normalize=False,
						  title='Confusion matrix',
						  cmap=plt.cm.Blues):
	"""
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
	
	cm = confusion_matrix(actual_labels, predicted_labels)
	
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix without normalization')
	
	print(cm)
	
	plt.clf()
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)
	
	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else
		"black")
	
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	
	plt.savefig(os.path.join(location, title))
'''

def plot_confusion_matrix(cm,
						  location,
						  classes=np.asarray(['NL', 'Diseased']), #np.asarray(['NL', 'MCI', 'AD']),
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
	
	plt.savefig(os.path.join(location, title))
	savePickle(location, title, cm)
	savePickle(location, title+'(normalized)', cmn)

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