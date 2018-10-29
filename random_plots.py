import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pickle as pkl
import torch
import numpy as np
import itertools

f = open('/home/ml/sbasu11/Documents/ADNI Project/ADNI_data/CNN/Outputs/20180627-100252/train_accuracy.pkl', 'r')
list_tensors = pkl.load(f)
train_accuracy = [d.data[0] for d in list_tensors]

f = open('/home/ml/sbasu11/Documents/ADNI Project/ADNI_data/CNN/Outputs/20180627-100252/valid_accuracy.pkl', 'r')
valid_accuracy = pkl.load(f)


plt.plot(train_accuracy, label='train accuracy', color='g')
plt.plot(valid_accuracy, label='validation accuracy', color='r')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()

plt.savefig('accuracy.png')

cm = open('/home/ml/sbasu11/Documents/ADNI Project/ADNI_data/CNN/Outputs/20180627-100252/Confusion matrix, '
		  'without normalization (Test)(normalized).pkl', 'r')

def plot_confusion_matrix(cm,
						  classes= np.asarray(['NL', 'Diseased']),
						  title='Confusion matrix',
						  cmap=plt.cm.Blues):
	"""
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
	
	'''
	# normalize
	cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	print("Normalized confusion matrix")
	print(cmn)
	'''
	
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
	
	plt.savefig(title)