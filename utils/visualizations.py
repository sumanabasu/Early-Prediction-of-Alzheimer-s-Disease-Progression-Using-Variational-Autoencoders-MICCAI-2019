'''
Collection of visualization functions
'''
import matplotlib.pyplot as plt
import os
import itertools
from paths import paths

def plot_confusion_matrix(cm, classes,
						  normalize=False,
						  title='Confusion matrix',
						  cmap=plt.cm.Blues):
	"""
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
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
	
	plt.savefig(os.path.join(paths['output']['figures'], title))


def plot_accuracy(train_acc, test_acc, title='Accuracy'):
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
	
	plt.savefig(os.path.join(paths['output']['figures'], title))