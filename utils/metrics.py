import numpy as np
from configurations.modelConfig import num_classes

def updateConfusionMatrix(actual_labels, predicted_labels):
	"""
	updates confusion matrix after every minibatch
	:param actual_labels:
	:param predicted_labels:
	:return:
	cnfusion matrix
	"""
	n_class = num_classes #3
	cm = np.zeros((n_class, n_class), int)
	for (al, pl) in zip(actual_labels, predicted_labels):
		#print(al, pl)
		cm[al, pl] += 1
	#print(cm)
	return cm

#updateConfusionMatrix([0,2,0],[0,0,1])

def calculateF1Score(cm):
	true_negative = cm[0,0]
	false_negative = cm[1,0]
	
	true_positive = cm[1,1]
	false_positive = cm[0,1]
	
	precision = (true_positive * 1.) / (true_positive + false_positive)
	recall = (true_positive * 1.0) / (true_positive + false_negative)
	
	f1_score = 2.0 * (recall * precision) / (recall + precision)
	
	return f1_score
	