import numpy as np

def updateConfusionMatrix(actual_labels, predicted_labels):
	"""
	updates confusion matrix after every minibatch
	:param actual_labels:
	:param predicted_labels:
	:return:
	cnfusion matrix
	"""
	n_class = 3
	cm = np.zeros((n_class, n_class))
	
	for (al, pl) in zip(actual_labels, predicted_labels):
		#print(al, pl)
		cm[al, pl] += 1
	#print(cm)
	return cm

#updateConfusionMatrix([0,2,0],[0,0,1])