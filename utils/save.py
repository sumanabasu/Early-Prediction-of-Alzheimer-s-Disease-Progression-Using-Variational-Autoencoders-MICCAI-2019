import os
import json
from paths import paths, file_names
import modelConfig
import torch
import pickle
from visualizations import plot_accuracy

def savePickle(location, file, data):
	pkl_file = open(os.path.join(location, file), 'wb')
	pickle.dump(data, pkl_file)
	pkl_file.close()

def saveModelandMetrics(modelObj):
	# Save Prameters
	with open(os.path.join(modelObj.expt_folder, file_names['output']['parameters']), 'w') as fp:
		json.dump([modelConfig.layer_config, modelConfig.params], fp)
	
	# Save the latest Trained Models
	torch.save(modelObj.model.state_dict(), os.path.join(modelObj.expt_folder, 'latest_model.pkl'))
	
	# Save metrics
	'''
	loss 		: 	train, validation loss over epochs
	accuracy 	: 	train, validation accuracy over epochs
	'''
	# TODO : plot loss and accuracy curves
	
	savePickle(location=modelObj.expt_folder, file=file_names['output']['train_loss'], data=modelObj.train_losses)
	savePickle(location=modelObj.expt_folder, file=file_names['output']['valid_loss'], data=modelObj.valid_losses)
	
	savePickle(location=modelObj.expt_folder, file=file_names['output']['train_loss'], data=modelObj.train_accuracy)
	savePickle(location=modelObj.expt_folder, file=file_names['output']['valid_loss'], data=modelObj.valid_accuracy)
	