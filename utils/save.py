import os
import json
from configurations.paths import paths, file_names
from configurations import modelConfig
import torch
import pickle

def savePickle(location, file, data):
	pkl_file = open(os.path.join(location, file), 'wb')
	pickle.dump(data, pkl_file)
	pkl_file.close()

def saveModelandMetrics(modelObj):
	# Save Prameters
	with open(os.path.join(modelObj.expt_folder, file_names['output']['parameters']), 'w') as fp:
		json.dump([modelConfig.layer_config, modelConfig.params, modelConfig.data_aug], fp)
	
	# Save the latest Trained Models
	torch.save(modelObj.model.state_dict(), os.path.join(modelObj.expt_folder, 'latest_model.pkl'))
	
	# Save metrics
	'''
	loss 		: 	train, validation loss over epochs
	accuracy 	: 	train, validation accuracy over epochs
	'''
	# TODO : plot loss and accuracy curves
	
	savePickle(location=modelObj.expt_folder, file=file_names['output']['train_loss_classification'],
			   data=modelObj.train_losses_class)
	savePickle(location=modelObj.expt_folder, file=file_names['output']['train_loss_vae'],
			   data=modelObj.train_losses_vae)
	savePickle(location=modelObj.expt_folder, file=file_names['output']['train_loss_mse'],
			   data=modelObj.train_mse)
	savePickle(location=modelObj.expt_folder, file=file_names['output']['train_loss_kld'],
			   data=modelObj.train_kld)
	savePickle(location=modelObj.expt_folder, file=file_names['output']['valid_loss'], data=modelObj.valid_losses)
	
	savePickle(location=modelObj.expt_folder, file=file_names['output']['train_accuracy'], data=modelObj.train_accuracy)
	savePickle(location=modelObj.expt_folder, file=file_names['output']['valid_accuracy'], data=modelObj.valid_accuracy)
	
	savePickle(location=modelObj.expt_folder, file=file_names['output']['train_f1_score'], data=modelObj.train_f1_Score)
	savePickle(location=modelObj.expt_folder, file=file_names['output']['valid_f1_score'], data=modelObj.valid_f1_Score)
	