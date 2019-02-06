'''
contains paths used in the project
'''

import time
		
paths = {
	'data' : {
		'hdf5_path'								:	'/home/ml/sbasu11/Documents/ADNI '
														'Project/ADNI_data/LabelDistribution/Inputs/',
		#'hdf5_path'								:	'/home/NOBACKUP/sbasu11/',	#'/home/ml/sbasu11/Documents/ADNI
		# Project/ADNI_data/CNN/Inputs/',
		#'/home/NOBACKUP/sbasu11/',
		'Input_to_Training_Model' 				: 	'/home/ml/sbasu11/Documents/ADNI Project/ADNI_data/LabelDistribution/Inputs/',
		'ADNI_study_data_original_labels'		: 	'/home/ml/sbasu11/Documents/ADNI Project/ADNI_data/Data Info/',
		'Raw_MRI_location'						: 	'/home/ml/sbasu11/Documents/ADNI Project/ADNI_data/ADNI_mnc/*.mnc'
	},
	'output'	:	{
		'base_folder'							:	'/home/ml/sbasu11/Documents/ADNI '
													  'Project/ADNI_data/LabelDistribution/Outputs/'
	}
}

file_names = {
	'data'	:	{
		'hdf5_file'								: 	'MRItoNextMRIandNextLabel.hdf5', #'data.hdf5',
		#'3DMRItoNextLabel.hdf5',
		'ADNI_study_data_oroginal_labels'		:	'DXSUM_PDXCONV_ADNIALL.csv',
		'MRI_to_next_label_mapping'				:	'3dMRItoNextMRIandNextLabel.csv',
		'MRI_to_curr_label_mapping'				:	'labels.csv',
		'RIDtoMRI'								:	'RIDtoMRIdict.pkl',
		'Train_set_indices'						: 	'train_set_indices.pkl',
		'Valid_set_indices'						: 	'valid_set_indices.pkl',
		'Test_set_indices'						: 	'test_set_indices.pkl'
	},
	'output'	:	{
		'parameters'							:	'parameters.json',
		'train_loss_classification'				:	'train_loss_classification.pkl',
		'train_loss_vae'						:	'train_loss_vae.pkl',
		'train_loss_mse'						:	'train_loss_mse.pkl',
		'train_loss_kld'						:	'train_loss_kld.pkl',
		'valid_loss'							:	'valid_loss.pkl',
		'train_accuracy'						:	'train_accuracy.pkl',
		'valid_accuracy'						:	'valid_accuracy.pkl',
		'train_f1_score'						:	'train_f1_score.pkl',
		'valid_f1_score'						:	'valid_f1_score.pkl'
	}
}
