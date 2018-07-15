'''
returns MRI indices for fixed train, valid and test set
'''

import pandas as pd
import numpy as np
import itertools
import pickle
import os
from paths import paths, file_names

def getTrainValidTestSplit(label_file,
						   rid_file,
						   shuffle,
						   random_seed,
						   train_size=0.8,
						   valid_size=0.1):
	# Read MRI to next label mapping as dataframe
	df = pd.read_csv(label_file)
	rid = rid = cPickle.load(open(rid_file, 'r'))
	
	# Find the set of RID (patients)
	rid_keys = list(set(df['RID']))
	print(len(rid_keys))
	
	if shuffle:
		np.random.seed(random_seed)
		np.random.shuffle(rid_keys)
	
	# Split RID into train:valid:test in 60:20:20 ratio
	train_split_end = int(np.floor(train_size * len(rid_keys)))
	print(0, train_split_end)
	valid_split_end = train_split_end + int(np.floor(valid_size * len(rid_keys)))
	print(train_split_end, valid_split_end)
	
	train_idx = list(itertools.chain.from_iterable(
		[rid[key] for key in rid_keys[:train_split_end]]))
	valid_idx = list(itertools.chain.from_iterable(
		[rid[key] for key in rid_keys[train_split_end:valid_split_end]]))
	test_idx = list(itertools.chain.from_iterable(
		[rid[key] for key in rid_keys[valid_split_end:]]))
	
	print(
		'Total number of patients (train + valid + test) :{}\nPatient count in train set:{}\nPatient count in valid '
		'set:{}\nPatient count in test set:{}\nMRI count in Train set:{}\nMRI count in Valid set:{},\nMRI count in '
		'Test set:{}'.format(
			len(rid_keys), len(rid_keys[:train_split_end]), len(rid_keys[train_split_end:valid_split_end]),
			len(rid_keys[valid_split_end:]), len(train_idx), len(valid_idx), len(test_idx)))
	
	# Save indices of each set in 3dMRItoNextLabel.csv file
	pkl_file = open(os.path.join(paths['data']['Input_to_Training_Model'], file_names['data']['Train_set_indices']),
					'wb')
	cPickle.dump(train_idx, pkl_file)
	pkl_file.close()
	
	pkl_file = open(os.path.join(paths['data']['Input_to_Training_Model'], file_names['data']['Valid_set_indices']),
					'wb')
	cPickle.dump(valid_idx, pkl_file)
	pkl_file.close()
	
	pkl_file = open(os.path.join(paths['data']['Input_to_Training_Model'], file_names['data']['Test_set_indices']),
					'wb')
	cPickle.dump(test_idx, pkl_file)
	pkl_file.close()


'''
# Run only when new splits are to be created
getTrainValidTestSplit(label_file=os.path.join(paths['data']['Input_to_Training_Model'],
											   file_names['data']['MRI_to_next_label_mapping']),
					   rid_file=os.path.join(paths['data']['Input_to_Training_Model'],
											 file_names['data']['RIDtoMRI']),
					   shuffle=True,
					   random_seed=200)
'''


def getIndicesTrainValidTest(requireslen=False):
	train_indices = pickle.load(open(os.path.join(paths['data']['Input_to_Training_Model'],
												  file_names['data']['Train_set_indices']), 'r'))
	
	valid_indices = pickle.load(open(os.path.join(paths['data']['Input_to_Training_Model'],
												  file_names['data']['Valid_set_indices']), 'r'))
	
	test_indices = pickle.load(open(os.path.join(paths['data']['Input_to_Training_Model'],
												 file_names['data']['Test_set_indices']), 'r'))
	if requireslen == True:
		return len(train_indices), len(valid_indices), len(test_indices)
	else:
		return train_indices, valid_indices, test_indices
