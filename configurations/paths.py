'''
contains paths used in the project
'''
paths = {
	'data' : {
		'Input_to_Training_Model' : '/home/ml/sbasu11/Documents/ADNI Project/ADNI_data/CNN/Inputs/',
		'ADNI_study_data_oroginal_labels'	: '/home/ml/sbasu11/Documents/ADNI Project/ADNI_data/Data Info/',
		'Raw_MRI_location'	: '/home/ml/sbasu11/Documents/ADNI Project/ADNI_data/ADNI_mnc/*.mnc',
		'hadf5_path'	:	'/home/NOBACKUP/sbasu11/'
	}
}

file_names = {
	'data'	:	{
		'ADNI_study_data_oroginal_labels'	:	'DXSUM_PDXCONV_ADNIALL.csv',
		'MRI_to_next_label_mapping'	:	'3dMRItoNextLabel.csv',
		'MRI_to_curr_label_mapping'		:	'labels.csv',
		'RIDtoMRI'	:	'RIDtoMRIdict.pkl',
		'hdf5_file'	: '3DMRItoNextLabel.hdf5',
		'Train_set_indices'	: 'train_set_indices.pkl',
		'Valid_set_indices'	: 'valid_set_indices.pkl',
		'Test_set_indices'	: 'test_set_indices.pkl'
	}
}
