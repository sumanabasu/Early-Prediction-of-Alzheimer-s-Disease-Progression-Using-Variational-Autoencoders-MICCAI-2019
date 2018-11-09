import h5py
import numpy as np
import nibabel
import glob
import pandas as pd
from paths import paths, file_names
import os

images_folder = paths['data']['Raw_MRI_location']
pointers_file = os.path.join(paths['data']['Input_to_Training_Model'], file_names['data'][
	'MRI_to_next_label_mapping'])
#current_label_file = os.path.join(paths['data']['Input_to_Training_Model'], file_names['data'][
# 'MRI_to_curr_label_mapping'])

files = glob.glob(images_folder)
#next_labels_df = pd.read_csv(next_label_file)
pointers_df = pd.read_csv(pointers_file)

data_shape4 = (pointers_df.shape[0], 1, 189, 233, 197)
#print data_shape, next_labels_df.columns.values
hdf5_path = os.path.join(paths['data']['Input_to_Training_Model'], file_names['data']['hdf5_file'])

hdf5_file = h5py.File(hdf5_path, mode = 'w')

dt = h5py.special_dtype(vlen=bytes)

hdf5_file.create_dataset("RID", (pointers_df.shape[0],), np.int16)
hdf5_file.create_dataset("FileName", (pointers_df.shape[0],), dtype=dt)
hdf5_file.create_dataset("NextFileName", (pointers_df.shape[0],), dtype=dt)
hdf5_file.create_dataset("CurrImage", data_shape4, np.float32)
hdf5_file.create_dataset("NextImage", data_shape4, np.float32)
hdf5_file.create_dataset("CurrLabel", (pointers_df.shape[0],), dtype = dt)
hdf5_file.create_dataset("NextLabel", (pointers_df.shape[0],), dtype = dt)


for idx, row in pointers_df.iterrows():
	selected_idx = [indx for indx,f in enumerate(files) if row['FileName'] in files[indx]]
	#print(idx, row['FileName'], files[selected_idx[0]])
	#label = curr_labels_df.loc[curr_labels_df['FileName'] == row['FileName'], 'DIAGNOSIS_LABEL'].iloc[0]
	#print(label)
	
	img_nib = nibabel.load(files[selected_idx[0]])
	img = img_nib.get_data()
	
	#get next MRI
	next_mri_idx = [indx for indx, f in enumerate(files) if row['Next_MRI_FileName'] in files[indx]]
	img_nib = nibabel.load(files[next_mri_idx[0]])
	img_next = img_nib.get_data()
	
	#print(idx, row['FileName'], row['Next_MRI_FileName'], files[selected_idx[0]], files[next_mri_idx[0]])

	hdf5_file["RID"][idx] = row['RID']
	hdf5_file["FileName"][idx] = row['FileName']
	hdf5_file["NextFileName"][idx] = row['Next_MRI_FileName']
	hdf5_file["CurrImage"][idx] = img[np.newaxis,:,:,:]
	hdf5_file["NextImage"][idx] = img[np.newaxis, :, :, :]
	hdf5_file["CurrLabel"][idx]	= row['Curr_LABEL']
	hdf5_file["NextLabel"][idx] = row['Next_LABEL']


	print(idx, hdf5_file["RID"][idx], hdf5_file["FileName"][idx], hdf5_file["CurrLabel"][idx], hdf5_file[
		"NextFileName"][idx], hdf5_file["NextLabel"][idx], hdf5_file["CurrImage"][idx].shape, hdf5_file["NextImage"][idx].shape)

hdf5_file.close()