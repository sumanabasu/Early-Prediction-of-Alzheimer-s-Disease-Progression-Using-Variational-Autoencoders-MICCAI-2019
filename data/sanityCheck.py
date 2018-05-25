'''
check what percentage of voxels fall in the specified range of (0, 200)
'''

import nibabel
import os
import glob
from paths import paths, file_names
import re
import numpy as np

def checkIntensityRange(file):
	img = nibabel.load(file)
	img = img.get_data()
	
	total_voxels = img.shape[0] * img.shape[1] * img.shape[2]
	
	'''
	np.place(img, (img < 0) | (img > 200), 0)
	print(img.shape, img.max(), img.min())
	'''
	
	within_range_voxel_count = ((img >= 0) & (img <= 200)).sum()
	outside_range_voxel_count = ((img < 0) | (img > 200)).sum()
	percnt_outside = (outside_range_voxel_count * 1.0) / total_voxels
	if percnt_outside > 0.1 :
		print(re.split(r'[/.]', file)[-2])
		print('percentage outside range : ', percnt_outside)
		return 1
	return 0
	
def run_tests():
	path = paths['data']['Input_to_Training_Model']
	all_files = glob.glob(os.path.join(path, paths['data']['Raw_MRI_location']))
	count = 0
	for file in all_files:
		print(re.split(r'[/.]', file)[-2])
		count += checkIntensityRange(file)
	print('\n Total Images Outside Stat : ', count)
	print('Total Percentage of Images Outside Stat : ', (count * 1.0/len(all_files)))
		
run_tests()