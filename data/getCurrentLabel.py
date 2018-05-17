'''
This code finds labels for each of the 3D MRI and stores the MRI filenames and corresponding labels in a csv file
'''

import glob
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
from paths import paths, file_names

path = paths['data']['ADNI_study_data_original_labels']
file_name = file_names['data']['ADNI_study_data_original_labels']

img_loc = paths['data']['Raw_MRI_location']
all_imgs = glob.glob(img_loc)

columns = ['FileName', 'Phase', 'RID', 'VISCODE', 'VISCODE2', 'EXAMDATE', 'DXCURREN', 'DXCHANGE', 'DIAGNOSIS_LABEL']
df_new = pd.DataFrame(columns=columns)

files = []

for indx in range(len(all_imgs)):
	fn = all_imgs[indx]
	fn = re.split(r'[/.]', fn)
	files.append(fn[-2])

# print files

file = os.path.join(path, file_name)
df = pd.read_csv(file, engine='python')

for idx, row in df.iterrows():
	if row['Phase'] != 'ADNI3':
		rid = row['RID']
		examdate = row['EXAMDATE']
		
		examdate = datetime.strptime(examdate, '%Y-%m-%d')
		
		rid = format(rid, '04d')
		selected_files = [s for s in files if '_' + str(rid) + '_' in s]
		if selected_files:
			dates_str = [re.split(r'[_]', fn)[-2] for fn in selected_files]
			file_dates = [datetime.strptime(d, '%Y%m%d') for d in dates_str]
			diff = np.array([abs(d - examdate).days for d in file_dates])
			
			'''
			print idx, rid, examdate
			print 'diff :', np.argmin(diff), diff[np.argmin(diff)]
			print selected_files
			#print matched_date
			#pdb.set_trace()
			'''
			
			if diff[np.argmin(diff)] <= 180:
				print selected_files[np.argmin(diff)]
				
				# Create new label field
				if (row['Phase'] == 'ADNI1'):
					if (row['DXCURREN'] == 1):
						DIAGNOSIS_LABEL = 'NL'
					elif (row['DXCURREN'] == 2):
						DIAGNOSIS_LABEL = 'MCI'
					else:
						DIAGNOSIS_LABEL = 'AD'
				else:
					if (row['DXCHANGE'] == 1 or 7 or 9):
						DIAGNOSIS_LABEL = 'NL'
					elif (row['DXCHANGE'] == 2 or 4 or 8):
						DIAGNOSIS_LABEL = 'MCI'
					else:
						DIAGNOSIS_LABEL = 'AD'
				
				df_temp = pd.DataFrame({
					'FileName' 		: selected_files[np.argmin(diff)],
					'Phase' 			: row['Phase'],
					'RID'				: [row['RID']],
					'VISCODE'			: [row['VISCODE']],
					'VISCODE2'			: [row['VISCODE2']],
					'EXAMDATE' 			: [row['EXAMDATE']],
					'DXCURREN'			: [row['DXCURREN']],
					'DXCHANGE'			: [row['DXCHANGE']],'DIAGNOSIS_LABEL' 	: [DIAGNOSIS_LABEL]
				})
				df_new = pd.concat([df_new, df_temp], ignore_index =True)
				
				# drop the filename from list
				files.remove(selected_files[np.argmin(diff)])

# Store data in new csv file
df_new = df_new[[columns[0], columns[1], columns[2], columns[3], columns[4],
				 columns[5], columns[6], columns[7], columns[8]]]

df_new.to_csv(os.path.join(paths['data']['Input_to_Training_Model'],
						   file_names['data']['MRI_to_curr_label_mapping']))

