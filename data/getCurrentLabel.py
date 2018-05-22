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
import re

path = paths['data']['ADNI_study_data_original_labels']
file_name = file_names['data']['ADNI_study_data_original_labels']

img_loc = paths['data']['Raw_MRI_location']
all_imgs = glob.glob(img_loc)

columns = ['FileName', 'RID', 'PTID', 'VISCODE', 'DX_bl', 'MMSE','MMSE_bl' 'DX', 'EXAMDATE']
df_new = pd.DataFrame(columns=columns)

files = []

for indx in range(len(all_imgs)):
	fn = all_imgs[indx]
	fn = re.split(r'[/.]', fn)
	files.append(fn[-2])

#print(files)

file = os.path.join(path, file_name)
df = pd.read_csv(file, engine='python')

# Discard entries with missing labels and backtracking
df = df[df['DX'].notnull() & (df['DX'] != 'MCI to NL') & (df['DX'] != 'Dementia to MCI')]

for idx, row in df.iterrows():
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
		print(idx, rid, examdate)
		print('diff :', np.argmin(diff), diff[np.argmin(diff)])
		print(selected_files)
		#print matched_date
		'''
		
		
		if diff[np.argmin(diff)] <= 180:
			print(selected_files[np.argmin(diff)])
			
			# squeeze baseline labels into 3 classes (NL, MCI, AD) dropping sub-categories of MCI
			if row['DX_bl'] == 'CN':
				baseline = 'NL'
			elif row['DX_bl'] == 'AD':
				baseline = 'AD'
			else:
				baseline = 'MCI'
				
			# squeeze into 3 classes (NL, MCI, AD) while keeping track of intermediate changes
			states = re.split('to ', row['DX'])
			if len(states) > 1:
				previous = states[0]
				current = states[1]
			else:
				previous = states[0]
				current = states[0]
			
			if previous == 'Dementia':
				previous = 'AD'
			if current == 'Dementia':
				current = 'AD'
			
			df_temp = pd.DataFrame({
				'FileName' 		: 	selected_files[np.argmin(diff)],
				'RID'			: 	[row['RID']],
				'PTID'			: 	[row['PTID']],
				'VISCODE'		: 	[row['VISCODE']],
				'DX_bl'			: 	[row['DX_bl']],
				'DX'			:	[row['DX']],
				'baseline'		:	baseline,
				'previous'		:	previous,
				'current'		:	current,
				'EXAMDATE' 		: 	[row['EXAMDATE']],
				'MMSE_bl'		: 	[row['MMSE_bl']],
				'MMSE'			: 	[row['MMSE']]
			})
			df_new = pd.concat([df_new, df_temp], ignore_index =True)
			
			# drop the filename from list
			files.remove(selected_files[np.argmin(diff)])

# Store data in new csv file
df_new = df_new[['FileName', 'RID', 'PTID', 'VISCODE', 'DX_bl', 'DX',\
				 'baseline', 'previous', 'current', 'EXAMDATE', 'MMSE_bl', 'MMSE']]

df_new.to_csv(os.path.join(paths['data']['Input_to_Training_Model'],
						   file_names['data']['MRI_to_curr_label_mapping']))