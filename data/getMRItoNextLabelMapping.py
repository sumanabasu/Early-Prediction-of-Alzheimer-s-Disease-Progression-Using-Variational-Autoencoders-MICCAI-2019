'''
this code assigns label of next time stamp to each 3D MRI file

Input :
Raw .mnc or .nii image folder
MRI to current label mapping (labels.csv)

Output :
3dMRItoNextLabel.csv
'''

import os
import glob
import pandas as pd
from paths import paths, file_names

path = paths['data']['Input_to_Training_Model']
all_files = glob.glob(os.path.join(path, paths['data']['Raw_MRI_location']))	#'ADNI_reg_nii/*'))

label_file = file_names['data']['MRI_to_curr_label_mapping']
df = pd.read_csv(os.path.join(path, label_file))

df_new = pd.DataFrame()

rids = df['RID']
rids = list(set(rids))
print len(rids)

for rid in rids:
	subframe = df.loc[df['RID'] == rid]
	subframe = subframe.sort_values(by=['EXAMDATE'])
	
	if len(subframe) > 1:
		indices = list(subframe.index)
		for i, idx in enumerate(indices[:-1]):
			# pdb.set_trace()
			subframe.set_value(idx, 'DIAGNOSIS_LABEL', subframe.loc[indices[i + 1]]['DIAGNOSIS_LABEL'])
		
		subframe = subframe.drop(subframe.index[len(subframe) - 1])
		
		df_new = pd.concat([df_new, subframe], ignore_index=True)

df_new.to_csv(os.path.join(path, file_names['data']['MRI_to_next_label_mapping']))