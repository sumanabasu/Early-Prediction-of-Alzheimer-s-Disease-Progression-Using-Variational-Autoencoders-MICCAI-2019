import os
from paths import paths, file_names
from dataLoader import dataLoader
from cnn import CnnVanilla
from train import Trainer
import time
import torch
from modelConfig import data_aug

def main():
	torch.multiprocessing.set_sharing_strategy('file_system')
	# create the experiment dirs
	timestr = time.strftime("%Y%m%d-%H%M%S")
	base_folder = paths['output']['base_folder']
	expt_folder = base_folder + timestr
	if not os.path.exists(expt_folder):
		os.mkdir(expt_folder)
		
	print('Run : {}\n'.format(timestr))

	# create an instance of the model\
	model = CnnVanilla()

	# create data generator
	datafile = os.path.join(paths['data']['hdf5_path'], file_names['data']['hdf5_file'])
	
	train_loader, valid_loader, test_loader = dataLoader(datafile, trans=data_aug)

	# create trainer and pass all required components to it
	trainer = Trainer(model, train_loader, valid_loader, expt_folder)

	# train model
	trainer.train()
	
	# test model
	trainer.test(test_loader)
		
if __name__ == '__main__':
	main()
	
	

