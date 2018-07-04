import os

from configurations.paths import paths, file_names
from configurations.modelConfig import layer_config, params, data_aug

from data.dataLoader import dataLoader, run_test_

from models.vae import VAE
from train import Trainer
import time
import torch

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
	model = VAE()
	
	# count model parameters
	print('Paramater Count :', sum(p.numel() for p in model.parameters()))
	
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
	
'''
from utils.visualizations import run_test

run_test()
'''