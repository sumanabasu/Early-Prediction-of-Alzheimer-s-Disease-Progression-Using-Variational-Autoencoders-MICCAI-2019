import os
from paths import paths, file_names
from dataLoader import dataLoader
from cnn import CnnVanilla
from train import Trainer
import time

def main():
	# create the experiments dirs
	timestr = time.strftime("%Y%m%d-%H%M%S")
	base_folder = paths['output']['base_folder']
	expt_folder = base_folder + timestr
	if not os.path.exists(expt_folder):
		os.mkdir(expt_folder)
		
	print('Run : {}\n'.format(timestr))

	# create an instance of the model you want
	model = CnnVanilla()

	# create your data generator
	datafile = os.path.join(paths['data']['hdf5_path'], file_names['data']['hdf5_file'])
	train_loader, valid_loader, test_loader = dataLoader(datafile)

	# create trainer and pass all required components to it
	trainer = Trainer(model, train_loader, valid_loader, expt_folder)

	# train your model
	trainer.train()
	
	# test model
	trainer.test(test_loader)
		
if __name__ == '__main__':
	main()
	
	

