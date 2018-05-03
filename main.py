import os
from paths import paths, file_names
from dataLoader import dataLoader
from cnn import CnnVanilla
from train import Trainer

def main():
	#TODO: create the experiments dirs

	#TODO: create an instance of the model you want
	model = CnnVanilla()

	# create your data generator
	datafile = os.path.join(paths['data']['hadf5_path'], file_names['data']['hdf5_file'])
	train_loader, valid_loader, test_loader = dataLoader(datafile)

	# create trainer and pass all required components to it
	trainer = Trainer(model, train_loader)

	# train your model
		
if __name__ == '__main__':
	main()
	
	

