import os

from configurations.paths import paths, file_names
from configurations.modelConfig import layer_config, params, data_aug

from data.dataLoader import dataLoader, run_test_

from models.probUnet import prior, posterior, generator, probCNN
from train import Trainer
import time
import torch

def main():
	torch.multiprocessing.set_sharing_strategy('file_system')
	#torch.backends.cudnn.enabled = False
	# create the experiment dirs
	timestr = time.strftime("%Y%m%d-%H%M%S")
	base_folder = paths['output']['base_folder']
	expt_folder = base_folder + timestr
	if not os.path.exists(expt_folder):
		os.mkdir(expt_folder)
	
	# set seed
	torch.manual_seed(params['train']['seed'])
		
	print('Run : {}\n'.format(timestr))

	# create an instance of the model\
	prior_network = prior()
	posterior_network = posterior()
	generator_network = generator()
	model = probCNN(prior_network, posterior_network, generator_network)
	#prior_network.volatile = True; posterior_network.volatile = True; generator_network.volatile = True;
	#model.volatile = True
	
	'''
	pretrained_dict = torch.load(
		'/home/ml/sbasu11/Documents/ADNI Project/ADNI_data/CNN/Outputs/20180719-210645/latest_model.pkl')
	
	# load the new state dict
	model.load_state_dict(pretrained_dict)
	'''
	
	'''
	# load pretrained weights
	pretrained_dict = torch.load(
		'/home/ml/sbasu11/Documents/ADNI Project/ADNI_data/CNN/Outputs/20180802-151154/latest_model.pkl')
	
	# load the new state dict
	model.load_state_dict(pretrained_dict)
	'''
	
	# count model parameters
	print('Paramater Count :', sum(p.numel() for p in model.parameters()))
	
	# create data generator
	datafile = os.path.join(paths['data']['hdf5_path'], file_names['data']['hdf5_file'])
	
	train_loader, valid_loader, test_loader = dataLoader(datafile, trans=data_aug)

	# create trainer and pass all required components to it
	trainer = Trainer(prior_network, posterior_network, generator_network, model, train_loader, valid_loader,
					  expt_folder)

	# train model
	trainer.train()
	
	# test model
	trainer.test(test_loader)
	
if __name__ == '__main__':
	#run_test_()
	main()