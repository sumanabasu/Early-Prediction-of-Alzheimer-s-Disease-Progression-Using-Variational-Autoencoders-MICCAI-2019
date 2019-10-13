import h5py
import os
from configurations.paths import paths, file_names
from configurations.modelConfig import params, layer_config
from data.splitDataset import getIndicesTrainValidTest
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch.autograd import Variable

import matplotlib as mpl
# %matplotlib inline
mpl.use('agg')
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
from skimage.transform import resize, rotate

from models.vae import VAE
import random

seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(seed)

num_classes = 2

datafile = os.path.join('/home/ml/sbasu11/Documents/ADNI Project/ADNI_data/CNN/Inputs/', file_names['data']['hdf5_file'])

class HDF5loader():
	def __init__(self, filename, train_indices=None):
		f = h5py.File(filename, 'r',  libver='latest', swmr=True)
		self.fn = f['FileName']
		self.img_f = f['Image4D']
		self.train_indices = train_indices
		if num_classes == 2:
			self.label = [0 if x == 'NL' else 1 for x in f[params['train']['timestamp']]]
		else:
			self.label = [0 if x == 'NL' else (2 if x == 'AD' else 1) for x in f[params['train']['timestamp']]]

		self.currlabel = [0 if x == 'NL' else 1 for x in f['CurrLabel']]
		#self.label = [0 if x == 'NL' else (2 if x == 'AD' else 1) for x in f['NextLabel']]
		#self.label = [0 if x == 'NL' else 1 for x in f['CurrLabel']] #for current label	#for binary
		# classification on current label
		#self.label = [0 if x == 'NL' else 1 for x in f['NextLabel']]	#for binary classification on next label

	def __getitem__(self, index):
		img = self.img_f[index]
		label = self.label[index]

		#print('original', img.shape) #(1, 189, 233, 197)

		# for coronal view (channels, depth, 0, 1)
		img = np.moveaxis(img, 1, 3)
		#print('1. ', img.shape)	#(1, 233, 197, 189)

		# drop 10 slices on either side since they are mostly black
		img = img[:, 10:-10, ::]

		# reshape to (depth, 0, 1, channels) for data augmentation
		img = np.moveaxis(img, 0, 3)
		#print('2. ', img.shape)	#(213, 197, 189, 1)

		# # random transformation
		# if self.trans is not None and index in self.train_indices:
		#     img = random_transform(img, **self.trans)

		# reshape back to (channels, depth, 0, 1)
		img = np.moveaxis(img, 3, 0)
		#print('3. ', img.shape)	#(1, 213, 197, 189)

		#normalizing image - Gaussian normalization per volume
		if np.std(img) != 0:  # to account for black images
			mean = np.mean(img)
			std = np.std(img)
			img = 1.0 * (img - mean) / std

		img = img.astype(float)
		img = torch.from_numpy(img).float()
		label = torch.LongTensor([label])

		return (img ,label, self.fn[index], self.currlabel[index])

	def __len__(self):
		return self.img_f.shape[0]


def dataLoader(hdf5_file):
	num_workers = 0
	pin_memory = False
	
	train_indices, valid_indices, test_indices = getIndicesTrainValidTest()
	
	test_sampler = SubsetRandomSampler(test_indices)
	
	data = HDF5loader(hdf5_file, train_indices=train_indices)
	
	test_loader = DataLoader(data, batch_size=params['train']['batch_size'], sampler=test_sampler,
							 num_workers=num_workers, pin_memory=pin_memory)
	
	return (test_loader)

test_loader = dataLoader(datafile)

model = VAE()


pretrained_dict = torch.load(
	'/home/ml/sbasu11/Documents/ADNI Project/ADNI_data/CNN/Outputs/20180719-210645/latest_model.pkl')

# load the new state dict
model.load_state_dict(pretrained_dict)
model.cuda()

activation = {}
def get_activation(name):
	def hook(model, input, output):
		activation[name] = output.detach()
	return hook

model.conv4.register_forward_hook(get_activation('conv4'))


def save_frames(clean_data, activation, slice_n, data_index,folder):
	"""
	:param folder:
	:param clean_data: one sample of B x 1 x H x W x D
	:param activation: one sample of B x channel x H_k x W_k x D_k
	:slice_n : slice number
	"""
	#     mpl.use('Agg')
	#     import matplotlib.pyplot as plt
	if not os.path.exists(folder):
		os.mkdir(folder)
	
	plt.axis('off')
	single_data = clean_data[data_index].squeeze(0).cpu().data.numpy()
	one_act = activation[data_index].float().cpu().data.numpy()
	# upsampling
	resized_act = resize(one_act, output_shape=(single_data.shape))
	
	brain = rotate(single_data[slice_n], 90)
	# cam = rotate(resized_act[slice_n], 90)
	plt.imshow(brain, cmap='gray')
	cmap = mpl.cm.Reds_r(np.linspace(0, 1, 20))
	cmap = mpl.colors.ListedColormap(cmap[0:, :-1])
	resized_act[slice_n] = (resized_act[slice_n] - np.min(resized_act[slice_n])) / (
				np.max(resized_act[slice_n]) - np.min(resized_act[slice_n]))
	saliency = cmap(resized_act[slice_n])
	saliency[..., -1] = 0.7 - resized_act[slice_n] * 0.7
	saliency = rotate(saliency, 90)
	plt.imshow(saliency)
	plt.savefig(os.path.join(folder, 'file%03d.png' % slice_n))


data_X = None
data_Y = None
# # done = {'nl_nl': 0, 'nl_ad': 0, 'ad_ad': 0}
# # done = {'nl_ad': 0}
# # fig, axes = plt.subplots(1, 1, figsize=(12, 6), subplot_kw={'xticks': [], 'yticks': []})
#
# # fig.subplots_adjust(hspace=0.3, wspace=0.05)
#
# for j, [images, nextlabels, fn, currlabels] in enumerate(test_loader):
# 	print(j)
# 	data_X = Variable(images).cuda()
# 	data_Y = Variable(nextlabels).cuda()
# 	for i in range(len(images)):
# 		currlabel = currlabels[i]
# 		nextlabel = int(nextlabels[i].cpu().numpy())
#
# 		# if currlabel == 0 and nextlabel == 0:
# 		# 	label = 'nl_nl'
# 		# 	lab = 0
# 		if currlabel == 0 and nextlabel == 1:
# 			label = 'nl_ad'
# 			lab = 1
# 			print(fn[i])
# 		# elif currlabel == 1 and nextlabel == 1:
# 		# 	label = 'ad_ad'
# 		# 	lab = 1
# 		else:
# 			continue
# 		print(label)
# 		#             label = int(data_Y[0].data.cpu().numpy()[0])
# 		# if done[label] >= 1:
# 		# 	continue
# 		# else:
# 		# 	done[label] += 1
# 		data = data_X[0]
# 		data = data.unsqueeze(0)
# 		_, _, _, _, _, p_hat = model.forward(data)
# 		p_hat = torch.exp(p_hat).cpu().data.numpy()[0]
# 		print(p_hat)
# 		# if p_hat[lab] < 0.8:
# 		# 	print(fn[i])
# 		# 	continue
# 		# else:
# 		# 	done[label] += 1
# 		act = activation['conv4'].squeeze(0)
# 		# for slc in range(data.shape[2]):
# 		for slc in list([53, 91, 103, 124, 152]):
# 			print(slc)
# 			save_frames(data_X, act, slc, 0, 'slices_png_'+label+'_'+fn[i]+'_'+str(p_hat[lab]))
# 		print(j, label, fn[i], currlabel, nextlabel, p_hat)
# 		# break
# 	# if done['nl_nl'] >= 1 and done['nl_ad'] >= 1 and done['ad_ad'] >= 1:
# 	# if done['nl_nl'] >= 1:
# 	# 	break

# done = {'nl_nl': 0, 'nl_ad': 0, 'ad_ad': 0}
done = {'nl_ad': 0}

for j, [images, nextlabels, fn, currlabels] in enumerate(test_loader):
	print(j)
	data_X = Variable(images).cuda()
	data_Y = Variable(nextlabels).cuda()
	for i in range(len(images)):
		currlabel = currlabels[i]
		nextlabel = int(nextlabels[i].cpu().numpy())
		
		# if currlabel == 0 and nextlabel == 0:
		# 	label = 'nl_nl'
		# 	lab = 0
		if currlabel == 0 and nextlabel == 1:
			label = 'nl_ad'
			lab = 1
			print(fn[i])
		# elif currlabel == 1 and nextlabel == 1:
		# 	label = 'ad_ad'
		# 	lab = 1
		else:
			continue
		print(label)
		#             label = int(data_Y[0].data.cpu().numpy()[0])
		if done[label] >= 1:
			continue
		# else:
		# 	done[label] += 1
		data = data_X[0]
		data = data.unsqueeze(0)
		_, _, _, _, _, p_hat = model.forward(data)
		p_hat = torch.exp(p_hat).cpu().data.numpy()[0]
		print(p_hat)
		if p_hat[lab] < 0.8:
			print(fn[i])
			continue
		else:
			done[label] += 1
		act = activation['conv4'].squeeze(0)
		for slc in range(data.shape[2]):
		# for slc in list([53, 91, 103, 124, 152]):
			print(slc)
			save_frames(data_X, act, slc, 0, 'small_brains/slices_png_' + label + '_' + fn[i] + '_' + str(p_hat[lab]))
		print(j, label, fn[i], currlabel, nextlabel, p_hat)
		break
	# if done['nl_nl'] >= 1 and done['nl_ad'] >= 1 and done['ad_ad'] >= 1:
	if done['nl_ad'] >= 1:
		break



