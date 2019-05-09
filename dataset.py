import pickle
from glob import glob
import numpy as np



def unpickle(path , token):

	if token =="train":
		for each in glob(path+"data*"):
			file = each.split("/")[-1]
			with open(file , 'rb') as fo:
				d = pickle.load(fo,encoding='bytes')
	elif token == 'test':
		for each in glob(path + 'test*'):
			file = each.split("/")[-1]
			with open(file , 'rb') as fo:
				d = pickle.load(fo , encoding='bytes')
	
	features = d[b'data'].reshape((len(d[b'data'])),3,32,32).transpose(0,2,3,1)
	labels = d[b'labels']
	filename = d[b'filenames']

	return features, labels[:500]




