
"""
Steps involved in BOVW:

1. Determination of Image features for a given label
2. Construction of visual vocabulory by clutering 
3. frequency analysis
4. classification based on vocabulory generated

"""

import cv2
from sklearn.svm import SVC
import numpy as np
import argparse
from utils import * 
from dataset import *
from Test import *
from Train import TRAIN


class BOV:

	def __init__(self, cluster):

		self.cluster = cluster
		self.train_path = None
		self.test_path = None
		self.datapath = None
		self.imagehelpers = imageHelpers()
		self.saveimage = saveImage()
		self.file_helper = FileHelper()
		self.bov_helper = BovHelper()
		self.images = None
		self.descriptor_list =[]


	def recognise(self , testimages , test_path=None):

		key , des = self.imagehelpers.features(testimages)
		print("Shape of descriptors:",des.shape)

		vocab = np.array([[0 for i in range(self.cluster)]])

		test_ret = self.bov_helper.KMeans_obj.predict(des)

		for each in test_ret:
			vocab[0][each] +=1
		lb = self.bov_helper.clf.predict(vocab)

		return lb



if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-d' , '--datapath' ,
							 help='path to data')
	parser.add_argument('-t' , '--train' , 
							help = 'path to training images')
	parser.add_argument('-ts' , '--test' ,
							 help = 'path to test images')
	args = parser.parse_args()



	bov = BOV(cluster=300)
	bov.datapath = args.datapath
	bov.train_path = args.train
	bov.test_path = args.test

	#extracting train data and labels
	features , labels = unpickle(bov.datapath , 'train')
	train_images = bov.saveimage.save(features , bov.train_path)
	#extracting test data and labels
	test_features , test_labels = unpickle(bov.datapath , 'test')
	test_images = bov.saveimage.save(test_features , bov.test_path)

	#training the model
	TRAIN(labels, bov.train_path).training()

	#testing the model
	TEST(bov.test_path).testing()


	