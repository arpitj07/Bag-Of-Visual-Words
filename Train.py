import cv2
import numpy as np
from utils import *   


class TRAIN:

	def __init__(self , labels , trainpath):

		self.images = None
		self.trainpath = None
		self.count = None
		self.filehelper = FileHelper()
		self.img_helper = imageHelpers()
		self.descriptor_list = None
		self.bov_helper = BovHelper()
		self.train_labels = labels
		self.trainpath = trainpath

	def training(self ):
		
		self.images , self.count = self.filehelper.getFiles(self.trainpath)
		label_count=0
		#for _ , imlist in self.images.iteritems():
		for im in self.images:

			kp , des = self.img_helper.features(im)
			self.descriptor_list.append(des)

		#perform clustering

		bov_descriptor_stack = self.bov_helper.formatND(self.descriptor_list)
		self.bov_helper.cluster()
		self.bov_helper.developVocabulory(n_images=self.count , descriptor_list=self.descriptor_list)


		self.bov_helper.standardize()
		self.bov_helper.train(self.train_labels)





