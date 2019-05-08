import cv2
import numpy as np
from utils import *   
from BagOfVisualWords import BOV


class TEST:

	def __init__(self , testpath):

		self.images = None
		self.testpath = None
		self.count = None
		self.filehelper = FileHelper()
		self.testpath = testpath


	def testing(self):

		self.images , self.count = self.filehelper.getFiles(self.testpath)
		predictions = []

		#for _ , imlist in self.images.iteritems():
		for im in self.images:
			print(im.shape)

			cl = BOV.recognise(im)
			predictions.append({ "image" : im , "class": cl})

		print("The predcitons:" , predictions)

		for each in predictions:

			plt.imshow(cv2.cvtColor(each['image']))
			plt.show()
			plt.title(each['class'])




