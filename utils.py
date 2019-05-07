import cv2
from glob import glob
from dataset import *
from scipy.misc import imsave
from PIL import Image


class saveImage:

	def save(self , features):
		for i in range(len(features)):

			name = "Images/image_" + i + ".jpg"
			im = Image.fromarray(features[i])
			im.save(name)


class FileHelper:

	def getFiles(self,path):
		imlist = []
		count =0

		for image in glob(path + "*"):
			#word = each.split("/")[-1]
			print(" ---- Reading Images ----")
			im = cv2.imread(image)
			imlist.append(im)
			count+=1

		return imlist,count



class imageHelpers:

	def __init__(self):
		self.sift_object = cv2.xfeatures2d.SIFT_create()

	def grayscale(self, image):
		gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
		return gray

	def features(self, image):
		keypoints , descriptors = self.sift_object.detectAndCompute(image , None)
		return keypoints , descriptors

