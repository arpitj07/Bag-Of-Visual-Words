import cv2
from glob import glob
from dataset import *
from scipy.misc import imsave
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class saveImage:

	def save(self , features , image_path , token):
		for i in range(500):

			name = image_path + "/image_" + str(i) + ".jpg"
			im = Image.fromarray(features[i])
			im.save(name)
		print("[Success] All" , str(token), "images are saved.....")

		return im


class FileHelper:

	def getFiles(self,path):
		imlist = []
		count =0

		for each in glob(path + "*.jpg"):
			#image = each.split("\\")[-1]
			
			im = cv2.imread(each)
			imlist.append(im)
			count+=1
		print("[Success] Reading Images complete.....")
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





class BovHelper:

	def __init__(self , n_clusters=20):

		self.clf = SVC()
		self.KMeans_obj = KMeans(n_clusters=n_clusters)
		self.KMeans_ret = None
		self.n_clusters = n_clusters
		self.descriptor_vstack = None
		self.mega_histogram = None


	def cluster(self):
		self.KMeans_ret = self.KMeans_obj.fit_predict(self.descriptor_vstack)


	def developVocabulory(self , n_images , descriptor_list , KMeans_ret=None):
		
		self.mega_histogram = np.array([np.zeros(self.n_clusters) for i in range(n_images)])
		old_count=0

		for i in range(n_images):
			l=len(descriptor_list[i])
			for j in range(l):
				if KMeans_ret is None:
					idx = self.KMeans_ret[old_count+j]
				else:
					idx = self.KMeans_ret[old_count+j]
				self.mega_histogram[i][idx] +=1
			old_count+=1

		print("----vocab hist created-----")



	def formatND(self , l):
	
		vStack = np.array(l[0])
		for i in range(1,305):
			v = np.vstack((vStack , l[i]))
			vStack =v
		self.descriptor_vstack = vStack.copy()
		print("[Success] stack creates ....")
		return vStack

	def standardize(self , std=None):

		if std is None:
			self.scale = StandardScaler().fit(self.mega_histogram)
			self.mega_histogram = self.scale.transform(self.mega_histogram)

		else:
			print("STD is not none external std supplied")
			self.mega_histogram = std.transform(self.mega_histogram)



	def train(self , train_labels):

		print("Training SVM")
		self.clf.fit(self.mega_histogram , train_labels)
		print("Training complete.....")


	def predict(self , iplist):

		predictions = self.clf.predict(iplist)
		return predictions




