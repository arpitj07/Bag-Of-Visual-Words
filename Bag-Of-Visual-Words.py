
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


args = argparse.ArgumentParser()

arg = args.add_argument('-d' , '--datapath' ,
							 help='path to data')
arg = args.add_argument('-t' , '--train' , 
							help = 'path to training images')
arg = args.add_argument('-ts' , '--test' ,
							 help = 'path to test images')



if "__init__" ==main:


	