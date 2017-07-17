import cv2
import tensorflow as tf
import csv

# import image
img=cv2.imread('train-tif/train_10051.tif',cv2.IMREAD_UNCHANGED)

# open csv labels file
with open('train.csv') as file:
	labels_reader = csv.reader(file, delimiter=',')
	for x in labels_reader:
		



#sess = tf.InteractiveSession()

# Input dimension is 256*256*4
#input = tf.placeholder(tf.float32, shape=[None, 262144])
