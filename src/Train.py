#! /usr/local/bin/python
"""" Author : Adnan Haider"""
import sys
sys.path.append("/home/dawna/mah90/Kaggle/Kaggle-PlanetAmazon/src")
sys.path.append("/home/dawna/mah90/Kaggle/Kaggle-PlanetAmazon/src/lib")
import tensorflow as tf
#import lib to construct DNN architecture
import ConvLib
import readwrite
import minibatch
import cv2
import numpy as np
import savegraph

inputdatafile = sys.argv[1]
labelfile = sys.argv[2]

#define appropiate placeholders for input and output
classes =17 # for  planetary dataset, classes = 17
height = 128
width = 128
channels = 4
Y = tf.placeholder(tf.int32, shape=[None,classes])
images,Y_cloudy,Y_Atomosphere,Y_rest  = ConvLib.placeholder_Planetaryinputs(height,width,channels,classes)




# Each number in the dictionary corresponds to layer id
layerWeights = {1: [5,5,4,64],2: [5,5,64,32], 3:[5,5,32,16] }
poolingWindows ={1: [[1,2,2,1],[1,2,2,1]],2 :[[1,2,2,1],[1,2,2,1]],3: [[1,2,2,1],[1,2,2,1]]}
denseLayerDim = 1024
"""Note in Poolingwindows the first tensor in [[1,2,2,1],[1,2,2,1]] descibes the shape of the pooling window while the second tensor
describes the stride"""


#Create a DNN architecture with intermediate layers acting as conv nets
convNet = ConvLib.constructConvNet(images,layerWeights=layerWeights,poolingWindows=poolingWindows,ReLUON=True)
denseLayer = ConvLib.createFullConnectedDenseLayer(convNet, [32* 32 * 16, denseLayerDim])
output_Y_cloudy = ConvLib.cloudy_logit(denseLayer,denseLayerDim)
output_Y_atmosphere = ConvLib.atmos_logit(denseLayer,denseLayerDim)
output_land_logit =ConvLib.land_logit(denseLayer,denseLayerDim)
output_Y_atmosphere

objectiveFunc = ConvLib.constructObjectiveFunction(cloudy_output=output_Y_cloudy, atmosphere_output=output_Y_atmosphere,rest_output=output_land_logit,Y_cloudy=Y_cloudy,Y_Atomosphere=Y_Atomosphere,Y_rest=Y_rest)
train_step = tf.train.AdamOptimizer(1e-4).minimize(objectiveFunc)

#make predictions
prediction =  ConvLib.predictLabels(cloudy_output=output_Y_cloudy,atmosphere_output=output_Y_atmosphere,rest_output=output_land_logit)
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



"""" START TRAINING """
#intialise variables
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
data_wrapper = minibatch.MiniBatch(inputdatafile, labelfile, seed=10 ,image_dim=128)
for i in range(100):
  images_data,labels = data_wrapper.GetMinibatch(10)
  cloudy_label,atmos_label,rest_label = data_wrapper.createLabelpartitions(10,labels)
  train_step.run(feed_dict={images:images_data,Y_cloudy:cloudy_label,Y_Atomosphere:atmos_label,Y_rest:rest_label})
  if i%10 ==0:
  	training_accuracy = accuracy.eval(feed_dict={images:images_data, Y:labels})
  	#save variable graph
  	savegraph.Save('model',sess)
  	# graph.get_operation_by_name('cloudy/cloudy_output')
	print("step %d, training accuracy %g"%(i, training_accuracy))
  
	 

