#! /usr/local/bin/python
"""" Author : Adnan Haider"""
import sys
sys.path.append("/home/dawna/mah90/Kaggle/Kaggle-PlanetAmazon/src")
import tensorflow as tf
#import lib to construct DNN architecture
import lib.ConvLib


#define appropiate placeholders for input and output
classes =17 # for  planetary dataset, classes = 17
height = 256
width = 256
channels = 4
x = tf.placeholder(tf.float32, shape=[None, 262144])
Y = tf.placeholder(tf.int32, shape=[None,classes])
images_placeholder,Y_cloudy,Y_Atomosphere,Y_rest  =ConvLib.placeholder_Planetaryinputs(images,height,width,channels,classes)




# Each number in the dictionary corresponds to layer id
layerWeights = {1: [5,5,1,32],2: [5,5,32,64]}
poolingWindows ={1: [[1,2,2,1],[1,2,2,1]],2 :[[1,2,2,1],[1,2,2,1]]}
denseLayerDim = 1024
"""Note in Poolingwindows the first tensor in [[1,2,2,1],[1,2,2,1]] descibes the shape of the pooling window while the second tensor
describes the stride"""


#Create a DNN architecture with intermediate layers acting as conv nets
convNet = ConvLib.constructConvNet(images,layerWeights=layerWeights,poolingWindows=poolingWindows,ReLUON=True)
denseLayer = ConvLib.createFullConnectedDenseLayer(convNet, [64* 64 * 64, denseLayerDim])
output_Y_cloudy = ConvLib.cloudy_logit(denselayer,denseLayerDim)
output_Y_atmosphere = ConvLib.atmos_logit(denseLayer,denseLayerDim)
output_land_logit =ConvLib.land_logit(denseLayer,denseLayerDim)


objectiveFunc = ConvLib.constructObjectiveFunction(cloudy_output=output_Y_cloudy, atmosphere_output=output_Y_atmosphere,rest_output=output_land_logit,Y_cloudy=Y_cloudy,Y_Atomosphere=Y_Atomosphere,Y_rest=Y_rest)
train_step = tf.train.AdamOptimizer(1e-4).minimize(objectiveFunc)

#make predictions
prediction =  ConvLib.predictLabels(cloudy_output=output_Y_cloudy,atmosphere_output=output_Y_atmosphere,rest_output=output_land_logit)
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



"""" START TRAINING """
#intialise variables
sess.run(tf.global_variables_initializer())


