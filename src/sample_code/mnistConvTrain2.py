""" Author : Adnan Haider
Note: all routines have been tested with Mnist except the objective function
"""
import sys
sys.path.append("/home/dawna/mah90/Kaggle/Kaggle-PlanetAmazon/src")
sys.path.append("/home/dawna/mah90/Kaggle/Kaggle-PlanetAmazon/src/lib")

import tensorflow as tf
import ConvLib # import lib to create flexible convNets
sess = tf.InteractiveSession()

#To do experiments with MNIST

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#define appropiate placeholders for input and output
classes =10 # for  planetary dataset, classes = 17
x = tf.placeholder(tf.float32, shape=[None, 784])
images,y_ = ConvLib.placeholder_Regularinputs(x,28,28,1,classes)
""" For planetary problem, the input and output placeholders will be
images_placeholder,Y_cloudy,Y_Atomosphere,Y_rest  =ConvLib.placeholder_Planetaryinputs(images,height,width,channels,classes)
"""


# Each number in the dictionary corresponds to layer id
layerWeights = {1: [5,5,1,32],2: [5,5,32,64]}
poolingWindows ={1: [[1,2,2,1],[1,2,2,1]],2 :[[1,2,2,1],[1,2,2,1]]}
"""Note in Poolingwindows the first tensor in [[1,2,2,1],[1,2,2,1]] descibes the shape of the pooling window while the second tensor
describes the stride"""

#Create a deep Convnet
convNet = ConvLib.constructConvNet(images,layerWeights=layerWeights,poolingWindows=poolingWindows,ReLUON=True)
denseLayer = ConvLib.createFullConnectedDenseLayer(convNet, [7 * 7 * 64, 1024])
"""possibly add dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
"""

#output of network for MNIST
W_fc2 = ConvLib.weight_variable([1024, 10])
b_fc2 = ConvLib.bias_variable([10])
y_conv = tf.matmul(denseLayer, W_fc2) + b_fc2

#Build computational node to compute error: For MNIST
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
""" For planetary dataset:
loss = ConvLib.constructObjectiveFunction(DenseLayer=denseLayer, DenseLayer_dim,classes,Y_cloudy,Y_Atomosphere,Y_rest)
"""
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# evaluation on test data: FOR MNIST
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
##---NOte the evaluation criteria is not done for Kaggle yet




"""" START TRAINING """
#intialise variables
sess.run(tf.global_variables_initializer())

for i in range(2000):
        #get next batch
        batch = mnist.train.next_batch(10)
        print batch
        if i%100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]})
                print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
                                                                                                                                                                                                                     
