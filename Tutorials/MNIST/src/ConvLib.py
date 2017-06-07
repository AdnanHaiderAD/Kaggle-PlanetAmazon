#! /usr/local/bin/python
"""" Author : Adnan Haider"""
import sys
sys.path.append("/home/dawna/mah90/Kaggle/Kaggle-PlanetAmazon/Tutorials/MNIST/src")
import tensorflow as tf

#function to setup input and label nodes for regular classification such as MNIST
def placeholder_Regularinputs(images,height,width,channels,classes):
  """Generate placeholder variables to represent the input tensors.
  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.
  Returns:
    images_placeholder: Images placeholder that is reshaped from a vector to an image
    labels_placeholder: Labels placeholder.
        """
        # Note that the shapes of the placeholders match the shapes of the full
        # image and label tensors, except the first dimension is now batch_size
        # rather than the full size of the train or test data sets.
  images_placeholder = tf.reshape(images,[-1,height,width,channels])
  labels_placeholder = tf.placeholder(tf.int32, shape=[None,classes])
  return images_placeholder, labels_placeholder

#function to setup input and label nodes for the planetary multimodal classification task
def placeholder_Planetaryinputs(images,height,width,channels,classes):
  """Generate placeholder variables to represent the input tensors.
  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.
  Args:
    batch_size: The batch size will be baked into both placeholders.
  Returns:
    images_placeholder: Images placeholder that is reshaped from a vector to an image
    labels_placeholder: Labels placeholder.
        """
        # Note that the shapes of the image placeholders match the shapes of the full
        # image  xcept the first dimension is now batch_size
  images_placeholder = tf.placeholder(tf.float32, shape=[None,height,width,channels])
  Y_cloudy = tf.placeholder(tf.int32, shape=[None,1])
  Y_Atomosphere = tf.placeholder(tf.int32, shape=[None,3])
  Y_rest = tf.placeholder(tf.int32, shape=[None,classes-4])
  return images_placeholder,Y_cloudy,Y_Atomosphere,Y_rest 

#functions to create weight and bias objects
def weight_variable(shape):
  """ for conv nets , typical shape will [height,width,inputchanels,numberoffeaturemaps]"""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  """ the default activation function is Relus so to ensure intially all hiddens units are active,b =0.1"""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1]):
  """default : pooling window size 2 by 2 and strides by 1"""
  return tf.nn.max_pool(x,ksize,strides,padding='SAME')



def constructConvNet(X,layerWeights,poolingWindows={}):
  """ This functions creates a Deep convolutional network upto a dense layer
  	 Inputs :
     X : input
     layerWeights  is dictionary of weights shapes . Eg layerWeights = {1: [5,5,1,32],2: [5,5,32,64]}
     poolingWindows is dictionary of Ksizes : Eg {1:[1,2,2,1]}
     outputs:
  creates a deep convolutional network architecture"""
  count = 1
  hiddenLayers ={}
  while count <= len(layerWeights.keys()):
    with tf.name_scope('hidden'+str(count)):
      W = weight_variable(layerWeights[count])
      b = bias_variable(layerWeights[count][3:4])
      if count == 1:
         activation = conv2d(X,W) + b
      else:
        activation = conv2d(hiddenLayers[count-1],W)+b
      non_linearActivation = tf.nn.relu(activation)
      if len(poolingWindows) ==0:
        hiddenLayers[count] = max_pool(non_linearActivation)
      else:
       hiddenLayers[count] = max_pool(non_linearActivation,ksize=poolingWindows[count][0],strides=poolingWindows[count][1])
    count+=1
  return hiddenLayers[count-1]


def createFullConnectedDenseLayer(convNetLayer, dims):
  """ inputs:  Convetnet layer, shape : dimensions of the weight matrix [ inputdim, outputdim]"""
  W_fc1 = weight_variable(dims)
  b_fc1 = bias_variable(dims[1:])
  h_pool2_flat = tf.reshape(convNetLayer, [-1, dims[0]])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
  return h_fc1

def constructObjectiveFunction(DenseLayer, DenseLayer_dim,classes,Y_cloudy,Y_Atomosphere,Y_rest):
	""" assumes the labels are partioned as follows Y =(y1,y2,...yk) -> y1<-cloudy,(y2..y4)<-Y-Atmosphere
	"""
	W_cloudy = weight_variable([DenseLayer_dim, 1])
	b_cloudy = bias_variable([1])
	output_c = tf.matmul(DenseLayer,W_cloudy)+ b_cloudy
	loss_c = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_cloudy, logits=output_c)

	W_Atmosphere = weight_variable([DenseLayer_dim,3])
	b_Atmosphere = 	bias_variable([3])
	output_Atmosphere = tf.matmul(DenseLayer,W_Atmosphere)+ b_Atmosphere
	loss_Atmosphere = tf.nn.softmax_cross_entropy_with_logits(labels=Y_Atomosphere, logits=output_Atmosphere)

	W_rest = weight_variable([DenseLayer_dim,classes-4])
	b_rest = bias_variable([classes-4])
	output_R = tf.matmul(DenseLayer,W_rest)+ b_rest
	loss_R = f.nn.sigmoid_cross_entropy_with_logits(labels=Y_rest, logits=output_R)

	loss = loss_c + (1-Y_cloudy) *(loss_Atmosphere + loss_R)
	return loss
