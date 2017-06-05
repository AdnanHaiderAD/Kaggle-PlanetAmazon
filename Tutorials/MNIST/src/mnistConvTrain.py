#! /usr/local/bin/python
import sys
sys.path.append("/home/dawna/mah90/Kaggle/Kaggle-PlanetAmazon/Tutorials/MNIST/src")

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


import tensorflow as tf
sess = tf.InteractiveSession()

# definiing input and output
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


#functions to create weight and bias objects
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


#defining convulational operations

#Computes a 2-D convolution given 4-D `input` and `filter` tensors.\n\n  Given an input tensor of shape `[batch, in_height, in_width, in_channels]`\n  and a filter / kernel tensor of shape\n  `[filter_height, filter_width, in_channels, out_channels]`, this op\n  performs the following:\n\n  1. Flattens the filter to a 2-D matrix with shape\n     `[filter_height * filter_width * in_channels, output_channels]`.\n  2. Extracts image patches from the input tensor to form a *virtual*\n     tensor of shape `[batch, out_height, out_width,\n     filter_height * filter_width * in_channels]`.\n  3. For each patch, right-multiplies the filter matrix and the image patch\n     vector.\n\n  In detail, with the default NHWC format,\n\n      output[b, i, j, k] =\n          sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *\n                          filter[di, dj, q, k]\n\n  Must have `strides[0] = strides[3] = 1`.  For the most common case of the same\n  horizontal and vertices strides, `strides = [1, stride, stride, 1]`.\n\n  Args:\n    input: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.\n      A 4-D tensor. The dimension order is interpreted according to the value\n      of `data_format`, see below for details.\n    filter: A `Tensor`. Must have the same type as `input`.\n      A 4-D tensor of shape\n      `[filter_height, filter_width, in_channels, out_channels]`\n    strides: A list of `ints`.\n      1-D tensor of length 4.  The stride of the sliding window for each\n      dimension of `input`. The dimension order is determined by the value of\n        `data_format`, see below for details.\n    padding: A `string` from: `"SAME", "VALID"`.\n      The type of padding algorithm to use.\n    use_cudnn_on_gpu: An optional `bool`. Defaults to `True`.\n    data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.\n      Specify the data format of the input and output data. With the\n      default format "NHWC", the data is stored in the order of:\n          [batch, height, width, channels].\n      Alternatively, the format could be "NCHW", the data storage order of:\n          [batch, channels, height, width].\n    name: A name for the operation (optional).\n\n  Returns:\n    A `Tensor`. Has the same type as `input`.\n    A 4-D tensor. The dimension order is determined by the value of\n    `data_format`, see below for details.\n 

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Performs the max pooling on the input.\n\n  Args:\n    value: A 4-D `Tensor` with shape `[batch, height, width, channels]` and\n      type `tf.float32`.\n    ksize: A list of ints that has length >= 4.  The size of the window for\n      each dimension of the input tensor.\n    strides: A list of ints that has length >= 4.  The stride of the sliding\n      window for each dimension of the input tensor.\n    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.\n      See the @{tf.nn.convolution$comment here}\n    data_format: A string. 'NHWC' and 'NCHW' are supported.\n    name: Optional name for the operation.\n\n  Returns:\n    A `Tensor` with type `tf.float32`.  The max pooled output tensor.\n "

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')



W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

#convolve the image
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


#2nd convolution layer 
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


#After very max-pooling, th size of the images decrease

#now attaching a connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Apply dropout at the penultimate layer
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

#output of network

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#Build computational node to compute error:
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
cross_entropy =  tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv), reduction_indices=1))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


# evaluation on test data
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#intialise variables
sess.run(tf.global_variables_initializer())

for i in range(2000):
  #get next batch
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    	train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
    	print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
