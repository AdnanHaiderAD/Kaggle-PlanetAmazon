#! /usr/local/bin/python
import sys
sys.path.append("/home/dawna/mah90/Kaggle/Kaggle-PlanetAmazon/Tutorials/MNIST/src")

import numpy as np
import tensorflow as tf

# Model parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)
# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
prediction = 1/(1+tf.exp(-linear_model))
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
# training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x:x_train, y:y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss,pred  = sess.run([W, b, loss,prediction], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s pred is %s "%(curr_W, curr_b, curr_loss,pred))
