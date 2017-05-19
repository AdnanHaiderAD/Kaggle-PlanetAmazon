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


#A Variable is a value that lives in TensorFlow's computation graph. It can be used and even modified by the computation
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#Before Variables can be used within a session, they must be initialized using that session
sess.run(tf.global_variables_initializer())

y = tf.matmul(x,W) + b

#measure loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

#train using steepest descent

#What TensorFlow actually did in that single line was to add new operations to the computation graph. These operations included ones to compute gradients, compute parameter update steps, and apply update steps to the parameters.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#do training in seesion
for _ in range(1000):
  batch = mnist.train.next_batch(100)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})



#After training, checking how well we have done on the test data
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


