#! /usr/local/bin/python
import sys
sys.path.append("/home/dawna/mah90/Kaggle/Kaggle-PlanetAmazon/Tutorials/MNIST/src")

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


import tensorflow as tf
sess = tf.InteractiveSession()


FEA_DIM = 784
CLASSES = 10
hiddenUnits = [1000, 700 ,500]
# definiing input and output

def placeholder_inputs(fea_dim,classes):
  """Generate placeholder variables to represent the input tensors.
  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.
  Args:
    batch_size: The batch size will be baked into both placeholders.
  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=[None,fea_dim])
  labels_placeholder = tf.placeholder(tf.int32, shape=[None,classes])
  return images_placeholder, labels_placeholder



#functions to create weight and bias objects
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def inference(images,list_hiddenunits):
  count = 0
  hiddenLayers ={}
  while count < len(list_hiddenunits):
    with tf.name_scope('hidden'+str(count)):
      if count == 0:
        W = weight_variable([FEA_DIM,list_hiddenunits[count]])
        b = bias_variable([list_hiddenunits[count]])
        hiddenLayers[count] = tf.nn.relu(tf.matmul(images, W) + b)
      else :
        W = weight_variable([list_hiddenunits[count-1],list_hiddenunits[count]])
        b = bias_variable([list_hiddenunits[count]])
        hiddenLayers[count] = tf.nn.relu(tf.matmul(hiddenLayers[count-1], W) + b)
    count+=1    
  with tf.name_scope('softmax_linear'):
    weights = weight_variable([list_hiddenunits[count-1], CLASSES])
    biases = bias_variable([CLASSES])
    logits = tf.matmul(hiddenLayers[count-1], weights) + biases
  return logits

def loss (logits, labels):
  """Calculates the loss from the logits and the labels.
  Args:
  logits: Logits tensor, float - [batch_size, NUM_CLASSES].
  labels: Labels tensor, int32 - [batch_size]."""
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
  return cross_entropy


(x,y_) = placeholder_inputs(FEA_DIM, CLASSES)
logits = inference(x,hiddenUnits)
cross_entropy = loss(logits,y_)


train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


# evaluation on test data
correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#intialise variables
sess.run(tf.global_variables_initializer())

for i in range(8000):
  #get next batch
  batch = mnist.train.next_batch(500)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
 


