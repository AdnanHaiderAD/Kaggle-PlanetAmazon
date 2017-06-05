import tensorflow as tf
import math

def build_dnn_graph(r_pl, g_pl, b_pl, i_pl, num_layers, num_nodes, each_input_dim, activation):
  """Build a feed-forward DNN.
  Args:
    *_pl: Placeholders for each input image channel
    num_layers: Number of hidden layers
    num_nodes: Number of nodes per hidden layer
    each_input_dim: Input dimension of each image channel
    activation: Activation function (sigmoid, relu)
  Returns:
    Tensor up to the final hidden layer. You need to build the output layer separately.
  """

  hidden_activations = [''] * num_layers
  for i in range(1, num_layers+1):
    name_scope = "hidden%d" % i
    with tf.name_scope(name_scope):
      if i == 1: # Input to first hidden layer
        r_weights = tf.Variable(tf.truncated_normal([each_input_dim, num_nodes], stddev=1.0 / math.sqrt(float(each_input_dim))), name='weights')
        g_weights = tf.Variable(tf.truncated_normal([each_input_dim, num_nodes], stddev=1.0 / math.sqrt(float(each_input_dim))), name='weights')
        b_weights = tf.Variable(tf.truncated_normal([each_input_dim, num_nodes], stddev=1.0 / math.sqrt(float(each_input_dim))), name='weights')
        i_weights = tf.Variable(tf.truncated_normal([each_input_dim, num_nodes], stddev=1.0 / math.sqrt(float(each_input_dim))), name='weights')
        linear = tf.matmul(r_pl, r_weights) + tf.matmul(g_pl, g_weights) + tf.matmul(b_pl, b_weights) + tf.matmul(i_pl, i_weights)

      else:
        weights = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], stddev=1.0 / math.sqrt(float(num_nodes))), name='weights')
        linear = tf.matmul(hidden_activations[i-2], weights)

      biases = tf.Variable(tf.zeros([num_nodes]), name='biases')
      if activation == 'sigmoid':
        hidden_activations[i-1] = tf.nn.sigmoid(linear + biases)
      elif activation == 'relu':
        hidden_activations[i-1] = tf.nn.relu(linear + biases)
      else:
        raise ValueError('Error: unknown activation function ' + activation)

  return hidden_activations[i-1]

def build_output_independent(final_hidden, num_nodes, softmax_dim, sigmoid_dim):
  """Build the output layer, where the atmospheric conditions are conditionally independent of the other labels
  Args:
    final_hidden: Tensor up to the final hidden layer
    num_nodes: Number of nodes per hidden layer. Must match final_hidden
    softmax_dim: Number of softmax output units
    sigmoid_dim: Number of sigmoid output units
  Returns:
    [softmax_logits, [sigmoid logits]]
  """

  with tf.name_scope('atmos_linear'):
    weights = tf.Variable(tf.truncated_normal([num_nodes, softmax_dim], stddev=1.0 / math.sqrt(float(num_nodes))), name='weights')
    biases = tf.Variable(tf.zeros([softmax_dim]), name='biases')
    softmax_logits = tf.matmul(final_hidden, weights) + biases

  with tf.name_scope('other_linear'):
    weights = tf.Variable(tf.truncated_normal([num_nodes, sigmoid_dim], stddev=1.0 / math.sqrt(float(num_nodes))), name='weights')
    biases = tf.Variable(tf.zeros([sigmoid_dim]), name='biases')
    sigmoid_logits = tf.matmul(final_hidden, weights) + biases

  return [softmax_logits, sigmoid_logits]

def build_loss_CE_independent(softmax_labels, softmax_logits, sigmoid_labels, sigmoid_logits):
  """Build the CE loss criterions, where the atmospheric conditions are conditionally independent of the other labels
  Args:
    softmax_labels: Placeholder for atmospheric condition labels
    softmax_logits: Tensor of softmax logits
    sigmoid_labels: Placeholder for other labels
    sigmoid_logits: Tensor of sigmoid logits
  Return:
    [softmax_ce, sigmoid_ce]
  """

  softmax_ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=softmax_labels, logits=softmax_logits))
  sigmoid_ce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=sigmoid_labels, logits=sigmoid_logits))

  return [softmax_ce, sigmoid_ce]

def evaluate_f2(softmax_logits, sigmoid_logits, atmos_pl, other_labels_pl):
  predictions = get_map_predictions(softmax_logits, sigmoid_logits)

def evaluate_cer(softmax_logits, sigmoid_logits, atmos_pl, other_labels_pl):
  predictions = get_map_predictions(softmax_logits, sigmoid_logits)
  other_labels_accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions[1], other_labels_pl)))

def get_map_prediction(softmax_logits, sigmoid_logits):
  """Build graph to get most likely labels, where atmostphic conditions are modelled by a softmax, and other labels are modelled by individual sigmoids
  Args:
    softmax_logits: Tensor of softmax logits
    sigmoid_logits: Tensor of sigmoid logits
  Returns:
    {1,0} vector indicating whether labels are present
  """

  atmos_prediction = tf.argmax(softmax_logits)
  other_labels_prediction = tf.round(tf.nn.sigmoid(sigmoid_logits))

  return [atmos_prediction, other_labels_prediction]

