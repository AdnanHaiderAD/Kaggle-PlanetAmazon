import tensorflow as tf
import build_graph
import math
import minibatch

reader = minibatch.MiniBatch('train.scp', 'train.csv')

sess = tf.InteractiveSession()

# initialise placeholders
r_pl = tf.placeholder(tf.float32, shape=[None, 65536])
g_pl = tf.placeholder(tf.float32, shape=[None, 65536])
b_pl = tf.placeholder(tf.float32, shape=[None, 65536])
i_pl = tf.placeholder(tf.float32, shape=[None, 65536])

atmos_pl = tf.placeholder(tf.float32, shape=[None, 4])
other_labels_pl = tf.placeholder(tf.float32, shape=[None, 13])

fd = reader.FillFeedDict(5, r_pl, g_pl, b_pl, i_pl, atmos_pl, other_labels_pl)

# build graph
dnn = build_graph.build_dnn_graph(r_pl, g_pl, b_pl, i_pl, 2, 100, 65536, 'sigmoid')
logits = build_graph.build_output_independent(dnn, 100, 4, 13)
ce = build_graph.build_loss_CE_independent(atmos_pl, logits[0], other_labels_pl, logits[1])

sess.run(tf.global_variables_initializer())

train = tf.train.GradientDescentOptimizer(0.5).minimize(ce[0] + ce[1])
train.run(feed_dict=fd)


#placeholder = tf.placeholder(tf.float32)
#x = build_graph.build_dnn_graph(placeholder, 4, 100, 10, 'relu')
#y = build_graph.build_output_independent(x, 100, 4, 13)
#l1 = tf.placeholder(tf.float32)
#l2 = tf.placeholder(tf.float32)
#z = build_graph.build_loss_CE_independent(l1, y[0], l2, y[1])
#joint_loss = z[0] + z[1]
