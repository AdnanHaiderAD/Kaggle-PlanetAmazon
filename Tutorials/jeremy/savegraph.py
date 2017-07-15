# Save and load tensorflow computation graphs
#
# e.g.
# Save:
#
# import tensorflow as tf
# import savegraph
# 
# weights1 = tf.Variable(tf.truncated_normal([10,10], stddev=1.0))
# output = tf.nn.sigmoid(weights1)
# tf.add_to_collection('w1', weights1)
# tf.add_to_collection('out', output)
#
# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
#
# savegraph.Save('file', sess)
#
# =====================================================================
# Load:
#
# import tensorflow as tf
# import savegraph
#
# sess = tf.InteractiveSession()
#
# savegraph.Load('file', sess)
#
# weights1 = tf.get_collection('w1')[0]
# output = tf.get_collection('out')[0]

import tensorflow as tf

def Save(filename, sess):
  saver = tf.train.Saver()
  saver.save(sess, filename)
  saver.export_meta_graph(filename=filename+'.meta')

def Load(filename, sess):
  saver = tf.train.import_meta_graph(filename+'.meta')
  saver.restore(sess, filename)

