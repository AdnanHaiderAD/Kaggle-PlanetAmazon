# Save and load tensorflow variables, including tensors
#
# Save example:
#
# import savevariable
# import tensorflow as tf
#
# sess = tf.InteractiveSession()
# x = tf.Variable(tf.random_normal([5, 10], stddev=1))
# y = tf.Variable(tf.random_normal([2, 1], stddev=1))
# sess.run(tf.global_variables_initializer())
# savevariable.Save('save_file', [x, y], ['x', 'y'], sess)
#
# ================================================================
# Load example:
#
# import savevariable
# import tensorflow as tf
#
# sess = tf.InteractiveSession()
# x = tf.Variable(tf.random_normal([5, 10], stddev=1))
# y = tf.Variable(tf.random_normal([2, 1], stddev=1))
# savevariable.Load('save_file', x, 'x', sess)
# savevariable.Load('save_file', y, 'y', sess)


import tensorflow as tf

# Save a tensorflow variable.
# Usage: savevariable.Save('save_file', variable, variable_name, session)
def Save (filename, variable, variablename, sess):
  saver = tf.train.Saver({variablename: variable})
  saver.save(sess, filename)

# Save a list of tensorflow variables.
# Usage: savevariable.SaveMany('save_file', [list_of_variables], [list_of_variable_names], session)
def SaveMany (filename, variables, variablenames, sess):
  name_to_variable_dict = {}
  assert len(variables) == len(variablenames)
  for i in range(0, len(variables)):
    name_to_variable_dict[variablenames[i]] = variables[i]

  saver = tf.train.Saver(name_to_variable_dict)
  saver.save(sess, filename)

# Save a tensorflow variable.
# Usage: savevariable.Load('save_file', variable, variable_name, session)
def Load (filename, variable, variablename, sess):
  saver = tf.train.Saver({variablename: variable})
  saver.restore(sess, filename)

