import tensorflow as tf
import savevariable

sess = tf.InteractiveSession()
x = tf.Variable(tf.random_normal([5, 10], stddev=1))
y = tf.Variable(tf.random_normal([2, 1], stddev=1))
sess.run(tf.global_variables_initializer())
savevariable.SaveMany('tmp2', [x, y], ['x', 'y'], sess)
