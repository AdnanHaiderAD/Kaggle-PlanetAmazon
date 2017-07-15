import sys
sys.path.append("/home/mifs/jhmw2/kaggle/amazon/Kaggle-PlanetAmazon-Adnan/Tutorials/MNIST/src")
import tensorflow as tf
import ConvLib # import lib to create flexible convNets
import savevariable
sess = tf.InteractiveSession()

classes = 17
height = 256
width = 256
channels = 4
image_pl = tf.placeholder(tf.float32, shape=[None,height,width,channels])

layerWeights = {1: [5,5,4,32],2: [5,5,32,64]}
poolingWindows ={1: [[1,2,2,1],[1,2,2,1]],2 :[[1,2,2,1],[1,2,2,1]]}

convNet = ConvLib.constructConvNet(image_pl,layerWeights,poolingWindows)
denseLayer = ConvLib.createFullConnectedDenseLayer(convNet, [7 * 7 * 64, 1024])
tf.add_to_collection('convNet', convNet)
tf.add_to_collection('denseLayer', denseLayer)

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.save(sess, 'tmp')
saver.export_meta_graph(filename='tmp.meta')

#savevariable.Save('tmp', denseLayer, 'denseLayer', sess)

