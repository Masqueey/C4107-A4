# Dave MacDonald  100909202
# Max Leijtens 101093543
# COMP 4107 Assignment 4
# December 6th, 2017

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 128
test_size = 256


def makeOneHot(Y, depth=10):
    for y in Y:
        y = tf.one_hot(y, 10)
    return Y
               

def init_weights(name,shape):                                                       # Modified to take name (necessary for tf.get_variable)
    return tf.get_variable(name, shape, initializer=tf.glorot_normal_initializer()) # Use Glorot normal initialization. tf.get_variable allows us to specify shape and initializer.
#    return tf.Variable(tf.random_normal(shape, stddev=0.01))


# Added act argument for activation function, and changed the implementation to use the passed function.
def model(X, w, w1, w2, w_fc, w_o, p_keep_conv, p_keep_hidden, act):                            # 

    l1a = act(tf.nn.conv2d(X, w,                       # l1a shape=(?, 32, 32, 16)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 16, 16, 16)
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)


    l2a = act(tf.nn.conv2d(l1, w1,                       # l2a shape=(?, 16, 16, 20)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],              # l2 shape=(?, 8, 8, 20)
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)


    l3a = act(tf.nn.conv2d(l2, w2,                       # l2a shape=(?, 8, 8, 20)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],              # l2 shape=(?, 4, 4, 20)
                        strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.nn.dropout(l3, p_keep_conv)


    l4 = tf.reshape(l3, [-1, w_fc.get_shape().as_list()[0]])    # reshape to (?, 4x4x20)  #Modified to take l2 instead of l1
    l4 = tf.nn.dropout(l4, p_keep_conv)

    l5 = act(tf.matmul(l4, w_fc))
    l5 = tf.nn.dropout(l5, p_keep_hidden)

    pyx = tf.matmul(l5, w_o)
    return pyx

cifar = tf.keras.datasets.cifar10.load_data()
train, test = cifar
trX, trY = train
teX, teY = test
trX = trX.reshape(-1, 32, 32, 3)  # 32x32x3 input img
teX = teX.reshape(-1, 32, 32, 3)  # 32x32x3 input img

X = tf.placeholder("float", [None, 32, 32, 3], name="X")
Y = tf.placeholder("float", [None,10], name="Y")

# Modified to add variable names to match new interface of init_weights()
w = init_weights('w',[5, 5, 3, 16])       # 5x5x3 conv, 16 outputs
w1 = init_weights('w1',[5, 5, 16, 20])       # 5x5x3 conv, 20 outputs
w2 = init_weights('w2',[5, 5, 20, 20])
w_fc = init_weights('w_fc', [20 * 4 * 4, 125]) # FC 20 * 4 * 4 inputs, 625 outputs    # Modified dimension to match w2 output 64*7*7
w_o = init_weights('w_o',[125, 10])         # FC 625 inputs, 10 outputs (labels)

p_keep_conv = tf.placeholder("float", name="p_keep_conv")
p_keep_hidden = tf.placeholder("float", name="p_keep_hidden")
py_x = model(X, w, w1, w2, w_fc, w_o, p_keep_conv, p_keep_hidden, tf.nn.relu)       # Modified to pass the relu activation function.    #Modified to take w1

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()
    writer = tf.summary.FileWriter("cifar/graph",graph=sess.graph) # Added to log graph to file for visualization with TensorBoard.
    saver = tf.train.Saver()

    for i in range(5):      # Changed to 5 epochs, from 100.
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX)+1, batch_size))
        for start, end in training_batch:
#            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
#                                          p_keep_conv: 0.8, p_keep_hidden: 0.5})
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_conv: 1, p_keep_hidden: 1})    # Dropouts to 1 to "turn off" drop out

        test_indices = np.arange(len(teX)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]
        
        print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                         p_keep_conv: 1.0,
                                                         p_keep_hidden: 1.0})))
        saver.save(sess, "cifar/session.ckpt")
    writer.close()  # Added to log graph to file for visualization with TensorBoard.
