# Convolutional neural network
# 1 [6,6,1,32] convolutional layer (stride 2)
# 1 [5,5, 32, 64] convolutional layer (stride 2)
# 1 Fully connected 200 neuron layer
# 1 softmax output layer, with 10 neurons

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

X = tf.placeholder(tf.float32, [None, 28*28])
Y_ = tf.placeholder(tf.float32, [None, 10])

W1= tf.Variable(tf.truncated_normal([6,6,1,32], stddev=0.1))
b1= tf.Variable(tf.ones([32])/10)
W2 = tf.Variable(tf.truncated_normal([5,5,32,64], stddev=0.1))
b2 = tf.Variable(tf.ones([64])/10)
W3 = tf.Variable(tf.truncated_normal([7*7*64, 200], stddev=0.1))
b3 = tf.Variable(tf.ones([200])/10)
W4 = tf.Variable(tf.truncated_normal([200,10], stddev=0.1))
b4 = tf.Variable(tf.zeros([10]))

learning_rate = tf.placeholder(tf.float32)
pkeep = tf.placeholder(tf.float32)

Y1c = tf.nn.conv2d(tf.reshape(X, [-1, 28, 28, 1]), W1, [1, 2, 2, 1], padding='SAME')
Y1 = tf.nn.relu(Y1c+b1)

Y2c = tf.nn.conv2d(Y1, W2, [1, 2, 2, 1], padding='SAME')
Y2 = tf.nn.relu(Y2c+b2)

Y3 = tf.nn.relu(tf.matmul(tf.reshape(Y2, [-1, 7*7*64]), W3)+b3)
Y3d = tf.nn.dropout(Y3, keep_prob=pkeep)

Y4_logits = tf.matmul(Y3d, W4)+b4

#cost function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y4_logits, labels=Y_))

#optimise
optimiser = tf.train.AdamOptimizer()
train = optimiser.minimize(cross_entropy)

#calculate accuracy
correct_predictions = tf.equal(tf.argmax(Y_, 1), tf.argmax(Y4_logits, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
#run computations
sess = tf.Session()

sess.run(tf.global_variables_initializer())

def getLearningRate(x, min, max):
    return min+(max-min)*np.exp(-0.006*x)

for batch_number in range(10000):
    X_batch, Y_batch = mnist.train.next_batch(50)
    training_data = {X: X_batch,
                     Y_: Y_batch,
                     learning_rate: getLearningRate(batch_number, 0.0001, 0.0009),
                     pkeep: 0.75}
    sess.run(train, feed_dict=training_data)

    if((batch_number+1) % 100 == 0): #every 100 iterations
        testing_data = {X: mnist.test.images, Y_: mnist.test.labels, pkeep: 1.0}
        acc = sess.run(accuracy, feed_dict=testing_data)
        print("Accuracy: {:.3f}".format(acc))
