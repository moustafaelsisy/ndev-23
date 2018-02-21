# 4 layer deep neural network
# 1 input layer with 28*28 neurons
# 1 hidden layer with 200 neurons
# 1 hidden layer with 100 neurons
# 1 softmax output layer with 10 neurons

# Modifications:
# Replaced sigmoid activation in hidden layers with relu
# Used AdamOptimizer instead of GradientDescentOptimizer to swing by saddle points

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

X = tf.placeholder(tf.float32, [None, 28*28])
Y_ = tf.placeholder(tf.float32, [None, 10])

W1= tf.Variable(tf.truncated_normal([28*28, 200], stddev=0.1))
b1= tf.Variable(tf.ones([200])/10)
W2 = tf.Variable(tf.truncated_normal([200, 100], stddev=0.1))
b2 = tf.Variable(tf.ones([100])/10)
W3 = tf.Variable(tf.truncated_normal([100, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

learning_rate = tf.placeholder(tf.float32)

Y1 = tf.nn.relu(tf.matmul(X,W1)+b1)
Y2 = tf.nn.relu(tf.matmul(Y1, W2)+b2)
Y3_logits = tf.matmul(Y2, W3)+b3

#cost function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y3_logits, labels=Y_))

#optimise
optimiser = tf.train.AdamOptimizer(learning_rate)
train = optimiser.minimize(cross_entropy)

#calculate accuracy
correct_predictions = tf.equal(tf.argmax(Y_, 1), tf.argmax(Y3_logits, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
#run computations
sess = tf.Session()

sess.run(tf.global_variables_initializer())

def getLearningRate(x, min, max):
    return min+(max-min)*np.exp(-0.001*x)

for batch_number in range(4000):
    X_batch, Y_batch = mnist.train.next_batch(100)
    training_data = {X: X_batch, Y_: Y_batch, learning_rate: getLearningRate(batch_number, 0.0005, 0.01)}
    sess.run(train, feed_dict=training_data)

    if((batch_number+1) % 100 == 0): #every 100 iterations
        testing_data = {X: mnist.test.images, Y_: mnist.test.labels}
        acc = sess.run(accuracy, feed_dict=testing_data)
        print("Accuracy: {:.3f}".format(acc))
