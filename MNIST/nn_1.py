#2 layer (input and output) neural network

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

X = tf.placeholder(tf.float32, [None, 28*28])
Y_ = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([28*28, 10]))
b = tf.Variable(tf.truncated_normal([10], stddev=0.5))

init = tf.global_variables_initializer()

Y_logits = tf.matmul(X, W)+b

#cost function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_logits, labels=Y_))

#optimise
optimiser = tf.train.GradientDescentOptimizer(0.3)
train = optimiser.minimize(cross_entropy)

#calculate accuracy
correct_predictions = tf.equal(tf.argmax(Y_, 1), tf.argmax(Y_logits, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

#run computations
sess = tf.Session()

sess.run(init)

for batch_number in range(4000):
    X_batch, Y_batch = mnist.train.next_batch(100)
    training_data = {X: X_batch, Y_: Y_batch}
    sess.run(train, feed_dict=training_data)

    if((batch_number+1) % 100 == 0): #every 100 iterations
        testing_data = {X: mnist.test.images, Y_: mnist.test.labels}
        acc = sess.run(accuracy, feed_dict=testing_data)
        print("Accuracy: {:.3f}".format(acc))
