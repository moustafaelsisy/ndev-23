# 3 layer deep neural network
# 1 input layer with 28*28 neurons
# 1 hidden layer with 200 neurons
# 1 softmax output layer with 10 neurons

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

X = tf.placeholder(tf.float32, [None, 28*28])
Y_ = tf.placeholder(tf.float32, [None, 10])

W1= tf.Variable(tf.truncated_normal([28*28, 200], stddev=0.1))
b1= tf.Variable(tf.zeros([200]))
W2 = tf.Variable(tf.truncated_normal([200, 10], stddev=0.1))
b2 = tf.Variable(tf.zeros([10]))

init = tf.global_variables_initializer()

Y1 = tf.nn.sigmoid(tf.matmul(X,W1)+b1)
Y2_logits = tf.matmul(Y1, W2)+b2

#cost function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y2_logits, labels=Y_))

#optimise
optimiser = tf.train.GradientDescentOptimizer(0.7)
train = optimiser.minimize(cross_entropy)

#calculate accuracy
correct_predictions = tf.equal(tf.argmax(Y_, 1), tf.argmax(Y2_logits, 1))
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
