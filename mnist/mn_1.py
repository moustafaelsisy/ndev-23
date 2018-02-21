import tensorflow as tf
from tensorflow.exammples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_DATA', one_hot=True)

# setting up computation graph
X = tf.placeholder(tf.float32, [None, 28*28])
Y_ = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros[28*28, 10])	# weights
b = tf.Variable(tf.truncated_normal([10], stddev=0.5))	# biases

Y_logits = tf.matmul(X,W)+b # matrix multiplication of X and W plus biases

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Y_logits))

# optimiser
optimiser = tf.train.GradientDescentOptimizer(0.5)
training_step = optimiser.minimise(cross_entropy)

correct_predictions = tf.equal(tf.argmax(Y_, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_min(tf.cast(correct_predictions, tf.float32))

# session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for batch_number in range(4000):
	X_batch, Y_batch = mnist.train.next_batch(100)
	sess.run(training_step, feed_dict={X: X_batch, Y_: Y_batch})

	if((batch_number+1) % 100 == 0):
		feeding_dict = {X: mnist.test.images, Y: minst.test.labels}
		acc = sess.run(accuracy, feed_dict=feeding_dict)
		print("Accuracy: {:.2f}".format(acc))
