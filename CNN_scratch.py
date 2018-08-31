from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import random
import datetime as dt


import argparse
import sys

timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None
#hyper parameters




tf.set_random_seed(169) #for reproduciblity

def main(_):
	if tf.gfile.Exists(FLAGS.log_dir):
		tf.gfile.DeleteRecursively(FLAGS.log_dir)
	tf.gfile.MakeDirs(FLAGS.log_dir)
	train()

def train():
	#Import data
	mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot = True, fake_data = FLAGS.fake_data)

	sess = tf.InteractiveSession()

	with tf.name_scope("input"):
		x = tf.placeholder(tf.float32, [None, 784], name = "x-input")
		y_ = tf.placeholder(tf.float32, [None, 10], name = "y-input")
		keep_prob = tf.placeholder(tf.float32)

	with tf.name_scope("input_reshape"):
		x_img = tf.reshape(x, [-1, 28, 28, 1])
		tf.summary.image("input", x_img, 10)

	def variable_summaries(var):
		with tf.name_scope("summaries"):
			mean = tf.reduce_mean(var)
			tf.summary.scalar("mean", mean)
			with tf.name_scope("stddev"):
				stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
			tf.summary.scalar("stddev", stddev)
			tf.summary.scalar('max', tf.reduce_max(var))
			tf.summary.scalar('min', tf.reduce_min(var))
			tf.summary.histogram('histogram', var)

	def convolution_layer(input, layer_num, weight_shape, conv_strides, pool_size, pool_strides, last = False):
		with tf.variable_scope("layer" + str(layer_num)):
			with tf.name_scope("weights"):
				w = tf.get_variable("w", shape = weight_shape, initializer = tf.contrib.layers.xavier_initializer())
				variable_summaries(w)
			with tf.name_scope("conv2d"):
				conv_filter = tf.nn.conv2d(input, w, strides = conv_strides, padding = "SAME")
				tf.summary.histogram("conv_filter", conv_filter)
			relu_filter = tf.nn.relu(conv_filter)
			tf.summary.histogram("relu_activation", relu_filter)
			with tf.name_scope("max_pool"):
				max_pool_filter = tf.nn.max_pool(relu_filter, ksize = pool_size, strides = pool_strides, padding = "SAME")
				tf.summary.histogram("max_pool", max_pool_filter)
			L = tf.nn.dropout(max_pool_filter, keep_prob = keep_prob)
			if last == True:
				length = len(L.get_shape().as_list())
				L_shape = 1
				for i in range(1, length):
					L_shape = L_shape * int(L.get_shape()[i])
				L = tf.reshape(L, [-1, L_shape])
			return L

	def soft_layer(input, layer_num, L_shape, act = tf.nn.relu):
		with tf.variable_scope("layer"+str(layer_num)):
			with tf.name_scope("weights"):
				w = tf.get_variable("w", shape = [int(input.get_shape()[1]), L_shape], initializer = tf.contrib.layers.xavier_initializer())
				variable_summaries(w)
			with tf.name_scope("biases"):
				b = tf.get_variable("b", shape = [L_shape], initializer = tf.contrib.layers.xavier_initializer())
				variable_summaries(b)
			with tf.name_scope("Wx_plus_b"):
				preactivate = tf.add(tf.matmul(input, w), b)
				tf.summary.histogram("pre_activations", preactivate)
			activations = act(preactivate, name = "activation")
			tf.summary.histogram("activation", activations)
			L = tf.nn.dropout(activations, keep_prob = keep_prob)
			return L

	def output_layer(input, output_shape):
		with tf.variable_scope("output"):
			with tf.name_scope("weights"):
				w = tf.get_variable("w", shape = [int(input.get_shape()[1]), output_shape], initializer = tf.contrib.layers.xavier_initializer())
				variable_summaries(w)
			with tf.name_scope("biases"):
				b = tf.get_variable("b", shape = [output_shape], initializer = tf.contrib.layers.xavier_initializer())
				variable_summaries(b)
			with tf.name_scope("logits"):
				logits = tf.add(tf.matmul(input, w), b)
				tf.summary.histogram("logits", logits)
			return logits


	#layer design
	# 4 convolution layers with each 32, 64, 128, 128 depth
	# 1 softmax layer with 625 args

	layer_design = [32, 64, 128, 128, 625]
	hidden_filter = x_img
	for i in range(len(layer_design)):
		if i == 0:
			hidden_filter = convolution_layer(hidden_filter, layer_num = i, weight_shape = [3, 3, 1, layer_design[i]], conv_strides = [1, 1, 1, 1], pool_size = [1,2,2,1], pool_strides = [1,2,2,1])
		elif i <= len(layer_design)-3:
			hidden_filter = convolution_layer(hidden_filter, layer_num = i, weight_shape = [3, 3, layer_design[i-1], layer_design[i]], conv_strides = [1, 1, 1, 1], pool_size = [1,2,2,1], pool_strides = [1,2,2,1])
		elif i == len(layer_design)-2:
			hidden_filter = convolution_layer(hidden_filter, layer_num = i, weight_shape = [3, 3, layer_design[i-1], layer_design[i]], conv_strides = [1, 1, 1, 1], pool_size = [1,2,2,1], pool_strides = [1,2,2,1], last = True)
		elif i == len(layer_design)-1:
			hidden_filter = soft_layer(hidden_filter, layer_num = i, L_shape = layer_design[i])

	logits = output_layer(hidden_filter, 10)

	with tf.name_scope("Cost"):
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y_))
		#tf.scalar_summary(cost.op.name, cost)
		tf.summary.scalar("Cost", cost)

	global_step = tf.Variable(0, name='global_step', trainable=False)

	with tf.name_scope("Train"):
		optimizer = tf.train.AdamOptimizer(learning_rate= FLAGS.learning_rate)
		train_op = optimizer.minimize(cost, global_step = global_step)

	with tf.name_scope("Accuracy"):
		with tf.name_scope("correct_prediction"):
			correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
		with tf.name_scope("accuracy"):
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar("Accuracy", accuracy)



	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter(FLAGS.log_dir + "/train" + timestamp, sess.graph)
	test_writer = tf.summary.FileWriter(FLAGS.log_dir + "/test", sess.graph)

	# Call this after declaring all tf.Variables.
	saver = tf.train.Saver()
	tf.global_variables_initializer().run()

	ckpt = tf.train.get_checkpoint_state(FLAGS.log_dir)
    
	if ckpt and ckpt.model_checkpoint_path:
		#print(ckpt.model_checkpoint_path)
		saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables

	start = global_step.eval() # get last global_step
	print("Start from:", start)

	def feed_dict(train):
		if train or FLAGS.fake_data:
			xs, ys = mnist.train.next_batch(FLAGS.batch_size, fake_data = FLAGS.fake_data)
			k = FLAGS.dropout
		else:
			xs, ys = mnist.test.images, mnist.test.labels
			k = 1.0
		return {x:xs, y_:ys, keep_prob: k}

	for epoch in range(start, FLAGS.training_epochs):
		avg_cost = 0
		total_batch = int(mnist.train.num_examples / FLAGS.batch_size)

		for i in range(total_batch):
			if i % 10 == 0:
				c, summary, acc = sess.run([cost, merged, accuracy], feed_dict = feed_dict(False))
				avg_cost += c / total_batch
				test_writer.add_summary(summary, i)
			else:
				if i % 100 == 99:
					run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
					run_metadata = tf.RunMetadata()
					summary, _ = sess.run([merged, train_op],
			                              feed_dict=feed_dict(True),
			                              options=run_options,
			                              run_metadata=run_metadata)
					train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
					train_writer.add_summary(summary, i)
					print('Adding run metadata for', i)
				else:  # Record a summary
					summary, _ = sess.run([merged, train_op], feed_dict=feed_dict(True))
					train_writer.add_summary(summary, i)

		print("Accuracy at step %s: %s" % (i, acc))
		print("Cost at step %s: %s" % (i, avg_cost))

		global_step.assign(i).eval()
		saver.save(sess, FLAGS.data_dir + "/model.ckpt", global_step = global_step)
		
	train_writer.close()
	test_writer.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--fake_data', nargs='?', const=True, type=bool, default=False,help='If true, uses fake data for unit testing.')
	parser.add_argument('--max_steps', type=int, default=1000,help='Number of steps to run trainer.')
	parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
	parser.add_argument('--dropout', type=float, default=0.7,
                      help='Keep probability for training dropout.')
	parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
	parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/mnist/logs/mnist_with_summaries',
                      help='Summaries log directory')
	parser.add_argument('--training_epochs', type = int, default = 15, help = 'Number of epochs')
	parser.add_argument('--batch_size', type = int, default = 100)
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


