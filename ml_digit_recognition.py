"""
handwritten_digit_recognition.py

Trains a neural network to recognize handwritten digits from the MNIST database.

Author:  Anshul Kharbanda
Created: 10 - 28 - 2017
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data as mnist_input_data
from random import randint
from tqdm import tqdm
from time import time

# -------------------------------- HYPER PARAMS --------------------------------
LEARNING_RATE = 0.2 # How quickly the network learns (sensitivity to error)
BATCH_SIZE = 500 # The number of samples in a batch in each training epoch
TRAINING_EPOCHS = 5000 # The number of training epochs
LOGPATH = 'logs'

# --------------------------------- MNIST Data ---------------------------------
# Get MNIST Data
print('Getting MNIST digit data')
mnist = mnist_input_data.read_data_sets('MNIST_data/', one_hot=True)

# --------------------------- Neural Network System ----------------------------
"""
Convolution Neural Network:

input(784)
reshape(28,28,1)
convolve(4, 4)
relu(4, 4)
reshape(49)
softmax(10)
output(10)

Determines the numerical digit that the given image represents.
"""
# Initialize random seed
tf.set_random_seed(int(time()))

# Input z
z = tf.placeholder(tf.float32, [None, 784], name='z')

# Reshape layer
s0 = tf.reshape(z, [-1, 28, 28, 1])

# Convolution layer
Kc0 = tf.Variable(tf.random_uniform([4, 4, 1, 1]), name='Kc0')
c0 = tf.nn.conv2d(s0, Kc0, strides=[1, 4, 4, 1], padding='SAME')

# RELU layer
r0 = tf.nn.relu(c0)

# Reshape layer
s1 = tf.reshape(c0, [-1, 49])

# Fully-Connected Layer to Output P
Wp = tf.Variable(tf.random_uniform([49, 10]), name='Wp')
bp = tf.Variable(tf.random_uniform([10]), name='bp')
p = tf.nn.softmax(tf.matmul(s1, Wp) + bp)

# Training p
p_ = tf.placeholder(tf.float32, [None, 10], name='p_')

# Error function (Least-Squares)
error = tf.losses.mean_squared_error(labels=p_, predictions=p)

# Trainer
trainer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(error)

# Summary
tf.summary.scalar('error', error)
tf.summary.histogram('Kc0', Kc0)
tf.summary.histogram('Wp', Wp)
tf.summary.histogram('bp', bp)
summary = tf.summary.merge_all()

# ----------------------------- Show Sample Helper -----------------------------
def show_sample(name, images, labels, predicts, error):
    """
    HELPER FUNCTION

    Shows a sample of the given MNIST data and the resulting predictions
    from the neural network. Plots images, labels, and predictions in a
    subplot with the given rows and columns. Prints the given error
    afterwards.

    :param name: the name of the dataset
    :param images: the images of the MNIST data (Kx28x28 array)
    :param labels: the labels of the MNIST data
    :param predicts: the predictions from the Nerual Network
    :param error: the error of the prediction from the Neural Network
    """
    # Title formatters
    plot_title = '{name} Sample Digits'
    subplot_title = 'Expected: {expected}, Predicted: {predicted}'
    error_title = '{name} error: {error}'

    # Rows and columns of subplot
    rows = 2
    cols = 2

    # Randomized samples start
    start = randint(0, images.shape[0] - (rows*cols))

    # Get formatted data
    formatted_images = np.reshape(images, (-1, 28, 28))
    formatted_labels = np.argmax(labels, axis=1)
    formatted_predicts = np.argmax(predicts, axis=1)

    # Create subplot plot
    plt.figure(plot_title.format(name=name))
    for index in range(rows*cols):
        # Create subplot of each sample
        splt = plt.subplot(rows, cols, index+1)
        splt.set_title(subplot_title.format(
            expected=formatted_labels[start+index],
            predicted=formatted_predicts[start+index]
        ))
        splt.axis('off')
        splt.imshow(formatted_images[start+index,:,:], cmap='gray')

    # Show plot and then print error
    plt.show()
    print(error_title.format(name=name, error=error))

# ----------------------------------- Session ----------------------------------
with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())

    # Create session writer
    writer = tf.summary.FileWriter(LOGPATH,
                                   graph=tf.get_default_graph())

    # ---------------------- Initial Run -----------------------
    # Compute sample data
    print('Compute initial prediction')
    predicts_0 = sess.run(p, feed_dict={z:mnist.test.images})
    error_0 = sess.run(error, feed_dict={z:mnist.test.images,
                                         p_:mnist.test.labels})

    # Plot initial sample
    print('Plot initial prediction sample')
    show_sample(name='Initial',
                images=mnist.test.images,
                labels=mnist.test.labels,
                predicts=predicts_0,
                error=error_0)

    # --------------------- Training Step ----------------------
    print('Training Neural Network...')
    for i in tqdm(range(TRAINING_EPOCHS), desc='Training'):
        # Train batch and add summary
        batch_zs, batch_ps = mnist.train.next_batch(BATCH_SIZE)
        _, summ = sess.run([trainer, summary],
                           feed_dict={z:batch_zs, p_:batch_ps})
        writer.add_summary(summ, i)

    # ----------------------- Final Run ------------------------
    # Get Sample Images and labels
    print('Compute final prediction')
    predicts_1 = sess.run(p, feed_dict={z:mnist.test.images})
    error_1 = sess.run(error, feed_dict={z:mnist.test.images,
                                         p_:mnist.test.labels})

    # Plot final samples
    print('Plot final prediction sample')
    show_sample(name='Final',
                images=mnist.test.images,
                labels=mnist.test.labels,
                predicts=predicts_1,
                error=error_1)

    # ---------------------- Close Writer ----------------------
    writer.close()
