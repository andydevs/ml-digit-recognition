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

# -------------------------------- MNIST Data ----------------------------------
# Get MNIST Data
print('Getting MNIST digit data')
mnist = mnist_input_data.read_data_sets('MNIST_data/', one_hot=True)

# --------------------------- Neural Network System ----------------------------
"""
Single Layer Neural Network: 784 -> 10

Determines the numerical digit that the given image represents.
"""
# Input z
z = tf.placeholder(tf.float32, [None, 784])

# Weights and biases of layer
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Output p
p = tf.nn.softmax(tf.matmul(z, W) + b)

# Training p
p_ = tf.placeholder(tf.float32, [None, 10])

# Error function (Least-Squares)
J = tf.reduce_mean(tf.reduce_sum(tf.square(p_ - p), reduction_indices=[1]))

# Trainer
trainer = tf.train.GradientDescentOptimizer(0.5).minimize(J)

# ----------------------------- Show Sample Helper -----------------------------
def show_sample(name, images, labels, predicts, error, rows=2, cols=2, start=0):
    """
    HELPER FUNCTION

    Shows a sample of the given MNIST data and the resulting predictions from
    the neural network. Plots images, labels, and predictions in a subplot with
    the given rows and columns. Prints the given error afterwards.

    :param name: the name of the dataset
    :param images: the images of the MNIST data
    :param labels: the labels of the MNIST data
    :param predicts: the predictions from the Nerual Network
    :param error: the error of the prediction from the Neural Network
    :param rows: the number of rows in the subplot grid (default=2)
    :param cols: the number of columns in the subplot grid (default=2)
    :param start: the start of the data to sample (default=0)
    """
    simages = np.reshape(images, (-1, 28, 28))
    slabels = np.argmax(labels, axis=1)
    spredicts = np.argmax(predicts, axis=1)
    fig = plt.figure('Sample Digits')
    for index in range(rows*cols):
        splt = plt.subplot(rows, cols, index+1)
        splt.set_title('Expected: {expected}, Predicted: {predicted}'.format(
            expected=slabels[start+index],
            predicted=spredicts[start+index]
        ))
        splt.axis('off')
        splt.imshow(simages[start+index,:,:], cmap='gray')
    plt.show()
    print('{name} error: {error}'.format(name=name, error=error))

# ----------------------------------- Session ----------------------------------
with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())

    # Session params
    nmnist = mnist.test.images.shape[0]
    nbatch = 100
    epochs = 1000
    srows = 2
    scols = 2
    sstart = randint(0, nmnist-(srows*scols))

    # -------------------- Initial Run ---------------------
    # Compute sample data
    print('Compute initial prediction')
    predicts_0 = sess.run(p,
        feed_dict={
            z:mnist.test.images})
    error_0 = sess.run(J,
        feed_dict={
            z:mnist.test.images,
            p_:mnist.test.labels})

    # Plot initial sample
    print('Plot initial prediction sample')
    show_sample(
        name='Initial',
        images=mnist.test.images,
        labels=mnist.test.labels,
        predicts=predicts_0,
        error=error_0,
        rows=srows,
        cols=scols,
        start=sstart)

    # ------------------- Training Step --------------------
    print('Training...')
    for _ in range(epochs):
        batch_zs, batch_ps = mnist.train.next_batch(nbatch)
        sess.run(trainer,
            feed_dict={
                z:batch_zs,
                p_:batch_ps})

    # --------------------- Final Run ----------------------
    # Get Sample Images and labels
    print('Compute final prediction')
    predicts_1 = sess.run(p,
        feed_dict={
            z:mnist.test.images})
    error_1 = sess.run(J,
        feed_dict={
            z:mnist.test.images,
            p_:mnist.test.labels})

    # Plot final samples
    print('Plot final prediction sample')
    show_sample(
        name='Final',
        images=mnist.test.images,
        labels=mnist.test.labels,
        predicts=predicts_1,
        error=error_1,
        rows=srows,
        cols=scols,
        start=sstart)
