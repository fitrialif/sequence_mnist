import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np


# load the data
mnist_data_dir = './data/'
mnist = input_data.read_data_sets(
    mnist_data_dir, 
    one_hot=True
)

# Define some parameters
element_size = 28
time_steps = 28
num_classes = 10
batch_size = 128
hidden_layer_size = 128

# model summary save dir
LOG_DIR = "logs/RNN_with_summaries"


_input = tf.placeholder(
    tf.float32, 
    shape=[
        None, 
        time_steps, 
        element_size
    ], 
    name='inputs'
)

y = tf.placeholder(
    tf.float32,
    shape=[
        None,
        num_classes
    ],
    name='labels'
)


# this helper function simply adds some ops
# that take care of logging summaries
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(
                var - mean
            )))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


with tf.name_scope('rnn_weights'):
    with tf.name_scope('W_x'):
        W_x = tf.Variable(tf.zeros(
            [element_size, hidden_layer_size]
        ))
        variable_summaries(W_x)
    with tf.name_scope('W_h'):
        W_h = tf.Variable(tf.zeros(
            [hidden_layer_size, hidden_layer_size]
        ))
        variable_summaries(W_h)
    with tf.name_scope('Bias'):
        b_rnn = tf.Variable(tf.zeros(
            [hidden_layer_size]
        ))
        variable_summaries(b_rnn)


def rnn_step(previous_hidden_state, x):

    current_hidden_state = tf.tanh(
        tf.matmul(previous_hidden_state, W_h) +
        tf.matmul(x, W_x) +
        b_rnn
    )

    return current_hidden_state


processed_input = tf.transpose(_input, perm=[1, 0, 2])
# current input shape is now: 
# (time_steps, batch_size, element_size)

initial_hidden_state = tf.zeros(
    [batch_size, hidden_layer_size]
)

all_hidden_states = tf.scan(
    rnn_step, 
    processed_input, 
    initializer=initial_hidden_state,
    name='states'
)


# weights for output layers
with tf.name_scope('linear_layer_weights') as scope:
    with tf.name_scope('W_linear'):
        Wl = tf.Variable(tf.truncated_normal(
            [hidden_layer_size, num_classes],
            mean=0,
            stddev=0.01
        ))
        variable_summaries(Wl)
    with tf.name_scope('Bias_linear'):
        bl = tf.Variable(tf.truncated_normal(
            [num_classes],
            mean=0,
            stddev=0.01
        ))
        variable_summaries(bl)


# apply linear layer to the state vector
def get_linear_layer(hidden_state):
    return tf.matmul(hidden_state, Wl) + bl


with tf.name_scope('linear_layer_weights') as scope:
    # iterate across time, apply linear layer to all
    # RNN outputs
    all_outputs = tf.map_fn(
        get_linear_layer, 
        all_hidden_states
    )

    # get the last output
    output = all_outputs[-1]
    tf.summary.histogram('outputs', output)


with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=output, 
            labels=y
        )
    )

    tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.RMSPropOptimizer(
        1e-3, 
        0.9
    ).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    # tf.equal returns True if the args are equal
    # otherwise returns False
    correct_predictions = tf.equal(
        tf.argmax(y, 1),
        tf.argmax(output, 1)
    )
    accuracy = tf.reduce_mean(
        tf.cast(correct_predictions, tf.float32)
    ) * 100

    tf.summary.scalar('accuracy', accuracy)

# Merge all the summaries
merged = tf.summary.merge_all()

test_data = mnist.test.images[:batch_size].reshape((
    -1, 
    time_steps, 
    element_size
))

test_label = mnist.test.labels[:batch_size]

with tf.Session() as sess:
    # write summaries to LOG_DIR -- used by TENSORBOARD
    train_writer = tf.summary.FileWriter(
        LOG_DIR + '/train',
        graph=tf.get_default_graph()
    )

    test_writer = tf.summary.FileWriter(
        LOG_DIR + '/test',
        graph=tf.get_default_graph()
    )

    sess.run(tf.global_variables_initializer())

    for i in range(10000):

        # data comes as an unrolled vector, so batch_x: 
        # shape: (batch_size, 786)
        batch_x, batch_y = mnist.train.next_batch(batch_size)

        batch_x = batch_x.reshape(
            (batch_size, time_steps, element_size)
        )

        summary, _ = sess.run(
            [merged, train_step],
            feed_dict={_input:batch_x, y:batch_y}
        )

        # add to summaries
        train_writer.add_summary(summary, i)

        # TODO: why need to feed_dict x,y two times
        if i % 1000 == 0:
            acc, loss = sess.run(
                [accuracy, cross_entropy],
                feed_dict={_input:batch_x, y:batch_y}
            )

            print('''
                Iter: {}, 
                Minibatch loss: {:.6f}, 
                Training Accuracy: {:.5f}
                '''.format(i, loss, acc)
            )

        if i % 10:
            # calculate accuracy for 128 MNIST test-
            # images and add to summaries
            summary, acc = sess.run(
                [merged, accuracy],
                feed_dict={_input:test_data, y: test_label}
            )

            test_writer.add_summary(summary, i)

    test_acc = sess.run(
        accuracy, 
        feed_dict={_input:test_data, y: test_label}
    )

    print('Test Accuracy: {}'.format(test_acc))
