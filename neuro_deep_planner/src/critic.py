import tensorflow as tf
import math
import numpy as np

# Params of fully connected layers
FULLY_LAYER1_SIZE = 200
FULLY_LAYER2_SIZE = 200

# Params of conv layers
RECEPTIVE_FIELD1 = 8
RECEPTIVE_FIELD2 = 4
RECEPTIVE_FIELD3 = 3

STRIDE1 = 4
STRIDE2 = 2
STRIDE3 = 1

FILTER1 = 32
FILTER2 = 32
FILTER3 = 32

# Other Hyperparameters
LEARNING_RATE = 0.001       # standard learning rate

REGULARIZATION_DECAY = 0.00001  # for L2 Regularization

TARGET_DECAY = 0.9999         # for target networks

FINAL_WEIGHT_INIT = 0.003   # small init weights for output layer

# For plotting
PLOT_STEP = 10


class CriticNetwork:

    def __init__(self, image_size, action_size, image_no, graph, summary_writer, session):
        self.graph = graph
        with self.graph.as_default():
            self.sess = session

            self.train_counter = 1
            self.td_error_sum = 0
            self.action_gradient_sum = np.zeros(2)

            # Define fully connected layer size
            final_conv_height = (((((image_size - RECEPTIVE_FIELD1)/STRIDE1 + 1) - RECEPTIVE_FIELD2)/STRIDE2 + 1) -
                                 RECEPTIVE_FIELD3)/STRIDE3 + 1
            self.fully_size = (final_conv_height**2) * FILTER3

            # create actor network
            self.map_input = tf.placeholder("float", [None, image_size, image_size, image_no])
            self.action_input = tf.placeholder("float", [None, action_size], name="action_input")
            self.Q_output = self.create_network(action_size, image_no)

            # get all the variables in the actor network
            with tf.variable_scope("critic") as scope:
                self.critic_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope.name)

            # Create Exponential moving Average Object
            self.ema_obj = tf.train.ExponentialMovingAverage(decay=TARGET_DECAY)

            # Create the shadow variables, and add ops to maintain moving averages
            # of critic network
            self.compute_ema = self.ema_obj.apply(self.critic_variables)

            # create target actor network
            self.map_input_target = tf.placeholder("float", [None, image_size, image_size, image_no])
            self.action_input_target = tf.placeholder("float", [None, action_size])
            self.Q_output_target = self.create_target_network(self.ema_obj)

            # L2 Regularization for all Variables
            self.regularization = 0
            for variable in self.critic_variables:
                self.regularization += tf.nn.l2_loss(variable)

            # Define training optimizer
            self.y_input = tf.placeholder("float", [None, 1], name="y_input")
            self.td_error = tf.reduce_mean(tf.pow(self.Q_output - self.y_input, 2))

            # Add regularization to loss
            self.loss = self.td_error + REGULARIZATION_DECAY * self.regularization

            self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)

            self.action_gradients = tf.gradients(self.Q_output, self.action_input)

            # summary stuff
            self.summary_writer = summary_writer

            # Variables for plotting
            self.action_grads_mean_plot = [0, 0]
            self.td_error_plot = 0

    def create_network(self, action_size, image_no):

        with tf.variable_scope('critic'):

            weights_conv1 = create_variable([RECEPTIVE_FIELD1, RECEPTIVE_FIELD1, image_no, FILTER1],
                                            RECEPTIVE_FIELD1 * RECEPTIVE_FIELD1 * image_no)
            biases_conv1 = create_variable([FILTER1], RECEPTIVE_FIELD1 * RECEPTIVE_FIELD1 * image_no)

            weights_conv2 = create_variable([RECEPTIVE_FIELD2, RECEPTIVE_FIELD2, FILTER1, FILTER2],
                                            RECEPTIVE_FIELD2 * RECEPTIVE_FIELD2 * FILTER1)
            biases_conv2 = create_variable([FILTER2], RECEPTIVE_FIELD2 * RECEPTIVE_FIELD2 * FILTER1)

            weights_conv3 = create_variable([RECEPTIVE_FIELD3, RECEPTIVE_FIELD3, FILTER2, FILTER3],
                                            RECEPTIVE_FIELD3 * RECEPTIVE_FIELD3 * FILTER2)
            biases_conv3 = create_variable([FILTER3], RECEPTIVE_FIELD3 * RECEPTIVE_FIELD3 * FILTER2)

            weights_actions = create_variable([action_size, FULLY_LAYER1_SIZE], self.fully_size)
            weights_fully1 = create_variable([self.fully_size, FULLY_LAYER1_SIZE], self.fully_size)
            biases_fully1 = create_variable([FULLY_LAYER1_SIZE], self.fully_size)

            weights_fully2 = create_variable([FULLY_LAYER1_SIZE, FULLY_LAYER2_SIZE], FULLY_LAYER1_SIZE)
            biases_fully2 = create_variable([FULLY_LAYER2_SIZE], FULLY_LAYER1_SIZE)

            weights_final = create_variable_final([FULLY_LAYER2_SIZE, 1])
            biases_final = create_variable_final([1])

        # 3 convolutional layers
        conv1 = tf.nn.relu(tf.nn.conv2d(self.map_input, weights_conv1, strides=[1, STRIDE1, STRIDE1, 1],
                                        padding='VALID') + biases_conv1)
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights_conv2, strides=[1, STRIDE2, STRIDE2, 1], padding='VALID') +
                           biases_conv2)
        conv3 = tf.nn.relu(tf.nn.conv2d(conv2, weights_conv3, strides=[1, STRIDE3, STRIDE3, 1], padding='VALID') +
                           biases_conv3)

        # reshape output tensor to a rank 1 tensor
        conv3_flat = tf.reshape(conv3, [-1, self.fully_size])

        # more operations
        fully1 = tf.nn.relu(tf.matmul(conv3_flat, weights_fully1) + tf.matmul(self.action_input, weights_actions) +
                            biases_fully1)
        fully2 = tf.nn.relu(tf.matmul(fully1, weights_fully2) + biases_fully2)
        q_output = tf.matmul(fully2, weights_final) + biases_final

        # return all ops
        return q_output

    def create_target_network(self, ema_obj):

        with tf.variable_scope('critic_target'):
            weights_conv1 = ema_obj.average(self.critic_variables[0])
            biases_conv1 = ema_obj.average(self.critic_variables[1])
            weights_conv2 = ema_obj.average(self.critic_variables[2])
            biases_conv2 = ema_obj.average(self.critic_variables[3])
            weights_conv3 = ema_obj.average(self.critic_variables[4])
            biases_conv3 = ema_obj.average(self.critic_variables[5])
            weights_actions = ema_obj.average(self.critic_variables[6])
            weights_fully1 = ema_obj.average(self.critic_variables[7])
            biases_fully1 = ema_obj.average(self.critic_variables[8])
            weights_fully2 = ema_obj.average(self.critic_variables[9])
            biases_fully2 = ema_obj.average(self.critic_variables[10])
            weights_final = ema_obj.average(self.critic_variables[11])
            biases_final = ema_obj.average(self.critic_variables[12])

        # 3 convolutional layers
        conv1 = tf.nn.relu(tf.nn.conv2d(self.map_input_target, weights_conv1, strides=[1, STRIDE1, STRIDE1, 1],
                                        padding='VALID') + biases_conv1)
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights_conv2, strides=[1, STRIDE2, STRIDE2, 1], padding='VALID') +
                           biases_conv2)
        conv3 = tf.nn.relu(tf.nn.conv2d(conv2, weights_conv3, strides=[1, STRIDE3, STRIDE3, 1], padding='VALID') +
                           biases_conv3)

        # reshape output tensor to a rank 1 tensor
        conv3_flat = tf.reshape(conv3, [-1, self.fully_size])

        # more operations
        fully1 = tf.nn.relu(tf.matmul(conv3_flat, weights_fully1) + tf.matmul(self.action_input_target,
                                                                              weights_actions) + biases_fully1)
        fully2 = tf.nn.relu(tf.matmul(fully1, weights_fully2) + biases_fully2)
        q_output = tf.matmul(fully2, weights_final) + biases_final

        # return all ops
        return q_output

    def train(self, y_batch, state_batch, action_batch):

        # run optimizer and compute some summary values
        td_error_value, _ = self.sess.run([self.td_error, self.optimizer], feed_dict={self.y_input: y_batch,
                                                                                      self.map_input: state_batch,
                                                                                      self.action_input: action_batch})

        self.update_target()

        self.td_error_plot += td_error_value

        # Only save files every 10 steps
        if (self.train_counter % PLOT_STEP) == 0:

            self.td_error_plot /= PLOT_STEP

            # Add td error to the summary writer
            summary = tf.Summary(value=[tf.Summary.Value(tag='td_error_mean',
                                                         simple_value=np.asscalar(self.td_error_plot))])
            self.summary_writer.add_summary(summary, self.train_counter)

            self.td_error_plot = 0.0

        self.train_counter += 1

    def update_target(self):

        self.sess.run(self.compute_ema)

    def get_action_gradient(self, state_batch, action_batch):

        action_gradients = self.sess.run(self.action_gradients, feed_dict={self.map_input: state_batch,
                                                                           self.action_input: action_batch})[0]

        # Create summaries for the action gradients and add them to the summary writer
        action_grads_mean = np.mean(action_gradients, axis=0)
        self.action_grads_mean_plot += action_grads_mean

        # Only save files every 10 steps
        if (self.train_counter % PLOT_STEP) == 0:

            self.action_grads_mean_plot /= PLOT_STEP

            summary_actor_grads_0 = tf.Summary(value=[tf.Summary.Value(tag='action_grads_mean[0]',
                                                                       simple_value=np.asscalar(
                                                                           self.action_grads_mean_plot[0]))])
            summary_actor_grads_1 = tf.Summary(value=[tf.Summary.Value(tag='action_grads_mean[1]',
                                                                       simple_value=np.asscalar(
                                                                           self.action_grads_mean_plot[1]))])
            self.summary_writer.add_summary(summary_actor_grads_0, self.train_counter)
            self.summary_writer.add_summary(summary_actor_grads_1, self.train_counter)

            self.action_grads_mean_plot = [0, 0]

        return action_gradients

    def evaluate(self, state_batch, action_batch):

        return self.sess.run(self.Q_output, feed_dict={self.map_input: state_batch, self.action_input: action_batch})

    def target_evaluate(self, state_batch, action_batch):

        return self.sess.run(self.Q_output_target, feed_dict={self.map_input_target: state_batch,
                                                              self.action_input_target: action_batch})


# f fan-in size
def create_variable(shape, f):
    return tf.Variable(tf.random_uniform(shape, -1/math.sqrt(f), 1/math.sqrt(f)))


def create_variable_final(shape):
    return tf.Variable(tf.random_uniform(shape, -FINAL_WEIGHT_INIT, FINAL_WEIGHT_INIT))
