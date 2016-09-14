import tensorflow as tf
import numpy as np
from critic import create_variable
from critic import create_variable_final

# Params of fully connected layers
FULLY_LAYER1_SIZE = 200
FULLY_LAYER2_SIZE = 200

# Params of conv layers
RECEPTIVE_FIELD1 = 4
RECEPTIVE_FIELD2 = 4
RECEPTIVE_FIELD3 = 4
# RECEPTIVE_FIELD4 = 3

STRIDE1 = 2
STRIDE2 = 2
STRIDE3 = 2
# STRIDE4 = 1

FILTER1 = 32
FILTER2 = 32
FILTER3 = 32
# FILTER4 = 64

# How fast is learning
LEARNING_RATE = 0.0005

# How fast does the target net track
TARGET_DECAY = 0.9999

# How often do we plot variables during training
PLOT_STEP = 10


class ActorNetwork:

    def __init__(self, image_size, action_size, image_no, session, summary_writer):

        self.graph = session.graph

        with self.graph.as_default():

            # Get session and summary writer from ddpg
            self.sess = session
            self.summary_writer = summary_writer

            # Get input dimensions from ddpg
            self.image_size = image_size
            self.action_size = action_size
            self.image_no = image_no

            # Calculate the fully connected layer size
            height_layer1 = (image_size - RECEPTIVE_FIELD1)/STRIDE1 + 1
            height_layer2 = (height_layer1 - RECEPTIVE_FIELD2)/STRIDE2 + 1
            height_layer3 = (height_layer2 - RECEPTIVE_FIELD3)/STRIDE3 + 1
            # height_layer4 = (height_layer3 - RECEPTIVE_FIELD4)/STRIDE4 + 1
            # self.fully_size = (height_layer4**2) * FILTER4
            self.fully_size = (height_layer3**2) * FILTER3

            # Create actor network
            self.map_input = tf.placeholder("float", [None, self.image_size, self.image_size, self.image_no])
            self.action_output = self.create_network()

            # Get all the variables in the actor network for exponential moving average, create ema op
            with tf.variable_scope("actor") as scope:
                self.actor_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope.name)
            self.ema_obj = tf.train.ExponentialMovingAverage(decay=TARGET_DECAY)
            self.compute_ema = self.ema_obj.apply(self.actor_variables)

            # Create target actor network
            self.map_input_target = tf.placeholder("float", [None, self.image_size, self.image_size, self.image_no])
            self.action_output_target = self.create_target_network()

            # Define the gradient operation that delivers the gradients with the action gradient from the critic
            self.q_gradient_input = tf.placeholder("float", [None, action_size])
            self.parameters_gradients = tf.gradients(self.action_output, self.actor_variables, -self.q_gradient_input)

            # Define the optimizer
            self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients,
                                                                                       self.actor_variables))

            # Variables for plotting
            self.actions_mean_plot = [0, 0]
            self.target_actions_mean_plot = [0, 0]

            self.train_counter = 0

    def create_network(self):

        with tf.variable_scope('actor'):

            weights_conv1 = create_variable([RECEPTIVE_FIELD1, RECEPTIVE_FIELD1, self.image_no, FILTER1],
                                            RECEPTIVE_FIELD1 * RECEPTIVE_FIELD1 * self.image_no, "weights_conv1")
            biases_conv1 = create_variable([FILTER1], RECEPTIVE_FIELD1 * RECEPTIVE_FIELD1 * self.image_no,
                                           "biases_conv1")

            weights_conv2 = create_variable([RECEPTIVE_FIELD2, RECEPTIVE_FIELD2, FILTER1, FILTER2],
                                            RECEPTIVE_FIELD2 * RECEPTIVE_FIELD2 * FILTER1, "weights_conv2")
            biases_conv2 = create_variable([FILTER2], RECEPTIVE_FIELD2 * RECEPTIVE_FIELD2 * FILTER1, "biases_conv2")

            weights_conv3 = create_variable([RECEPTIVE_FIELD3, RECEPTIVE_FIELD3, FILTER2, FILTER3],
                                            RECEPTIVE_FIELD3 * RECEPTIVE_FIELD3 * FILTER2, "weights_conv3")
            biases_conv3 = create_variable([FILTER3], RECEPTIVE_FIELD3 * RECEPTIVE_FIELD3 * FILTER2, "biases_conv3")

            # weights_conv4 = create_variable([RECEPTIVE_FIELD4, RECEPTIVE_FIELD4, FILTER3, FILTER4],
            #                                 RECEPTIVE_FIELD4 * RECEPTIVE_FIELD4 * FILTER3, "weights_conv4")
            # biases_conv4 = create_variable([FILTER4], RECEPTIVE_FIELD4 * RECEPTIVE_FIELD4 * FILTER3, "biases_conv4")

            weights_fully1 = create_variable([self.fully_size, FULLY_LAYER1_SIZE], self.fully_size)
            biases_fully1 = create_variable([FULLY_LAYER1_SIZE], self.fully_size)

            weights_fully2 = create_variable([FULLY_LAYER1_SIZE, FULLY_LAYER2_SIZE], FULLY_LAYER1_SIZE)
            biases_fully2 = create_variable([FULLY_LAYER2_SIZE], FULLY_LAYER1_SIZE)

            weights_final = create_variable_final([FULLY_LAYER2_SIZE, self.action_size])
            biases_final = create_variable_final([self.action_size])

        # 4 Convolutional layers
        conv1 = tf.nn.relu(tf.nn.conv2d(self.map_input, weights_conv1, strides=[1, STRIDE1, STRIDE1, 1],
                                        padding='VALID') + biases_conv1)

        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights_conv2, strides=[1, STRIDE2, STRIDE2, 1], padding='VALID') +
                           biases_conv2)

        conv3 = tf.nn.relu(tf.nn.conv2d(conv2, weights_conv3, strides=[1, STRIDE3, STRIDE3, 1], padding='VALID') +
                           biases_conv3)

        # conv4 = tf.nn.relu(tf.nn.conv2d(conv3, weights_conv4, strides=[1, STRIDE4, STRIDE4, 1], padding='VALID') +
        #                    biases_conv4)

        # Reshape output tensor to a rank 1 tensor
        # conv_flat = tf.reshape(conv4, [-1, self.fully_size])
        conv_flat = tf.reshape(conv3, [-1, self.fully_size])

        # 2 Fully connected layers
        fully1 = tf.nn.relu(tf.matmul(conv_flat, weights_fully1) + biases_fully1)
        fully2 = tf.nn.relu(tf.matmul(fully1, weights_fully2) + biases_fully2)

        return tf.matmul(fully2, weights_final) + biases_final

    def create_target_network(self):

        weights_conv1 = self.ema_obj.average(self.actor_variables[0])
        biases_conv1 = self.ema_obj.average(self.actor_variables[1])
        weights_conv2 = self.ema_obj.average(self.actor_variables[2])
        biases_conv2 = self.ema_obj.average(self.actor_variables[3])
        weights_conv3 = self.ema_obj.average(self.actor_variables[4])
        biases_conv3 = self.ema_obj.average(self.actor_variables[5])
        # weights_conv4 = self.ema_obj.average(self.actor_variables[6])
        # biases_conv4 = self.ema_obj.average(self.actor_variables[7])
        # weights_fully1 = self.ema_obj.average(self.actor_variables[8])
        # biases_fully1 = self.ema_obj.average(self.actor_variables[9])
        # weights_fully2 = self.ema_obj.average(self.actor_variables[10])
        # biases_fully2 = self.ema_obj.average(self.actor_variables[11])
        # weights_final = self.ema_obj.average(self.actor_variables[12])
        # biases_final = self.ema_obj.average(self.actor_variables[13])
        weights_fully1 = self.ema_obj.average(self.actor_variables[6])
        biases_fully1 = self.ema_obj.average(self.actor_variables[7])
        weights_fully2 = self.ema_obj.average(self.actor_variables[8])
        biases_fully2 = self.ema_obj.average(self.actor_variables[9])
        weights_final = self.ema_obj.average(self.actor_variables[10])
        biases_final = self.ema_obj.average(self.actor_variables[11])

        # 4 Convolutional layers
        conv1 = tf.nn.relu(tf.nn.conv2d(self.map_input_target, weights_conv1, strides=[1, STRIDE1, STRIDE1, 1],
                                        padding='VALID') + biases_conv1)
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights_conv2, strides=[1, STRIDE2, STRIDE2, 1], padding='VALID') +
                           biases_conv2)
        conv3 = tf.nn.relu(tf.nn.conv2d(conv2, weights_conv3, strides=[1, STRIDE3, STRIDE3, 1], padding='VALID') +
                           biases_conv3)
        # conv4 = tf.nn.relu(tf.nn.conv2d(conv3, weights_conv4, strides=[1, STRIDE4, STRIDE4, 1], padding='VALID') +
        #                   biases_conv4)

        # Reshape output tensor to a rank 1 tensor
        # conv_flat = tf.reshape(conv4, [-1, self.fully_size])
        conv_flat = tf.reshape(conv3, [-1, self.fully_size])

        # 2 Fully connected layers
        fully1 = tf.nn.relu(tf.matmul(conv_flat, weights_fully1) + biases_fully1)
        fully2 = tf.nn.relu(tf.matmul(fully1, weights_fully2) + biases_fully2)

        return tf.matmul(fully2, weights_final) + biases_final

    def restore_pretrained_weights(self, filter_path):

        # First restore the actor net
        saver = tf.train.Saver({"weights_conv1": self.actor_variables[0],
                                "biases_conv1":  self.actor_variables[1],
                                "weights_conv2": self.actor_variables[2],
                                "biases_conv2":  self.actor_variables[3],
                                "weights_conv3": self.actor_variables[4],
                                "biases_conv3":  self.actor_variables[5],
                                # "weights_conv4": self.actor_variables[6],
                                # "biases_conv4":  self.actor_variables[7]
                                })

        saver.restore(self.sess, filter_path)

        # Now restore the target net with
        saver_target = tf.train.Saver({"weights_conv1": self.ema_obj.average(self.actor_variables[0]),
                                       "biases_conv1":  self.ema_obj.average(self.actor_variables[1]),
                                       "weights_conv2": self.ema_obj.average(self.actor_variables[2]),
                                       "biases_conv2":  self.ema_obj.average(self.actor_variables[3]),
                                       "weights_conv3": self.ema_obj.average(self.actor_variables[4]),
                                       "biases_conv3":  self.ema_obj.average(self.actor_variables[5]),
                                       # "weights_conv4": self.ema_obj.average(self.actor_variables[6]),
                                       # "biases_conv4":  self.ema_obj.average(self.actor_variables[7])
                                       })

        saver_target.restore(self.sess, filter_path)

    def train(self, q_gradient_batch, state_batch):

        # Train the actor net
        self.sess.run(self.optimizer, feed_dict={self.q_gradient_input: q_gradient_batch, self.map_input: state_batch})

        # Update the target
        self.update_target()

        self.train_counter += 1

    def update_target(self):

        self.sess.run(self.compute_ema)

    def get_action(self, state):

        return self.sess.run(self.action_output, feed_dict={self.map_input: [state]})[0]

    def evaluate(self, state_batch):

        # Get an action batch
        actions = self.sess.run(self.action_output, feed_dict={self.map_input: state_batch})

        # Create summaries for the actions
        actions_mean = np.mean(np.asarray(actions, dtype=float), axis=0)
        self.actions_mean_plot += actions_mean

        # Only save files every PLOT_STEP steps
        if self.train_counter % PLOT_STEP == 0:

            self.actions_mean_plot /= PLOT_STEP

            summary_action_0 = tf.Summary(value=[tf.Summary.Value(tag='actions_mean[0]',
                                                                  simple_value=np.asscalar(
                                                                      self.actions_mean_plot[0]))])
            summary_action_1 = tf.Summary(value=[tf.Summary.Value(tag='actions_mean[1]',
                                                                  simple_value=np.asscalar(
                                                                      self.actions_mean_plot[1]))])
            self.summary_writer.add_summary(summary_action_0, self.train_counter)
            self.summary_writer.add_summary(summary_action_1, self.train_counter)

            self.actions_mean_plot = [0, 0]

        return actions

    def target_evaluate(self, state_batch):

        # Get action batch
        actions = self.sess.run(self.action_output_target, feed_dict={self.map_input_target: state_batch})

        # Create summaries for the target actions
        actions_mean = np.mean(np.asarray(actions, dtype=float), axis=0)
        self.target_actions_mean_plot += actions_mean

        # Only save files every 10 steps
        if (self.train_counter % PLOT_STEP) == 0:

            self.target_actions_mean_plot /= PLOT_STEP

            summary_target_action_0 = tf.Summary(value=[tf.Summary.Value(tag='target_actions_mean[0]',
                                                                         simple_value=np.asscalar(
                                                                             self.target_actions_mean_plot[0]))])
            summary_target_action_1 = tf.Summary(value=[tf.Summary.Value(tag='target_actions_mean[1]',
                                                                         simple_value=np.asscalar(
                                                                             self.target_actions_mean_plot[1]))])
            self.summary_writer.add_summary(summary_target_action_0, self.train_counter)
            self.summary_writer.add_summary(summary_target_action_1, self.train_counter)

            self.target_actions_mean_plot = [0, 0]

        return actions
