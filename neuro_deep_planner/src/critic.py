import tensorflow as tf
import math
import numpy as np

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

# How much do we regularize the weights of the net
REGULARIZATION_DECAY = 0.0

# How fast does the target net track
TARGET_DECAY = 0.9999

# In what range are we initializing the weights in the final layer
FINAL_WEIGHT_INIT = 0.003

# How often do we plot variables during training
PLOT_STEP = 10


class CriticNetwork:

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

            # Create critic network
            self.map_input = tf.placeholder("float", [None, image_size, image_size, image_no])
            self.action_input = tf.placeholder("float", [None, action_size], name="action_input")
            self.Q_output = self.create_network()

            # Get all the variables in the critic network for exponential moving average, create ema op
            with tf.variable_scope("critic") as scope:
                self.critic_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope.name)
            self.ema_obj = tf.train.ExponentialMovingAverage(decay=TARGET_DECAY)
            self.compute_ema = self.ema_obj.apply(self.critic_variables)

            # Create target actor network
            self.map_input_target = tf.placeholder("float", [None, image_size, image_size, image_no])
            self.action_input_target = tf.placeholder("float", [None, action_size])
            self.Q_output_target = self.create_target_network()

            # L2 Regularization for all Variables
            self.regularization = 0
            for variable in self.critic_variables:
                self.regularization += tf.nn.l2_loss(variable)

            # Define the loss with regularization term
            self.y_input = tf.placeholder("float", [None, 1], name="y_input")
            self.td_error = tf.reduce_mean(tf.pow(self.Q_output - self.y_input, 2))
            self.loss = self.td_error + REGULARIZATION_DECAY * self.regularization

            # Define the optimizer
            self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)

            # Define the action gradients for the actor training step
            self.action_gradients = tf.gradients(self.Q_output, self.action_input)

            # Variables for plotting
            self.action_grads_mean_plot = [0, 0]
            self.td_error_plot = 0

            # Training step counter (gets incremented after each training step)
            self.train_counter = 0

    def create_network(self):

        with tf.variable_scope('critic'):

            weights_conv1 = create_variable([RECEPTIVE_FIELD1, RECEPTIVE_FIELD1, self.image_no, FILTER1],
                                            RECEPTIVE_FIELD1 * RECEPTIVE_FIELD1 * self.image_no, "weights_conv1")
            biases_conv1 = create_variable([FILTER1], RECEPTIVE_FIELD1 * RECEPTIVE_FIELD1 * self.image_no,
                                           "biases_conv1")

            weights_conv2 = create_variable([RECEPTIVE_FIELD2, RECEPTIVE_FIELD2, FILTER1, FILTER2],
                                            RECEPTIVE_FIELD2 * RECEPTIVE_FIELD2 * FILTER1, "weights_conv2")
            biases_conv2 = create_variable([FILTER2], RECEPTIVE_FIELD2 * RECEPTIVE_FIELD2 * FILTER1,
                                           "biases_conv2")

            weights_conv3 = create_variable([RECEPTIVE_FIELD3, RECEPTIVE_FIELD3, FILTER2, FILTER3],
                                            RECEPTIVE_FIELD3 * RECEPTIVE_FIELD3 * FILTER2, "weights_conv3")
            biases_conv3 = create_variable([FILTER3], RECEPTIVE_FIELD3 * RECEPTIVE_FIELD3 * FILTER2, "biases_conv3")

            # weights_conv4 = create_variable([RECEPTIVE_FIELD4, RECEPTIVE_FIELD4, FILTER3, FILTER4],
            #                                RECEPTIVE_FIELD4 * RECEPTIVE_FIELD4 * FILTER3, "weights_conv4")
            # biases_conv4 = create_variable([FILTER4], RECEPTIVE_FIELD4 * RECEPTIVE_FIELD4 * FILTER3, "biases_conv4")

            weights_actions = create_variable([self.action_size, FULLY_LAYER1_SIZE], self.fully_size)
            weights_fully1 = create_variable([self.fully_size, FULLY_LAYER1_SIZE], self.fully_size)
            biases_fully1 = create_variable([FULLY_LAYER1_SIZE], self.fully_size)

            weights_fully2 = create_variable([FULLY_LAYER1_SIZE, FULLY_LAYER2_SIZE], FULLY_LAYER1_SIZE)
            biases_fully2 = create_variable([FULLY_LAYER2_SIZE], FULLY_LAYER1_SIZE)

            weights_final = create_variable_final([FULLY_LAYER2_SIZE, 1])
            biases_final = create_variable_final([1])

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
        fully1 = tf.nn.relu(tf.matmul(conv_flat, weights_fully1) + tf.matmul(self.action_input, weights_actions) +
                            biases_fully1)
        fully2 = tf.nn.relu(tf.matmul(fully1, weights_fully2) + biases_fully2)

        return tf.matmul(fully2, weights_final) + biases_final

    def create_target_network(self):

        weights_conv1 = self.ema_obj.average(self.critic_variables[0])
        biases_conv1 = self.ema_obj.average(self.critic_variables[1])
        weights_conv2 = self.ema_obj.average(self.critic_variables[2])
        biases_conv2 = self.ema_obj.average(self.critic_variables[3])
        weights_conv3 = self.ema_obj.average(self.critic_variables[4])
        biases_conv3 = self.ema_obj.average(self.critic_variables[5])
        # weights_conv4 = self.ema_obj.average(self.critic_variables[6])
        # biases_conv4 = self.ema_obj.average(self.critic_variables[7])
        # weights_actions = self.ema_obj.average(self.critic_variables[8])
        # weights_fully1 = self.ema_obj.average(self.critic_variables[9])
        # biases_fully1 = self.ema_obj.average(self.critic_variables[10])
        # weights_fully2 = self.ema_obj.average(self.critic_variables[11])
        # biases_fully2 = self.ema_obj.average(self.critic_variables[12])
        # weights_final = self.ema_obj.average(self.critic_variables[13])
        # biases_final = self.ema_obj.average(self.critic_variables[14])
        weights_actions = self.ema_obj.average(self.critic_variables[6])
        weights_fully1 = self.ema_obj.average(self.critic_variables[7])
        biases_fully1 = self.ema_obj.average(self.critic_variables[8])
        weights_fully2 = self.ema_obj.average(self.critic_variables[9])
        biases_fully2 = self.ema_obj.average(self.critic_variables[10])
        weights_final = self.ema_obj.average(self.critic_variables[11])
        biases_final = self.ema_obj.average(self.critic_variables[12])

        # 4 Convolutional layers
        conv1 = tf.nn.relu(tf.nn.conv2d(self.map_input_target, weights_conv1, strides=[1, STRIDE1, STRIDE1, 1],
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
        fully1 = tf.nn.relu(tf.matmul(conv_flat, weights_fully1) + tf.matmul(self.action_input_target,
                                                                             weights_actions) + biases_fully1)
        fully2 = tf.nn.relu(tf.matmul(fully1, weights_fully2) + biases_fully2)

        return tf.matmul(fully2, weights_final) + biases_final

    def restore_pretrained_weights(self, filter_path):

        # First restore the critic filters
        saver = tf.train.Saver({"weights_conv1": self.critic_variables[0],
                                "biases_conv1":  self.critic_variables[1],
                                "weights_conv2": self.critic_variables[2],
                                "biases_conv2":  self.critic_variables[3],
                                "weights_conv3": self.critic_variables[4],
                                "biases_conv3":  self.critic_variables[5],
                                # "weights_conv4": self.critic_variables[6],
                                # "biases_conv4":  self.critic_variables[7]
                                })

        saver.restore(self.sess, filter_path)

        # Now restore the target net filters
        saver_target = tf.train.Saver({"weights_conv1": self.ema_obj.average(self.critic_variables[0]),
                                       "biases_conv1":  self.ema_obj.average(self.critic_variables[1]),
                                       "weights_conv2": self.ema_obj.average(self.critic_variables[2]),
                                       "biases_conv2":  self.ema_obj.average(self.critic_variables[3]),
                                       "weights_conv3": self.ema_obj.average(self.critic_variables[4]),
                                       "biases_conv3":  self.ema_obj.average(self.critic_variables[5]),
                                       # "weights_conv4": self.ema_obj.average(self.critic_variables[6]),
                                       # "biases_conv4":  self.ema_obj.average(self.critic_variables[7])
                                       })

        saver_target.restore(self.sess, filter_path)

    def train(self, y_batch, state_batch, action_batch):

        # Run optimizer and compute some summary values
        td_error_value, _ = self.sess.run([self.td_error, self.optimizer], feed_dict={self.y_input: y_batch,
                                                                                      self.map_input: state_batch,
                                                                                      self.action_input: action_batch})

        # Now update the target net
        self.update_target()

        # Increment the td error plot variable for td error average plotting
        self.td_error_plot += td_error_value

        # Only save data every 10 steps
        if self.train_counter % PLOT_STEP == 0:

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

        # Get the action gradients for the actor optimization
        action_gradients = self.sess.run(self.action_gradients, feed_dict={self.map_input: state_batch,
                                                                           self.action_input: action_batch})[0]

        # Create summaries for the action gradients and add them to the summary writer
        action_grads_mean = np.mean(action_gradients, axis=0)
        self.action_grads_mean_plot += action_grads_mean

        # Only save data every 10 steps
        if self.train_counter % PLOT_STEP == 0:

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


# Create variables with a fan-in condition
def create_variable(shape, f, name=None, trainable=True):
    return tf.Variable(tf.random_uniform(shape, -1/math.sqrt(f), 1/math.sqrt(f)), name=name, trainable=trainable)


def create_variable_final(shape):
    return tf.Variable(tf.random_uniform(shape, -FINAL_WEIGHT_INIT, FINAL_WEIGHT_INIT))
