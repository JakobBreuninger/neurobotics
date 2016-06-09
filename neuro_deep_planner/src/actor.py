import tensorflow as tf
from critic import create_variable
from critic import create_variable_final


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
LEARNING_RATE = 0.0001  # standard learning rate

TARGET_DECAY = 0.999    # for target networks

FINAL_WEIGHT_INIT = 0.0003   # small init weights for output layer


class ActorNetwork:

    def __init__(self, image_size, action_size, image_no, batch_size):

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.InteractiveSession()

            self.image_size = image_size
            self.action_size = action_size
            self.image_no = image_no

            # Define fully connected layer size
            final_conv_height = (((((image_size - RECEPTIVE_FIELD1)/STRIDE1 + 1) - RECEPTIVE_FIELD2)/STRIDE2 + 1) -
                                 RECEPTIVE_FIELD3)/STRIDE3 + 1
            self.fully_size = (final_conv_height**2) * FILTER3

            # create actor network
            self.map_input, self.action_output = self.create_network()

            # Get all the variables in the actor network
            with tf.variable_scope("actor") as scope:

                self.actor_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope.name)

            # Create target actor network
            self.map_input_target, self.action_output_target = self.create_target_network()

            # Create Exponential moving Average Object
            self.ema_obj = tf.train.ExponentialMovingAverage(decay=TARGET_DECAY)

            # Create the shadow variables, and add ops to maintain moving averages
            # of actor network
            self.compute_ema = self.ema_obj.apply(self.actor_variables)

            # Define training rules
            self.q_gradient_input = tf.placeholder("float", [None, action_size])
            self.parameters_gradients = tf.gradients(self.action_output, self.actor_variables,
                                                     -self.q_gradient_input/batch_size)
            self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients,
                                                                                       self.actor_variables))

            # initialize al variables
            self.sess.run(tf.initialize_all_variables())

    def create_network(self):

        map_input = tf.placeholder("float", [None, self.image_size, self.image_size, self.image_no])

        with tf.variable_scope('actor'):

            weights_conv1 = create_variable([RECEPTIVE_FIELD1, RECEPTIVE_FIELD1, self.image_no, FILTER1],
                                            RECEPTIVE_FIELD1 * RECEPTIVE_FIELD1)
            biases_conv1 = create_variable([FILTER1], RECEPTIVE_FIELD1 * RECEPTIVE_FIELD1)

            weights_conv2 = create_variable([RECEPTIVE_FIELD2, RECEPTIVE_FIELD2, FILTER1, FILTER2],
                                            RECEPTIVE_FIELD2 * RECEPTIVE_FIELD2 * FILTER1)
            biases_conv2 = create_variable([FILTER2], RECEPTIVE_FIELD2 * RECEPTIVE_FIELD2 * FILTER1)

            weights_conv3 = create_variable([RECEPTIVE_FIELD3, RECEPTIVE_FIELD3, FILTER2, FILTER3],
                                            RECEPTIVE_FIELD3 * RECEPTIVE_FIELD3 * FILTER2)
            biases_conv3 = create_variable([FILTER3], RECEPTIVE_FIELD3 * RECEPTIVE_FIELD3 * FILTER2)

            weights_fully1 = create_variable([self.fully_size, FULLY_LAYER1_SIZE], self.fully_size)
            biases_fully1 = create_variable([FULLY_LAYER1_SIZE], self.fully_size)

            weights_fully2 = create_variable([FULLY_LAYER1_SIZE, FULLY_LAYER2_SIZE], FULLY_LAYER1_SIZE)
            biases_fully2 = create_variable([FULLY_LAYER2_SIZE], FULLY_LAYER1_SIZE)

            weights_final = create_variable_final([FULLY_LAYER2_SIZE, self.action_size])
            biases_final = create_variable_final([self.action_size])

        # 3 convolutional layers
        conv1 = tf.nn.relu(tf.nn.conv2d(map_input, weights_conv1, strides=[1, STRIDE1, STRIDE1, 1], padding='VALID') +
                           biases_conv1)
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights_conv2, strides=[1, STRIDE2, STRIDE2, 1], padding='VALID') +
                           biases_conv2)
        conv3 = tf.nn.relu(tf.nn.conv2d(conv2, weights_conv3, strides=[1, STRIDE3, STRIDE3, 1], padding='VALID') +
                           biases_conv3)

        # reshape output tensor to a rank 1 tensor
        conv3_flat = tf.reshape(conv3, [-1, self.fully_size])

        # more operations
        fully1 = tf.nn.relu(tf.matmul(conv3_flat, weights_fully1) + biases_fully1)
        fully2 = tf.nn.relu(tf.matmul(fully1, weights_fully2) + biases_fully2)
        action_output = tf.tanh(tf.matmul(fully2, weights_final) + biases_final)

        # return output op
        return map_input, action_output

    def create_target_network(self):

        map_input = tf.placeholder("float", [None, self.image_size, self.image_size, self.image_no])

        with tf.variable_scope('actor_target'):
            weights_conv1 = self.ema_obj.average(self.actor_variables[0])
            biases_conv1 = self.ema_obj.average(self.actor_variables[1])
            weights_conv2 = self.ema_obj.average(self.actor_variables[2])
            biases_conv2 = self.ema_obj.average(self.actor_variables[3])
            weights_conv3 = self.ema_obj.average(self.actor_variables[4])
            biases_conv3 = self.ema_obj.average(self.actor_variables[5])
            weights_fully1 = self.ema_obj.average(self.actor_variables[6])
            biases_fully1 = self.ema_obj.average(self.actor_variables[7])
            weights_fully2 = self.ema_obj.average(self.actor_variables[8])
            biases_fully2 = self.ema_obj.average(self.actor_variables[9])
            weights_final = self.ema_obj.average(self.actor_variables[10])
            biases_final = self.ema_obj.average(self.actor_variables[11])

        # 3 convolutional layers
        conv1 = tf.nn.relu(tf.nn.conv2d(map_input, weights_conv1, strides=[1, STRIDE1, STRIDE1, 1], padding='VALID') +
                           biases_conv1)
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights_conv2, strides=[1, STRIDE2, STRIDE2, 1], padding='VALID') +
                           biases_conv2)
        conv3 = tf.nn.relu(tf.nn.conv2d(conv2, weights_conv3, strides=[1, STRIDE3, STRIDE3, 1], padding='VALID') +
                           biases_conv3)

        # Reshape output tensor to a rank 1 tensor
        conv3_flat = tf.reshape(conv3, [-1, self.fully_size])

        # more operations
        fully1 = tf.nn.relu(tf.matmul(conv3_flat, weights_fully1) + biases_fully1)
        fully2 = tf.nn.relu(tf.matmul(fully1, weights_fully2) + biases_fully2)
        action_output = tf.tanh(tf.matmul(fully2, weights_final) + biases_final)

        # return all ops
        return map_input, action_output

    def update_target(self):
        self.sess.run(self.compute_ema)

    def evaluate(self, state_batch):
        return self.sess.run(self.action_output, feed_dict={self.map_input: state_batch})

    def train(self, q_gradient_batch, state_batch):
        self.sess.run(self.optimizer, feed_dict={self.q_gradient_input: q_gradient_batch, self.map_input: state_batch})
        self.update_target()

    def get_action(self, state):
        return self.sess.run(self.action_output, feed_dict={self.map_input: [state]})[0]

    def target_evaluate(self, state_batch):
        return self.sess.run(self.action_output_target, feed_dict={self.map_input_target: state_batch})
