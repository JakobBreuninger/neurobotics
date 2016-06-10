import tensorflow as tf
import math

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
LEARNING_RATE = 0.0001       # standard learning rate

REGULARIZATION_DECAY = 0.01  # for L2 Regularization

TARGET_DECAY = 0.999         # for target networks

FINAL_WEIGHT_INIT = 0.0003   # small init weights for output layer


class CriticNetwork:

    def __init__(self, image_size, action_size, image_no, batch_size):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.InteractiveSession()

            self.train_counter = 1
            self.td_error_sum = 0

            # Define fully connected layer size
            final_conv_height = (((((image_size - RECEPTIVE_FIELD1)/STRIDE1 + 1) - RECEPTIVE_FIELD2)/STRIDE2 + 1) -
                                 RECEPTIVE_FIELD3)/STRIDE3 + 1
            self.fully_size = (final_conv_height**2) * FILTER3

            # create actor network
            self.map_input, self.action_input, self.Q_output = self.create_network(image_size, action_size, image_no)

            # get all the variables in the actor network
            with tf.variable_scope("critic") as scope:
                self.critic_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope.name)

            # Create Exponential moving Average Object
            self.ema_obj = tf.train.ExponentialMovingAverage(decay=TARGET_DECAY)

            # Create the shadow variables, and add ops to maintain moving averages
            # of critic network
            self.compute_ema = self.ema_obj.apply(self.critic_variables)

            # create target actor network
            self.map_input_target, self.action_input_target, self.Q_output_target = self.create_target_network(
                image_size, action_size, image_no, self.ema_obj, self.critic_variables)

            # L2 Regularization for all Variables
            self.regularization = 0
            for variable in self.critic_variables:
                self.regularization = self.regularization + tf.nn.l2_loss(variable)

            # Define training optimizer
            self.y_input = tf.placeholder("float", [None, 1], name="y_input")
            self.td_error = tf.reduce_mean(tf.pow(self.Q_output-self.y_input, 2))
            self.loss = self.td_error + self.regularization



            self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)

            self.action_gradients = tf.gradients(self.Q_output, self.action_input)

            self.summary_writer = tf.train.SummaryWriter('data', self.graph)

            # initiallize all variables
            self.sess.run(tf.initialize_all_variables())

    def create_network(self, image_size, action_size, image_no):
        map_input = tf.placeholder("float", [None, image_size, image_size, image_no])
        action_input = tf.placeholder("float", [None, action_size], name="action_input")

        with tf.variable_scope('critic'):

            weights_conv1 = create_variable([RECEPTIVE_FIELD1, RECEPTIVE_FIELD1, image_no, FILTER1],
                                            RECEPTIVE_FIELD1 * RECEPTIVE_FIELD1)
            biases_conv1 = create_variable([FILTER1], RECEPTIVE_FIELD1 * RECEPTIVE_FIELD1)

            weights_conv2 = create_variable([RECEPTIVE_FIELD2, RECEPTIVE_FIELD2, FILTER1, FILTER2],
                                            RECEPTIVE_FIELD2 * RECEPTIVE_FIELD2 * FILTER1)
            biases_conv2 = create_variable([FILTER2], RECEPTIVE_FIELD2 * RECEPTIVE_FIELD2 * FILTER1)

            weights_conv3 = create_variable([RECEPTIVE_FIELD3, RECEPTIVE_FIELD3, FILTER2, FILTER3],
                                            RECEPTIVE_FIELD3 * RECEPTIVE_FIELD3 * FILTER2)
            biases_conv3 = create_variable([FILTER3], RECEPTIVE_FIELD3 * RECEPTIVE_FIELD3 * FILTER2)

            weights_actions = create_variable([action_size, FULLY_LAYER1_SIZE], action_size)
            weights_fully1 = create_variable([self.fully_size, FULLY_LAYER1_SIZE], self.fully_size)
            biases_fully1 = create_variable([FULLY_LAYER1_SIZE], self.fully_size)

            weights_fully2 = create_variable([FULLY_LAYER1_SIZE, FULLY_LAYER2_SIZE], FULLY_LAYER1_SIZE)
            biases_fully2 = create_variable([FULLY_LAYER2_SIZE], FULLY_LAYER1_SIZE)

            weights_final = create_variable_final([FULLY_LAYER2_SIZE, 1])
            biases_final = create_variable_final([1])

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
        fully1 = tf.nn.relu(tf.matmul(conv3_flat, weights_fully1) + tf.matmul(action_input, weights_actions) +
                            biases_fully1)
        fully2 = tf.nn.relu(tf.matmul(fully1, weights_fully2) + biases_fully2)
        q_output = tf.matmul(fully2, weights_final) + biases_final

        # return all ops
        return map_input, action_input, q_output

    def create_target_network(self, image_size, action_size, image_no, ema_obj, critic_variables):

        map_input = tf.placeholder("float", [None, image_size, image_size, image_no])
        action_input = tf.placeholder("float", [None, action_size])

        with tf.variable_scope('critic_target'):
            weights_conv1 = ema_obj.average(critic_variables[0])
            biases_conv1 = ema_obj.average(critic_variables[1])
            weights_conv2 = ema_obj.average(critic_variables[2])
            biases_conv2 = ema_obj.average(critic_variables[3])
            weights_conv3 = ema_obj.average(critic_variables[4])
            biases_conv3 = ema_obj.average(critic_variables[5])
            weights_actions = ema_obj.average(critic_variables[6])
            weights_fully1 = ema_obj.average(critic_variables[7])
            biases_fully1 = ema_obj.average(critic_variables[8])
            weights_fully2 = ema_obj.average(critic_variables[9])
            biases_fully2 = ema_obj.average(critic_variables[10])
            weights_final = ema_obj.average(critic_variables[11])
            biases_final = ema_obj.average(critic_variables[12])

        # reshape image to apply convolution
        # input_image = tf.reshape(input_array, [-1,image_size,image_size,1])

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
        fully1 = tf.nn.relu(tf.matmul(conv3_flat, weights_fully1) + tf.matmul(action_input, weights_actions) +
                            biases_fully1)
        fully2 = tf.nn.relu(tf.matmul(fully1, weights_fully2) + biases_fully2)
        q_output = tf.matmul(fully2, weights_final) + biases_final

        # return all ops
        return map_input, action_input, q_output

    def train(self, y_batch, state_batch, action_batch):
        td_error_value, _ = self.sess.run([self.td_error, self.optimizer], feed_dict={self.y_input: y_batch, self.map_input: state_batch, self.action_input:
                                                 action_batch})
        self.update_target()
        self.td_error_sum += td_error_value/100

        # write the mean td_error over the last 100 batches to the summary
        if (self.train_counter % 100) == 0:
            summary = tf.Summary(value=[tf.Summary.Value(tag='td_error', simple_value=self.td_error_sum)])
            self.summary_writer.add_summary(summary, self.train_counter)
            self.td_error_sum = 0
        self.train_counter += 1

    def update_target(self):
        self.sess.run(self.compute_ema)

    def gradients(self, state_batch, action_batch):
        return self.sess.run(self.action_gradients, feed_dict={
            self.map_input: state_batch,
            self.action_input: action_batch
        })[0]

    def evaluate(self, state_batch, action_batch):
        return self.sess.run(self.Q_output, feed_dict={self.map_input: state_batch, self.action_input: action_batch})

    def target_evaluate(self, state_batch, action_batch):
        return self.sess.run(self.Q_output_target, feed_dict={
            self.map_input_target: state_batch,
            self.action_input_target: action_batch
        })


# f fan-in size
def create_variable(shape, f):
    return tf.Variable(tf.random_uniform(shape, -1/math.sqrt(f), 1/math.sqrt(f)))


def create_variable_final(shape):
    return tf.Variable(tf.random_uniform(shape, -FINAL_WEIGHT_INIT, FINAL_WEIGHT_INIT))
