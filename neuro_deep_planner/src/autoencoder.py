import numpy as np
import random
from collections import deque
from critic import create_variable
import tensorflow as tf

# For saving replay buffer
import pickle
import os

# Visualization
from state_visualizer import CostmapVisualizer

# Hyper Parameters:
BATCH_SIZE = 16              # How big are our batches

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


# For plotting
PLOT_STEP = 10
INPUT_OUTPUT_STEP = 100


class AutoEncoder:

    def __init__(self):

        # Hardcode input size and action size
        self.height = 84
        self.width = self.height
        self.depth = 4

        self.session = tf.Session()

        self.summary_op = tf.merge_all_summaries()
        self.summary_writer = tf.train.SummaryWriter('data')

        # Define map input
        self.map_input = tf.placeholder("float", [None, self.width, self.height, self.depth])

        # Define trainable variables
        self.weights_conv1 = create_variable([RECEPTIVE_FIELD1, RECEPTIVE_FIELD1, self.depth, FILTER1],
                                             RECEPTIVE_FIELD1 * RECEPTIVE_FIELD1 * self.depth, "weights_conv1")
        self.biases_conv1 = create_variable([FILTER1], RECEPTIVE_FIELD1 * RECEPTIVE_FIELD1 * self.depth, "biases_conv1")

        self.weights_conv2 = create_variable([RECEPTIVE_FIELD2, RECEPTIVE_FIELD2, FILTER1, FILTER2],
                                             RECEPTIVE_FIELD2 * RECEPTIVE_FIELD2 * FILTER1, "weights_conv2")
        self.biases_conv2 = create_variable([FILTER2], RECEPTIVE_FIELD2 * RECEPTIVE_FIELD2 * FILTER1, "biases_conv2")

        self.weights_conv3 = create_variable([RECEPTIVE_FIELD3, RECEPTIVE_FIELD3, FILTER2, FILTER3],
                                             RECEPTIVE_FIELD3 * RECEPTIVE_FIELD3 * FILTER2, "weights_conv3")
        self.biases_conv3 = create_variable([FILTER3], RECEPTIVE_FIELD3 * RECEPTIVE_FIELD3 * FILTER2, "biases_conv3")

        # Define expansion variables
        self.weights_conv1_transpose = create_variable([RECEPTIVE_FIELD1, RECEPTIVE_FIELD1, self.depth, FILTER1],
                                                       RECEPTIVE_FIELD1 * RECEPTIVE_FIELD1 * self.depth,
                                                       "weights_conv1_tran")
        self.biases_conv1_transpose = create_variable([self.depth], RECEPTIVE_FIELD1 * RECEPTIVE_FIELD1 * self.depth,
                                                      "biases_conv1_tran")

        self.weights_conv2_transpose = create_variable([RECEPTIVE_FIELD2, RECEPTIVE_FIELD2, FILTER1, FILTER2],
                                                       RECEPTIVE_FIELD2 * RECEPTIVE_FIELD2 * FILTER1,
                                                       "weights_conv2_tran")
        self.biases_conv2_transpose = create_variable([FILTER1], RECEPTIVE_FIELD2 * RECEPTIVE_FIELD2 * FILTER1,
                                                      "biases_conv2_tran")

        self.weights_conv3_transpose = create_variable([RECEPTIVE_FIELD3, RECEPTIVE_FIELD3, FILTER2, FILTER3],
                                                       RECEPTIVE_FIELD3 * RECEPTIVE_FIELD3 * FILTER2,
                                                       "weights_conv3_tran")
        self.biases_conv3_transpose = create_variable([FILTER3], RECEPTIVE_FIELD3 * RECEPTIVE_FIELD3 * FILTER2,
                                                      "biases_conv3_tran")

        # Initialize auto-encoder and define the output
        self.map_output = self.get_output()

        # Get the loss
        self.loss = tf.reduce_mean(tf.pow(self.map_input - self.map_output, 2))

        # Define the optimizer
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)

        # Save path for session
        self.save_path = os.path.expanduser('~')+"/Desktop/auto_encoder/my_model"
        self.saver = tf.train.Saver({"weights_conv1": self.weights_conv1,
                                     "biases_conv1":  self.biases_conv1,
                                     "weights_conv2": self.weights_conv2,
                                     "biases_conv2":  self.biases_conv2,
                                     "weights_conv3": self.weights_conv3,
                                     "biases_conv3":  self.biases_conv3})

        # Either initialize the variables or load them
        self.session.run(tf.initialize_all_variables())
        self.saver.restore(self.session, self.save_path+"")

        self.summary_writer.add_graph(self.session.graph)

        # Initialize replay buffer (ring buffer with max length)
        self.replay_buffer = deque()

        # Initialize time step
        self.time_step = 0

        # Are we saving a new initial buffer or loading an existing one or neither?
        self.replay_buffer = pickle.load(open(os.path.expanduser('~')+"/Desktop/auto_encoder_buffer.p", "rb"))

        # Save the loss parameter to plot
        self.current_loss_plot = 0.0

        # Create a viewer to compare input and output
        self.viewer = CostmapVisualizer()

        # Plot the filters
        # self.transposed_weights = tf.transpose(self.weights_conv1, [3, 0, 1, 2])
        # self.filter_summary = tf.image_summary("filters", self.transposed_weights, max_images=32)

    def get_output(self):

        # Compress
        conv1 = tf.nn.conv2d(self.map_input, self.weights_conv1, strides=[1, STRIDE1, STRIDE1, 1], padding='VALID')
        relu1 = tf.nn.relu(conv1 + self.biases_conv1)

        conv2 = tf.nn.conv2d(relu1, self.weights_conv2, strides=[1, STRIDE2, STRIDE2, 1], padding='VALID')
        relu2 = tf.nn.relu(conv2 + self.biases_conv2)

        conv3 = tf.nn.conv2d(relu2, self.weights_conv3, strides=[1, STRIDE3, STRIDE3, 1], padding='VALID')
        relu3 = tf.nn.relu(conv3 + self.biases_conv3)

        # Expand
        conv_3_transpose_shape = tf.pack([BATCH_SIZE, 9, 9, FILTER2])
        conv3_transpose = tf.nn.conv2d_transpose(relu3, self.weights_conv3_transpose, conv_3_transpose_shape,
                                                 strides=[1, STRIDE3, STRIDE3, 1], padding='VALID')
        lin3_transpose = tf.nn.relu(conv3_transpose + self.biases_conv3_transpose)

        conv_2_transpose_shape = tf.pack([BATCH_SIZE, 20, 20, FILTER1])
        conv2_transpose = tf.nn.conv2d_transpose(lin3_transpose, self.weights_conv2_transpose, conv_2_transpose_shape,
                                                 strides=[1, STRIDE2, STRIDE2, 1], padding='VALID')
        lin2_transpose = tf.nn.relu(conv2_transpose + self.biases_conv2_transpose)

        conv_1_transpose_shape = tf.pack([BATCH_SIZE, 84, 84, 4])
        conv1_transpose = tf.nn.conv2d_transpose(lin2_transpose, self.weights_conv1_transpose, conv_1_transpose_shape,
                                                 strides=[1, STRIDE1, STRIDE1, 1], padding='VALID')
        lin1_transpose = tf.nn.relu(conv1_transpose + self.biases_conv1_transpose)

        return lin1_transpose

    def train(self):

        # Sample a random minibatch of N transitions from replay buffer
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)

        state_batch = [data[0] for data in minibatch]

        # Get loss
        current_loss, output_batch, _ = self.session.run([self.loss, self.map_output, self.optimizer],
                                                         feed_dict={self.map_input: state_batch})

        # Visualize the loss
        self.current_loss_plot += current_loss
        if self.time_step > 0 and self.time_step % PLOT_STEP == 0:

            self.current_loss_plot /= PLOT_STEP

            # Add td error to the summary writer
            summary = tf.Summary(value=[tf.Summary.Value(tag='td_error_mean',
                                                         simple_value=np.asscalar(self.current_loss_plot))])
            self.summary_writer.add_summary(summary, self.time_step)

            self.current_loss_plot = 0.0

        # Visualize the actual input and output to compare
        if self.time_step % INPUT_OUTPUT_STEP == 0:
            input_batch_np = np.asarray(state_batch)
            output_batch_np = np.asarray(output_batch)

            # Scale up to grey scale again
            state_batch_np = np.multiply(input_batch_np[0], -100.0)
            state_batch_np = np.add(state_batch_np, 100.0)
            input_map = np.expand_dims(state_batch_np, axis=0)

            state_batch_np = np.multiply(output_batch_np[0], -100.0)
            state_batch_np = np.add(state_batch_np, 100.0)
            output_map = np.expand_dims(state_batch_np, axis=0)

            plot = np.vstack((input_map, output_map))
            self.viewer.set_data(plot)
            self.viewer.run()

            print "t:", self.time_step

        # Save model if necessary
        if self.time_step > 0 and self.time_step % 20000 == 0:
            # Append the step number to the checkpoint name:
            self.saver.save(self.session, self.save_path)

        # Update time step
        self.time_step += 1


def main():

    # Initialize the ANNs
    auto_encoder = AutoEncoder()

    for i in range(100000):

        auto_encoder.train()


if __name__ == '__main__':
    main()
