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

STRIDE1 = 4
STRIDE2 = 2

FILTER1 = 32
FILTER2 = 32

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

        # Initialize auto-encoder and define the output
        self.map_output = self.create_network()

        # Get the loss
        self.loss = tf.reduce_mean(tf.pow(self.map_input - self.map_output, 2))

        # Define the optimizer
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)

        self.session.run(tf.initialize_all_variables())

        self.summary_writer.add_graph(self.session.graph)

        # Initialize replay buffer (ring buffer with max length)
        self.replay_buffer = deque()

        # Initialize time step
        self.time_step = 0

        # Save path for session
        # self.save_path = os.path.expanduser('~')+"/Desktop/auto_encoder/my_model"
        # self.saver = tf.train.Saver()

        # Are we saving a new initial buffer or loading an existing one or neither?
        self.replay_buffer = pickle.load(open(os.path.expanduser('~')+"/Desktop/auto_encoder_buffer.p", "rb"))

        # Save the loss parameter to plot
        self.current_loss_plot = 0.0

        # Create a viewer to compare input and output
        self.viewer = CostmapVisualizer()

        # Plot the filters
        self.transposed_weights = tf.transpose(self.weights_conv1, [3, 0, 1, 2])
        self.split_weight_1, self.split_weight_2, self.split_weight_3, self.split_weight_4 = tf.split(3, 4, self.transposed_weights)

        self.filter_summary = tf.image_summary("filters", self.split_weight_1, max_images=32)


    def create_network(self):

        # Define trainable variables
        self.weights_conv1 = create_variable([RECEPTIVE_FIELD1, RECEPTIVE_FIELD1, self.depth, FILTER1],
                                             RECEPTIVE_FIELD1 * RECEPTIVE_FIELD1 * self.depth)
        biases_conv1 = create_variable([FILTER1], RECEPTIVE_FIELD1 * RECEPTIVE_FIELD1 * self.depth)

        weights_conv2 = create_variable([RECEPTIVE_FIELD2, RECEPTIVE_FIELD2, FILTER1, FILTER2],
                                        RECEPTIVE_FIELD2 * RECEPTIVE_FIELD2 * FILTER1)
        biases_conv2 = create_variable([FILTER2], RECEPTIVE_FIELD2 * RECEPTIVE_FIELD2 * FILTER1)

        # Define expansion variables which mirror the trainable ones but are not trainable
        weights_conv1_transpose = create_variable([RECEPTIVE_FIELD1, RECEPTIVE_FIELD1, self.depth, FILTER1],
                                                  RECEPTIVE_FIELD1 * RECEPTIVE_FIELD1 * self.depth)
        biases_conv1_transpose = create_variable([self.depth], RECEPTIVE_FIELD1 * RECEPTIVE_FIELD1 * self.depth)

        weights_conv2_transpose = create_variable([RECEPTIVE_FIELD2, RECEPTIVE_FIELD2, FILTER1, FILTER2],
                                                  RECEPTIVE_FIELD2 * RECEPTIVE_FIELD2 * FILTER1)
        biases_conv2_transpose = create_variable([FILTER1], RECEPTIVE_FIELD2 * RECEPTIVE_FIELD2 * FILTER1)

        # Compress
        conv1 = tf.nn.conv2d(self.map_input, self.weights_conv1, strides=[1, STRIDE1, STRIDE1, 1], padding='VALID')
        relu1 = tf.nn.relu(conv1 + biases_conv1)

        conv2 = tf.nn.conv2d(relu1, weights_conv2, strides=[1, STRIDE2, STRIDE2, 1], padding='VALID')
        relu2 = tf.nn.relu(conv2 + biases_conv2)

        # Expand
        conv_2_transpose_shape = tf.pack([BATCH_SIZE, 20, 20, FILTER1])
        conv2_transpose = tf.nn.conv2d_transpose(relu2, weights_conv2_transpose, conv_2_transpose_shape,
                                                 strides=[1, STRIDE2, STRIDE2, 1], padding='VALID')
        lin2_transpose = tf.nn.relu(conv2_transpose + biases_conv2_transpose)

        conv_1_transpose_shape = tf.pack([BATCH_SIZE, 84, 84, 4])
        conv1_transpose = tf.nn.conv2d_transpose(lin2_transpose, weights_conv1_transpose, conv_1_transpose_shape,
                                                 strides=[1, STRIDE1, STRIDE1, 1], padding='VALID')
        lin1_transpose = tf.nn.relu(conv1_transpose + biases_conv1_transpose)

        return lin1_transpose

    def train(self):

        # Sample a random minibatch of N transitions from replay buffer
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)

        state_batch = [data[0] for data in minibatch]

        # Get loss
        weights, current_loss, output_batch, _ = self.session.run([self.filter_summary, self.loss, self.map_output, self.optimizer],
                                                                  feed_dict={self.map_input: state_batch})

        # Visualize the loss
        self.current_loss_plot += current_loss
        if self.time_step > 0 and (self.time_step % PLOT_STEP) == 0:

            self.current_loss_plot /= PLOT_STEP

            # Add td error to the summary writer
            summary = tf.Summary(value=[tf.Summary.Value(tag='td_error_mean',
                                                         simple_value=np.asscalar(self.current_loss_plot))])
            self.summary_writer.add_summary(summary, self.time_step)

            self.summary_writer.add_summary(weights, self.time_step)

            self.current_loss_plot = 0.0

        # Visualize the actual input and output to compare
        if (self.time_step % INPUT_OUTPUT_STEP) == 0:
            input_batch_np = np.asarray(state_batch)
            output_batch_np = np.asarray(output_batch)

            # Scale up to grey scale again
            state_batch_np = np.multiply(input_batch_np, -100.0)
            state_batch_np = np.add(state_batch_np, 100.0)
            input = np.expand_dims(state_batch_np[0], axis=0)

            state_batch_np = np.multiply(output_batch_np, -100.0)
            state_batch_np = np.add(state_batch_np, 100.0)
            output = np.expand_dims(state_batch_np[0], axis=0)

            plot = np.vstack((input, output))
            self.viewer.set_data(plot)
            self.viewer.run()

        # Save model if necessary
        # if self.time_step % 10000 == 0:
        #    # Append the step number to the checkpoint name:
        #    self.saver.save(self.session, self.save_path, global_step=self.time_step)

        # Update time step
        self.time_step += 1


def main():

    # Initialize the ANNs
    auto_encoder = AutoEncoder()

    for i in range(100000):

        auto_encoder.train()


if __name__ == '__main__':
    main()
