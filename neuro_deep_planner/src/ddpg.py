import numpy as np
import random
from collections import deque
from ou_noise import OUNoise
from critic import CriticNetwork
from actor import ActorNetwork
from tensorflow_grad_inverter import GradInverter
import tensorflow as tf

# For saving replay buffer
import pickle
import os

# Visualization
from state_visualizer import CostmapVisualizer


# Maximum replay buffer size
REPLAY_BUFFER_SIZE = 10000

# Minimum replay buffer size before we start training
REPLAY_START_SIZE = 1000

# How big are our mini batches
BATCH_SIZE = 32

# How big is our discount factor for rewards
GAMMA = 0.90

# How does our noise behave (MU = Center value, THETA = How strong is noise pulled to MU, SIGMA = Variance of noise)
MU = 0.0
THETA = 0.1
SIGMA = 0.1

# Should we load a saved net
PRE_TRAINED_NETS = False

# If we use a pretrained net
NET_SAVE_PATH = os.path.join(os.path.dirname(__file__), os.pardir)+"/pre_trained_networks/my_model"
NET_LOAD_PATH = os.path.join(os.path.dirname(__file__), os.pardir)+"/pre_trained_networks/my_model"

# If we don't use a pretrained net we should load pre-trained filters from this path
FILTER_LOAD_PATH = os.path.join(os.path.dirname(__file__), os.pardir) + "/pre_trained_filters/my_model"

# Should we use an existing initial buffer with experiences
NEW_INITIAL_BUFFER = False

# Visualize an initial state batch for debugging
VISUALIZE_BUFFER = False

# How often are we saving the net
SAVE_STEP = 1000


class DDPG:

    def __init__(self):

        # Initialize our session
        self.session = tf.Session()
        self.graph = self.session.graph

        with self.graph.as_default():

            # View the state batches
            self.visualize_input = VISUALIZE_BUFFER
            if self.visualize_input:
                self.viewer = CostmapVisualizer()

            # Hardcode input size and action size
            self.height = 86
            self.width = self.height
            self.depth = 4
            self.action_dim = 2

            # Initialize the current action and the old action and old state for setting experiences
            self.old_state = np.zeros((self.width, self.height, self.depth), dtype='float')
            self.old_action = np.ones(2, dtype='float')
            self.network_action = np.zeros(2, dtype='float')
            self.noise_action = np.zeros(2, dtype='float')
            self.action = np.zeros(2, dtype='float')

            # Initialize the grad inverter object to keep the action bounds
            self.action_bounds = [[0.3, 0.3],
                                  [-0.3, -0.3]]
            self.grad_inv = GradInverter(self.action_bounds)

            # Initialize summary writers to plot variables during training
            self.summary_op = tf.merge_all_summaries()
            self.summary_writer = tf.train.SummaryWriter(os.path.expanduser('~')+'/data')

            # Initialize actor and critic networks
            self.actor_network = ActorNetwork(self.height, self.action_dim, self.depth, self.session,
                                              self.summary_writer)
            self.critic_network = CriticNetwork(self.height, self.action_dim, self.depth, self.session,
                                                self.summary_writer)

            # Initialize the saver to save the network params
            self.saver = tf.train.Saver(max_to_keep=3)

            # Should we load the pre-trained params?
            # If so: Load the full pre-trained net
            # Else:  Initialize all variables the overwrite the conv layers with the pretrained filters
            if PRE_TRAINED_NETS:
                self.saver.restore(self.session, NET_LOAD_PATH)
            else:
                self.session.run(tf.initialize_all_variables())
                self.critic_network.restore_pretrained_weights(FILTER_LOAD_PATH)
                self.actor_network.restore_pretrained_weights(FILTER_LOAD_PATH)

            # Initialize replay buffer (ring buffer with max length)
            self.replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

            # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
            self.exploration_noise = OUNoise(self.action_dim, MU, THETA, SIGMA)
            self.noise_flag = True

            # Initialize time step
            self.training_step = 0

            # Flag: don't learn the first experience
            self.first_experience = True

            # Are we saving a new initial buffer or loading an existing one or neither?
            self.save_initial_buffer = NEW_INITIAL_BUFFER
            if not self.save_initial_buffer:
                self.replay_buffer = pickle.load(open(os.path.expanduser('~')+"/Desktop/initial_replay_buffer.p", "rb"))
            else:
                self.replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

            # After the graph has been filled add it to the summary writer
            self.summary_writer.add_graph(self.graph)

    def train(self):

        # Check if the buffer is big enough to start training
        if self.get_buffer_size() > REPLAY_START_SIZE:

            # Are we saving a pretrained buffer?
            if self.save_initial_buffer:
                self.save_buffer()
                self.save_initial_buffer = False

            # Sample a random minibatch of N transitions from replay buffer
            minibatch = random.sample(self.replay_buffer, BATCH_SIZE)

            # Split the batch into the sub components
            state_batch = [data[0] for data in minibatch]
            action_batch = [data[1] for data in minibatch]
            reward_batch = [data[2] for data in minibatch]
            next_state_batch = [data[3] for data in minibatch]

            # Are we visualizing the first state batch for debugging?
            # If so: We have to scale up the values for grey scale before plotting
            if self.visualize_input:
                state_batch_np = np.asarray(state_batch)
                state_batch_np = np.multiply(state_batch_np, -100.0)
                state_batch_np = np.add(state_batch_np, 100.0)
                self.viewer.set_data(state_batch_np)
                self.viewer.run()
                self.visualize_input = False

            # Calculate y for the td_error of the critic
            y_batch = []
            next_action_batch = self.actor_network.target_evaluate(next_state_batch)
            q_value_batch = self.critic_network.target_evaluate(next_state_batch, next_action_batch)
            for i in range(0, BATCH_SIZE):
                is_episode_finished = minibatch[i][4]
                if is_episode_finished:
                    y_batch.append([reward_batch[i]])
                else:
                    y_batch.append(reward_batch[i] + GAMMA * q_value_batch[i])

            # Now that we have the y batch lets train the critic
            self.critic_network.train(y_batch, state_batch, action_batch)

            # Get the action batch so we can calculate the action gradient with it
            # Then get the action gradient batch and adapt the gradient with the gradient inverting method
            action_batch_for_gradients = self.actor_network.evaluate(state_batch)
            q_gradient_batch = self.critic_network.get_action_gradient(state_batch, action_batch_for_gradients)
            q_gradient_batch = self.grad_inv.invert(q_gradient_batch, action_batch_for_gradients)

            # Now we can train the actor
            self.actor_network.train(q_gradient_batch, state_batch)

            # Save model if necessary
            if self.training_step > 0 and self.training_step % SAVE_STEP == 0:
                self.saver.save(self.session, NET_SAVE_PATH)

            # Update time step
            self.training_step += 1

    def get_action(self, state):

        # Get the action
        self.action = self.actor_network.get_action(state)

        # Are we using noise?
        if self.noise_flag:
            self.action += self.exploration_noise.noise()

        # Life q value output for this action and state
        self.print_q_value(state, self.action)

        return self.action

    def get_buffer_size(self):

        return len(self.replay_buffer)

    def set_experience(self, state, reward, is_episode_finished):

        # Make sure we're saving a new old_state for the first experience of every episode
        if self.first_experience:
            self.first_experience = False
        else:
            self.replay_buffer.append((self.old_state, self.old_action, reward, state, is_episode_finished))

        if is_episode_finished:
            self.first_experience = True

        # Safe old state and old action for next experience
        self.old_state = state
        self.old_action = self.action

    def save_buffer(self):

        pickle.dump(self.replay_buffer, open(os.path.expanduser('~')+"/Desktop/initial_replay_buffer.p", "wb"))

    def print_q_value(self, state, action):

        string = "-"
        q_value = self.critic_network.evaluate([state], [action])
        stroke_pos = 30 * q_value[0][0] + 30
        if stroke_pos < 0:
            stroke_pos = 0
        elif stroke_pos > 60:
            stroke_pos = 60
        print '[' + stroke_pos * string + '|' + (60-stroke_pos) * string + ']', "Q: ", q_value[0][0], \
            "\tt: ", self.training_step
