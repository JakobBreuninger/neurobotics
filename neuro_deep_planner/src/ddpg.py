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


# Hyper Parameters:
REPLAY_BUFFER_SIZE = 10000   # How big can the buffer get
REPLAY_START_SIZE = 1000     # When do we start training

BATCH_SIZE = 32              # How big are our batches

GAMMA = 0.95                  # Discount factor

MU = 0.0                     # Center value of noise
THETA = 0.1                  # Specifies how strong noise values are pulled towards mu
SIGMA = 0.1                  # Variance of noise

# Should we load a saved net
PRE_TRAINED_NETS = False

# Should we use an existing initial buffer with experiences
NEW_INITIAL_BUFFER = False


class DDPG:

    def __init__(self):

        # view the state batches
        self.visualize_input = False
        if self.visualize_input:
            self.viewer = CostmapVisualizer()

        # Hardcode input size and action size
        self.height = 84
        self.width = self.height
        self.depth = 4
        self.action_dim = 2
        self.action_bounds = [[0.5, 0.5],
                              [-0.5, -0.5]]

        # Initialize the current action and the old action for setting experiences
        self.old_action = np.ones(2, dtype='float')
        self.network_action = np.zeros(2, dtype='float')
        self.noise_action = np.zeros(2, dtype='float')
        self.action = np.zeros(2, dtype='float')

        # Initialize the grad inverter object
        self.grad_inv = GradInverter(self.action_bounds)

        # Initialize the old state
        self.old_state = np.zeros((self.width, self.height, self.depth), dtype='float')

        self.session = tf.Session()

        self.summary_op = tf.merge_all_summaries()
        self.summary_writer = tf.train.SummaryWriter('data')

        # Initialize actor and critic networks
        self.actor_network = ActorNetwork(self.height, self.action_dim, self.depth, self.session.graph,
                                          self.summary_writer, self.session)
        self.critic_network = CriticNetwork(self.height, self.action_dim, self.depth, self.session.graph,
                                            self.summary_writer, self.session)

        self.saver = tf.train.Saver(max_to_keep=3)
        self.save_path = os.path.join(os.path.dirname(__file__), os.pardir)+"/pre_trained_networks/my_model"

        # Should we load the pre-trained params
        if PRE_TRAINED_NETS:
            self.saver.restore(self.session, self.save_path)

        else:
            # Initialize the variables and restore pretrained ones
            self.session.run(tf.initialize_all_variables())
            self.critic_network.restore_pretrained_weights()
            self.actor_network.restore_pretrained_weights()

        self.summary_writer.add_graph(self.session.graph)

        # Initialize replay buffer (ring buffer with max length)
        self.replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

        # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
        self.exploration_noise = OUNoise(self.action_dim, MU, THETA, SIGMA)
        self.noise_flag = True

        # Initialize time step
        self.time_step = 0

        # Flag: don't learn the first experience
        self.first_experience = True

        # Are we saving a new initial buffer or loading an existing one or neither?
        self.save_initial_buffer = NEW_INITIAL_BUFFER
        if not self.save_initial_buffer:
            self.replay_buffer = pickle.load(open(os.path.expanduser('~')+"/Desktop/initial_replay_buffer.p", "rb"))
        else:
            self.replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    def train(self):

        if self.get_buffer_size() > REPLAY_START_SIZE:

            # Lets save a replay buffer with some initial experiences to start out with!
            if self.save_initial_buffer:
                self.save_buffer()
                self.save_initial_buffer = False

            # Sample a random minibatch of N transitions from replay buffer
            minibatch = random.sample(self.replay_buffer, BATCH_SIZE)

            state_batch = [data[0] for data in minibatch]
            action_batch = [data[1] for data in minibatch]
            reward_batch = [data[2] for data in minibatch]
            next_state_batch = [data[3] for data in minibatch]

            if self.visualize_input:
                state_batch_np = np.asarray(state_batch)

                # Scale up to grey scale again
                state_batch_np = np.multiply(state_batch_np, -100.0)
                state_batch_np = np.add(state_batch_np, 100.0)
                self.viewer.set_data(state_batch_np)
                self.viewer.run()
                self.visualize_input = False

            # Calculate y
            y_batch = []
            next_action_batch = self.actor_network.target_evaluate(next_state_batch)
            q_value_batch = self.critic_network.target_evaluate(next_state_batch, next_action_batch)

            for i in range(0, BATCH_SIZE):
                is_episode_finished = minibatch[i][4]
                if is_episode_finished:
                    y_batch.append([reward_batch[i]])
                else:
                    y_batch.append(reward_batch[i] + GAMMA * q_value_batch[i])

            self.critic_network.train(y_batch, state_batch, action_batch)

            # Update the actor policy using the sampled gradient:
            action_batch_for_gradients = self.actor_network.evaluate(state_batch)

            # Get the action gradient batch
            q_gradient_batch = self.critic_network.get_action_gradient(state_batch, action_batch_for_gradients)

            # Testing new gradient invert method
            q_gradient_batch = self.grad_inv.invert(q_gradient_batch, action_batch_for_gradients)

            self.actor_network.train(q_gradient_batch, state_batch)

            # Save model if necessary
            if self.time_step % 10000 == 0:

                # Append the step number to the checkpoint name:
                self.saver.save(self.session, self.save_path, global_step=self.time_step)

            # Update time step
            self.time_step += 1

    def get_action(self, state):

        # Select action a_t according to the current policy and exploration noise
        self.network_action = self.actor_network.get_action(state)
        self.noise_action = self.exploration_noise.noise()
        self.action = self.network_action

        if self.noise_flag:
            self.action += self.noise_action

        # Life q value output for this action and state
        self.print_q_value(state, self.action)

        # TODO: Should we clip or limit these values?
        return self.action

    def get_buffer_size(self):

        return len(self.replay_buffer)

    def set_experience(self, state, reward, is_episode_finished):

        if self.first_experience:
            self.first_experience = False
        else:
            # Store transition (s_t, a_t, r_t, s_{t+1}) in replay buffer
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
            "\tt: ", self.time_step

