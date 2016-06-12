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


# Hyper Parameters:
REPLAY_BUFFER_SIZE = 100000  # How big can the buffer get
REPLAY_START_SIZE = 5000     # When do we start training

BATCH_SIZE = 64              # How big are our batches

GAMMA = 0.99                 # Discount factor

MU = 0.0                     # Center value of noise
THETA = 0.1                  # Specifies how strong noise values are pulled towards mu
SIGMA = 0.1                  # Variance of noise


class DDPG:

    def __init__(self):

        # Hardcode input size and action size
        self.height = 84
        self.width = self.height
        self.depth = 4
        self.action_dim = 2
        self.action_bounds = [[1.0, 1.0],
                              [-1.0, -1.0]]

        # Initialize the current action and the old action for setting experiences
        self.old_action = np.ones(2, dtype='float')
        self.network_action = np.zeros(2, dtype='float')
        self.noise_action = np.zeros(2, dtype='float')
        self.action = np.zeros(2, dtype='float')

        # Initialize the grad inverter object
        self.grad_inv = GradInverter(self.action_bounds)

        # Initialize the old state
        self.old_state = np.zeros((self.width, self.height, self.depth), dtype='float')

        self.graph = tf.Graph()
        self.summary_op = tf.merge_all_summaries()
        self.summary_writer = tf.train.SummaryWriter('data', self.graph)

        # Initialize actor and critic networks
        self.actor_network = ActorNetwork(self.height, self.action_dim, self.depth, BATCH_SIZE, self.graph, self.summary_writer)
        self.critic_network = CriticNetwork(self.height, self.action_dim, self.depth, BATCH_SIZE, self.graph, self.summary_writer)

        # Initialize replay buffer (ring buffer with max length)
        self.replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

        # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
        self.exploration_noise = OUNoise(self.action_dim, MU, THETA, SIGMA)

        # Initialize time step
        self.time_step = 0

        # Flag: don't learn the first experience
        self.first_experience = True

        # Are we saving a new initial buffer or loading an existing one or neither?
        self.save_initial_buffer = False
        self.saved_buffer = True
        if self.saved_buffer:
            self.replay_buffer = pickle.load(open(os.path.dirname(__file__)+"/initial_replay_buffer.p", "rb"))
        else:
            self.replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    def train(self):

        if self.get_buffer_size() > REPLAY_START_SIZE:

            # Lets save a replay buffer with some initial experiences to start out with!
            if self.save_initial_buffer:
                self.save_buffer()
                self.save_initial_buffer = False

            if (self.time_step % 100) == 0:
                print("training step: ", self.time_step)
                print "network action:"
                print self.network_action
                print "noise_action:"
                print self.noise_action

            # Sample a random minibatch of N transitions from replay buffer
            minibatch = random.sample(self.replay_buffer, BATCH_SIZE)

            state_batch = [data[0] for data in minibatch]
            action_batch = [data[1] for data in minibatch]
            # action_batch = np.resize(action_batch, [BATCH_SIZE, 1])
            reward_batch = [data[2] for data in minibatch]
            next_state_batch = [data[3] for data in minibatch]

            # Calculate y
            y_batch = []
            next_action_batch = (self.actor_network.target_evaluate(next_state_batch))
            q_value_batch = self.critic_network.target_evaluate(next_state_batch, next_action_batch)

            # For debugging
            self.actor_network.evaluate(next_state_batch)

            for i in range(0, BATCH_SIZE):
                is_episode_finished = minibatch[i][4]
                if is_episode_finished:
                    y_batch.append([reward_batch[i]])
                else:
                    y_batch.append(reward_batch[i] + GAMMA * q_value_batch[i])

            self.critic_network.train(y_batch, state_batch, action_batch)

            # Update the actor policy using the sampled gradient:
            action_batch_for_gradients = self.actor_network.evaluate(state_batch)

            #q_gradient_batch = self.critic_network.get_action_gradient(state_batch, action_batch_for_gradients)

            # Testing new gradient invert method
            q_gradient_batch = self.grad_inv.invert(self.critic_network.get_action_gradient(state_batch,
                                                                                            action_batch_for_gradients),
                                                    action_batch_for_gradients)

            self.actor_network.train(q_gradient_batch, state_batch)

            # Update time step
            self.time_step += 1

    def get_action(self, state):

        # Select action a_t according to the current policy and exploration noise
        self.network_action = self.actor_network.get_action(state)
        self.noise_action = self.exploration_noise.noise()
        self.action = self.network_action + self.noise_action

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

        # Safe old state and old action for next experience
        self.old_state = state
        self.old_action = self.action

    def save_buffer(self):
        pickle.dump(self.replay_buffer, open(os.path.dirname(__file__)+"/initial_replay_buffer.p", "wb"))
