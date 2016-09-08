import tensorflow as tf
import numpy as np


# Parameters:
NUM_EXPERIENCES = 100	    # How many experiences stored per file
MIN_FILE_NUM = 10            # How many files at minimum do we need for training
PRE_STORED_DATA_FILES = 0   # How many files are already stored in our Data folder

MIN_FILES_IN_QUEUE = 10     # Files are added when this Number is reached
NEW_FILES_TO_ADD = 200      # How many files are added to the fifo file queue


class DataManager:

    def __init__(self, batch_size, p_experience_path, session):
        self.graph = session.graph
        with self.graph.as_default():

            # Make sure the directory for the data files exists
            if not tf.gfile.Exists(p_experience_path):
                tf.gfile.MakeDirs(p_experience_path)

            # initialize session
            self.sess = session

            # set batch size
            self.batch_size = batch_size

            self.experience_path = p_experience_path

            # set file counter to number of pre-stored files
            self.file_counter = PRE_STORED_DATA_FILES

            # set experience counter to 0 (will be increased until number of experiences per file is reached)
            self.experience_counter = 0

            # initialize FIFO queue for file names
            self.filename_queue = tf.FIFOQueue(capacity=1000, dtypes=[tf.string])

            # enqueue filename operation with placeholder
            self.filename_placeholder = tf.placeholder(tf.string)
            self.enqueue_op = self.filename_queue.enqueue(self.filename_placeholder)

            # operation to return size of the experience file queue
            self.filename_queue_size = self.filename_queue.size()

            # put prexisiting files into the fifo filename queue
            if PRE_STORED_DATA_FILES > 0:
                self.enqueue_prestored_experiences()

            # these variables are queried from outside
            self.state_batch, \
                self.action_batch, \
                self.reward_batch, \
                self.next_state_batch, \
                self.is_episode_finished_batch = self.build_next_batch_op()

            # filepath to stored files
            self.filename = self.experience_path + '/data_' + str(self.file_counter) + '.tfrecords'

            # init the write that writes the tfrecords
            self.writer = tf.python_io.TFRecordWriter(self.filename)

    # checks if training can start
    def enough_data(self):
        # if 1000 experiences have been acquired, start training
        if self.file_counter >= MIN_FILE_NUM:
            return True
        else:
            return False

    # builds the pipeline how random batches are generated from the experience files on the hard drive
    def build_next_batch_op(self):
        reader = tf.TFRecordReader()

        _, serialized_experience = reader.read(self.filename_queue)

        # decode feature example
        features = tf.parse_single_example(serialized_experience, features={
            'state': tf.FixedLenFeature([], tf.string),
            'action': tf.FixedLenFeature([2], tf.float32),
            'reward': tf.FixedLenFeature([], tf.float32),
            'next_state': tf.FixedLenFeature([], tf.string),
            'is_episode_finished': tf.FixedLenFeature([], tf.int64)})

        state = tf.decode_raw(features['state'], tf.uint8)
        state.set_shape([86*86*4])
        action = features['action']
        reward = features['reward']
        next_state = tf.decode_raw(features['next_state'], tf.uint8)
        next_state.set_shape([86*86*4])
        is_episode_finished = features['is_episode_finished']

        # reshape gridmaps
        state = tf.reshape(state, [86, 86, 4])
        next_state = tf.reshape(next_state, [86, 86, 4])

        # batch shuffling is done in a seperate thread
        state_batch, action_batch, reward_batch, next_state_batch, is_episode_finished_batch = tf.train.shuffle_batch(
            [state, action, reward, next_state, is_episode_finished], batch_size=self.batch_size, capacity=1000,
            min_after_dequeue=0)

        return state_batch, action_batch, reward_batch, next_state_batch, is_episode_finished_batch

    # output the next random batch
    def get_next_batch(self):
        state_batch, \
            action_batch, \
            reward_batch, \
            next_state_batch, \
            is_episode_finished_batch = self.sess.run(([self.state_batch,
                                                        self.action_batch,
                                                        self.reward_batch,
                                                        self.next_state_batch,
                                                        self.is_episode_finished_batch]))
        return state_batch, action_batch, reward_batch, next_state_batch, is_episode_finished_batch

    # enqueue the number of prestored experience files specified in PRE_STORED_DATA_FILES
    def enqueue_prestored_experiences(self):
        for i in range(0, PRE_STORED_DATA_FILES):
            filename = self.experience_path + '/data_' + str(i) + '.tfrecords'

            self.sess.run(self.enqueue_op, feed_dict={self.filename_placeholder: filename})
            print i

        print "enqueued " + str(PRE_STORED_DATA_FILES) + " filenames to filename queue"

    # checks if new files need to be enqueued to the fifo file queue
    def check_for_enqueue(self):
        if self.sess.run(self.filename_queue_size) < MIN_FILES_IN_QUEUE and self.file_counter >= 1:
            print "enqueuing files"
            if self.file_counter > 0:
                random_array = np.random.randint(0, high=self.file_counter, size=NEW_FILES_TO_ADD)
            else:
                random_array = np.zeros(NEW_FILES_TO_ADD, dtype=np.int8)

            for i in range(NEW_FILES_TO_ADD):
                filename = self.experience_path + '/data_' + str(random_array[i]) + '.tfrecords'
                self.sess.run(self.enqueue_op, feed_dict={self.filename_placeholder: filename})

    # stores experiences sequentially to files
    def store_experience_to_file(self, state, action, reward, next_state, is_episode_finished):
        # write experience to file (s_t, a_t, r_t, s_{t+1})

        state_raw = state.tostring()
        next_state_raw = next_state.tostring()

        # build example
        example = tf.train.Example(features=tf.train.Features(feature={
            'state': tf.train.Feature(bytes_list=tf.train.BytesList(value=[state_raw])),
            'action': tf.train.Feature(float_list=tf.train.FloatList(value=action.tolist())),
            'reward': tf.train.Feature(float_list=tf.train.FloatList(value=[reward])),
            'next_state': tf.train.Feature(bytes_list=tf.train.BytesList(value=[next_state_raw])),
            'is_episode_finished': tf.train.Feature(int64_list=tf.train.Int64List(value=[is_episode_finished]))
        }))

        self.writer.write(example.SerializeToString())

        self.experience_counter += 1

        # store NUM_EXPERIENCES experiences per file
        if self.experience_counter == NUM_EXPERIENCES:

            # write the file to hdd
            self.writer.close()

            # update counters
            self.experience_counter = 0
            self.file_counter += 1

            # create new file
            self.filename = self.experience_path + '/data_' + str(self.file_counter) + '.tfrecords'
            self.writer = tf.python_io.TFRecordWriter(self.filename)