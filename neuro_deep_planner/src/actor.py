import tensorflow as tf
import numpy as np
import math


# Hyper Parameters
FULLY_LAYER1_SIZE = 200
FULLY_LAYER2_SIZE = 200

# params of conv nets
RECEIPTIVE_FIELD1 = 8
RECEIPTIVE_FIELD2 = 4
RECEIPTIVE_FIELD3 = 3
STRIDE1 = 4
STRIDE2 = 2
STRIDE3 = 1
FILTER1 = 32
FILTER2 = 32
FILTER3 = 32


LEARNING_RATE = 0.0001
DECAY = 0.001 # for target networks


class ActorNetwork:
    def __init__(self,image_size,action_size, image_no, batch_size):

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.InteractiveSession()

            # create actor network
            self.map_input,\
            self.weights_conv1,\
            self.biases_conv1,\
            self.weights_conv2,\
            self.biases_conv2,\
            self.weights_conv3,\
            self.biases_conv3,\
            self.weights_fully1,\
            self.biases_fully1,\
            self.weights_fully2,\
            self.biases_fully2,\
            self.weights_final,\
            self.biases_final,\
			self.action_output = self.create_network(image_size,action_size, image_no)

            # Create Exponential Moing Average Object
            ema_obj = tf.train.ExponentialMovingAverage(decay=DECAY)

            # get all the variables in the actor network
            with tf.variable_scope("actor") as scope:
                actor_variables = tf.get_collection(tf.GraphKeys.VARIABLES,
                scope=scope.name)

            # Create the shadow variables, and add ops to maintain moving averages
            # of actor network
            actor_ema = ema_obj.apply(actor_variables)

            # create target actor network
            self.map_input_target,\
            self.weights_conv1_target,\
            self.biases_conv1_target,\
            self.weights_conv2_target,\
            self.biases_conv2_target,\
            self.weights_conv3_target,\
            self.biases_conv3_target,\
            self.weights_fully1_target,\
            self.biases_fully1_target,\
            self.weights_fully2_target,\
            self.biases_fully2_target,\
            self.weights_final_target,\
            self.biases_final_target,\
            self.action_output_target = self.create_target_network(image_size,
                action_size,image_no, ema_obj,actor_variables)


            # define training rules
            self.q_gradient_input = tf.placeholder("float",[None,action_size])
            self.parameters = [self.weights_conv1,self.biases_conv1,\
                            self.weights_conv2,self.biases_conv2,\
                            self.weights_conv3,self.biases_conv3,\
                            self.weights_fully1,self.biases_fully1,\
                            self.weights_fully2,self.biases_fully2,\
                            self.weights_final,self.biases_final]



            self.parameters_gradients = tf.gradients(self.action_output,self.parameters,-self.q_gradient_input/batch_size)
            self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients,self.parameters))

            self.sess.run(tf.initialize_all_variables())



    def create_network(self,image_size,action_size, image_no):


        map_input = tf.placeholder("float",[None,image_size,image_size,image_no])

        # this must be adjudted if the conv network architecture is changed:
        conv3_output = 7*7*32

        with tf.variable_scope('actor'):
            weights_conv1 = self.variable([RECEIPTIVE_FIELD1, RECEIPTIVE_FIELD1, image_no, FILTER1],RECEIPTIVE_FIELD1*RECEIPTIVE_FIELD1)
            biases_conv1 = self.variable([FILTER1],RECEIPTIVE_FIELD1*RECEIPTIVE_FIELD1)

            weights_conv2 = self.variable([RECEIPTIVE_FIELD2, RECEIPTIVE_FIELD2, FILTER1, FILTER2],RECEIPTIVE_FIELD2*RECEIPTIVE_FIELD2*FILTER1)
            biases_conv2 = self.variable([FILTER2],RECEIPTIVE_FIELD2*RECEIPTIVE_FIELD2*FILTER1)

            weights_conv3 = self.variable([RECEIPTIVE_FIELD3, RECEIPTIVE_FIELD3, FILTER2, FILTER3],RECEIPTIVE_FIELD3*RECEIPTIVE_FIELD3*FILTER2)
            biases_conv3 = self.variable([FILTER3],RECEIPTIVE_FIELD3*RECEIPTIVE_FIELD3*FILTER2)

            weights_fully1 = self.variable([conv3_output, FULLY_LAYER1_SIZE],conv3_output)
            biases_fully1 = self.variable([FULLY_LAYER1_SIZE],conv3_output)

            weights_fully2 = self.variable([FULLY_LAYER1_SIZE, FULLY_LAYER2_SIZE],FULLY_LAYER1_SIZE)
            biases_fully2 = self.variable([FULLY_LAYER2_SIZE],FULLY_LAYER1_SIZE)

            weights_final = self.variable([FULLY_LAYER2_SIZE, action_size],FULLY_LAYER2_SIZE)
            biases_final = self.variable([action_size],FULLY_LAYER2_SIZE)


        # reshape image to apply convolution
        #input_image = tf.reshape(input_array, [-1,image_size,image_size,1])

        # 3 convolutional layers
        conv1 = tf.nn.relu(tf.nn.conv2d(map_input, weights_conv1,
                    strides=[1, STRIDE1, STRIDE1, 1], padding='VALID')
                    + biases_conv1)
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights_conv2,
                    strides=[1, STRIDE2, STRIDE2, 1], padding='VALID')
    	            + biases_conv2)
        conv3 = tf.nn.relu(tf.nn.conv2d(conv2, weights_conv3,
                    strides=[1, STRIDE3, STRIDE3, 1], padding='VALID')
    	            + biases_conv3)

        # reshape output tensor to a rank 1 tensor
        conv3_flat = tf.reshape(conv3, [-1, conv3_output])

        # more operations
        fully1 = tf.nn.relu(tf.matmul(conv3_flat, weights_fully1) + biases_fully1)
        fully2 = tf.nn.relu(tf.matmul(fully1, weights_fully2) + biases_fully2)
        action_output= tf.matmul(fully2, weights_final) + biases_final

        # return all ops
        return map_input,weights_conv1,biases_conv1,weights_conv2,\
            biases_conv2,weights_conv3,biases_conv3, weights_fully1,\
            biases_fully1,weights_fully2, biases_fully2, weights_final,\
            biases_final,action_output

    def create_target_network(self,image_size,action_size,image_no,ema_obj,actor_variables):

        map_input = tf.placeholder("float",[None,image_size,image_size,image_no])

        # this must be adjudted if the conv network architecture is changed:
        conv3_output = 7*7*32

        with tf.variable_scope('actor_target'):
            weights_conv1 = ema_obj.average(actor_variables[0])
            biases_conv1 = ema_obj.average(actor_variables[1])
            weights_conv2 = ema_obj.average(actor_variables[2])
            biases_conv2 = ema_obj.average(actor_variables[3])
            weights_conv3 = ema_obj.average(actor_variables[4])
            biases_conv3 = ema_obj.average(actor_variables[5])
            weights_fully1 = ema_obj.average(actor_variables[6])
            biases_fully1 = ema_obj.average(actor_variables[7])
            weights_fully2 = ema_obj.average(actor_variables[8])
            biases_fully2 = ema_obj.average(actor_variables[9])
            weights_final = ema_obj.average(actor_variables[10])
            biases_final = ema_obj.average(actor_variables[11])


        # reshape image to apply convolution
        #input_image = tf.reshape(input_array, [-1,image_size,image_size,1])

        # 3 convolutional layers
        conv1 = tf.nn.relu(tf.nn.conv2d(map_input, weights_conv1,
                    strides=[1, STRIDE1, STRIDE1, 1], padding='VALID')
                    + biases_conv1)
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights_conv2,
                    strides=[1, STRIDE2, STRIDE2, 1], padding='VALID')
    	            + biases_conv2)
        conv3 = tf.nn.relu(tf.nn.conv2d(conv2, weights_conv3,
                    strides=[1, STRIDE3, STRIDE3, 1], padding='VALID')
    	            + biases_conv3)



        # reshape output tensor to a rank 1 tensor
        conv3_flat = tf.reshape(conv3, [-1, conv3_output])

        # more operations
        fully1 = tf.nn.relu(tf.matmul(conv3_flat, weights_fully1) + biases_fully1)
        fully2 = tf.nn.relu(tf.matmul(fully1, weights_fully2) + biases_fully2)
        action_output= tf.matmul(fully2, weights_final) + biases_final

        # return all ops
        return map_input,weights_conv1,biases_conv1,weights_conv2,\
            biases_conv2,weights_conv3,biases_conv3, weights_fully1,\
            biases_fully1,weights_fully2, biases_fully2, weights_final,\
            biases_final,action_output







    def train(self,q_gradient_batch,state_batch):
        self.sess.run(self.optimizer,feed_dict={
            self.q_gradient_input:q_gradient_batch,
            self.input_map:state_batch
            })



    def get_action(self,state):
        return self.sess.run(self.action_output,feed_dict={
            self.input_map:[state]
            })[0]

    def evaluate(self,state_batch):
        return self.sess.run(self.action_output,feed_dict={
            self.input_map:state_batch
            })


    def target_evaluate(self,state_batch):
        return self.sess.run(self.action_output_target,feed_dict={
            self.input_map_input:state_batch
            })



    # f fan-in size
    def variable(self,shape,f):
    	return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))
