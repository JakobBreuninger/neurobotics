import tensorflow as tf


class GradInverter:

    def __init__(self, action_bounds):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session()
        
            self.action_size = len(action_bounds[0])
            self.action_input = tf.placeholder(tf.float32, [None, self.action_size])

            self.p_max = tf.constant(action_bounds[0], dtype=tf.float32)
            self.p_min = tf.constant(action_bounds[1], dtype=tf.float32)

            self.p_range = tf.constant([x - y for x, y in zip(action_bounds[0], action_bounds[1])], dtype=tf.float32)

            self.p_diff_max = tf.div(-self.action_input + self.p_max, self.p_range)
            self.p_diff_min = tf.div(self.action_input - self.p_min, self.p_range)

            self.zeros_act_grad_filter = tf.zeros([self.action_size])
            self.act_grad = tf.placeholder(tf.float32, [None, self.action_size])

            self.grad_inverter = tf.select(tf.greater(self.act_grad, self.zeros_act_grad_filter),
                                           tf.mul(self.act_grad, self.p_diff_max),
                                           tf.mul(self.act_grad, self.p_diff_min))
        
    def invert(self, grad, action):
        return self.sess.run(self.grad_inverter, feed_dict={self.action_input: action, self.act_grad: grad})