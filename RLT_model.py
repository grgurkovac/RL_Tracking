import tensorflow as tf
import numpy as np

class RLT:

    def __init__(self, time_steps, batch_size, observation_size=18, annotation_size=8):

        self.time_steps = time_steps
        self.batch_size = batch_size

        self.observation_size = observation_size
        self.annotations_size = annotation_size

        self.frames_ph = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 3), name="frames")
        self.input_annot_ph = tf.placeholder(dtype=tf.float32, shape=(None, 8), name="input_annot")
        self.sampled_actions_ph = tf.placeholder(dtype=tf.float32, shape=(None, None, 8), name="sampled_actions")

        # forward_pass
        self.observation = self.observation_network(self.frames_ph)
        self.means = self.reccurent_network(self.observation)
        self.sampled_actions = self.sample_actions(self.means, N=self.batch_size)
        self.deterministic_actions = self.means



    def fake_roi_pooling(self, tensor, output_dims=(8, 8)):

        # izvuci dimenzije slike
        height = tf.shape(tensor)[1]
        width = tf.shape(tensor)[2]
        input_channels=tensor.get_shape()[-1]

        # dim % roi_pool_size = rest
        h_rest = tf.floormod(height, output_dims[0])
        w_rest = tf.floormod(width, output_dims[1])

        # padd = roi_pool_size - rest
        h_padd = tf.sign(h_rest)*output_dims[0]-h_rest # if 0 no change
        w_padd = tf.sign(w_rest)*output_dims[1]-w_rest

        # padd image
        h_div = tf.floordiv(h_padd, 2)
        h_mod = tf.mod(h_padd, 2)
        h_paddings = [h_div, h_div+h_mod]

        w_div = tf.floordiv(w_padd,2)
        w_mod = tf.mod(w_padd, 2)
        w_paddings = [w_div, w_div+w_mod]

        paddings = tf.convert_to_tensor([[0,0], h_paddings, w_paddings, [0,0]])
        tensor = tf.pad(tensor, paddings=paddings, constant_values=-np.inf)

        splits = tf.split(tensor, output_dims[0], axis=1)
        polled = []

        for hsplit in splits:
            wsplits = tf.split(hsplit, output_dims[1], axis=2)

            pooled = [tf.reduce_max(wsplit, axis=(1, 2), keepdims=True) for wsplit in wsplits]
            polled_w = tf.concat(pooled, axis=2)

            polled.append(polled_w)

        tensor = tf.concat(polled, axis=1)

        return tf.reshape(tensor, [-1, output_dims[0], output_dims[1], input_channels])


    def observation_network(self, tensor):
        with tf.variable_scope("Observation_network"):

            tensor = tf.layers.conv2d(tensor, 20, 5, 2, padding='SAME', name="conv1", kernel_initializer=tf.initializers.zeros)
            tensor = tf.nn.max_pool(tensor, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME", name="mp1")
            tensor = tf.layers.conv2d(tensor, 10, 3, 2, padding='SAME', name="conv2", kernel_initializer=tf.initializers.zeros)
            tensor = tf.layers.conv2d(tensor, 10, 3, 2, padding='SAME', name="conv3", kernel_initializer=tf.initializers.zeros)

            tensor = self.fake_roi_pooling(tensor, output_dims=(8, 8))

            tensor = tf.layers.flatten(tensor)
            tensor = tf.layers.dense(tensor, 10, activation=tf.nn.relu)
            tensor = tf.layers.dense(tensor, self.observation_size-self.annotations_size)

            tensor = tf.concat([tensor, self.input_annot_ph], axis=-1)
            return tensor


    def reccurent_network(self, tensor):

        lstm_size = 8
        # time_steps_tensorst  = tf.expand_dims(tensor, 0).shape # shape is (1, time_steps, observation_size)
        time_steps_tensors = tf.split(tensor, self.time_steps, axis=0)

        def lstm_cell(size):
            return tf.contrib.rnn.BasicLSTMCell(size)

        stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell(s) for s in [100,50,10,8]])
        # outputs, states = tf.nn.dynamic_rnn(stacked_lstm, time_steps_tensors,  dtype=tf.float32)

        # lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        outputs, states = tf.nn.static_rnn(stacked_lstm, time_steps_tensors,  dtype=tf.float32)

        outputs = tf.concat(
            outputs,
            axis=0
        )

        assert_op = tf.assert_equal(tf.shape(outputs), [self.time_steps, lstm_size], data=[tf.shape(outputs)], summarize=4)
        tf.add_to_collection("assert_ops", assert_op)

        return outputs


    def sample_actions(self, means, N):
        # create N random episodes, based on means
        Z = tf.random_uniform(shape=[N, self.time_steps, self.annotations_size])  # N samples
        sigmas = tf.get_variable(name="sigmas", shape=self.annotations_size, initializer=tf.constant_initializer(0.001))
        actions = means + Z*sigmas
        return actions

    def calculate_log_probs(self):
        # sum over actions
        return -tf.reduce_sum((tf.stop_gradient(self.sampled_actions) - self.means)**2, axis=2)


