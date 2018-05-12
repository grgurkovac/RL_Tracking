import tensorflow as tf
import numpy as np

class RLT:

    def __init__(self, batch_size, observation_size=1000, annotation_size=8, time_steps=None, max_time_steps=None):

        self.time_steps = time_steps
        self.max_time_steps = time_steps
        self.batch_size = batch_size
        self.lstm_sizes = [2000, 1000]


        self.observation_size = observation_size
        self.annotations_size = annotation_size

        self.frames_ph = tf.placeholder(dtype=tf.float32, shape=(self.time_steps, None, None, 3), name="frames")
        self.input_annot_ph = tf.placeholder(dtype=tf.float32, shape=(self.time_steps, 8), name="input_annot")
        self.sampled_actions_ph = tf.placeholder(dtype=tf.float32, shape=(None, None, 8), name="sampled_actions")

        # forward_pass
        self.observation = self.observation_network(self.frames_ph)
        self.means = self.reccurent_network(self.observation)
        self.sampled_actions = self.sample_actions(self.means, N=self.batch_size)
        self.deterministic_actions = self.means

        # self.means = tf.Print(self.means, [tf.gradients(self.means, self.observation)], message="observation")



    def fake_roi_pooling(self, tensor, output_dims=(8, 8)):

        with tf.variable_scope("fake_roi_pooling"):
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

            def YOLO_extractor(tensor):

                def build_from_pb(protobuf_file=r"./yolo/yolov2-tiny-voc.pb"):
                    with tf.gfile.FastGFile(protobuf_file, "rb") as f:

                        graph_def = tf.GraphDef()
                        graph_def.ParseFromString(f.read())

                    tf.import_graph_def(
                        graph_def,
                        name=""
                    )
                    # with open(self.FLAGS.metaLoad, 'r') as fp:
                    #     self.meta = json.load(fp)
                    # self.framework = create_framework(self.meta, self.FLAGS)

                    # Placeholders
                    inp = tf.get_default_graph().get_tensor_by_name('input:0')
                    out = tf.get_default_graph().get_tensor_by_name('output:0')

                    ops = [op.name for op in tf.get_default_graph().get_operations() if "leaky" in op.name]
                    last_leaky = ops[-2]
                    print("ll:", last_leaky)

                    op_dict = {op.name: op for op in tf.get_default_graph().get_operations() if "leaky" in op.name}
                    op = op_dict[last_leaky]
                    output = op.outputs[0]

                    return inp, output

                inp, out = build_from_pb()
                self.frames_ph = inp

                return out

            # tensor = tf.layers.conv2d(tensor, 10, 5, 2, padding='SAME', name="conv1", kernel_initializer=tf.initializers.zeros)
            # tensor = tf.nn.max_pool(tensor, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME", name="mp1")
            # tensor = tf.layers.conv2d(tensor, 20, 3, 2, padding='SAME', name="conv2", kernel_initializer=tf.initializers.zeros)
            # tensor = tf.layers.conv2d(tensor, 30, 3, 2, padding='SAME', name="conv3", kernel_initializer=tf.initializers.zeros)
            #
            # tensor = self.fake_roi_pooling(tensor, output_dims=(8, 8))
            #
            # tensor = tf.layers.flatten(tensor)

            tensor = YOLO_extractor(tensor=tensor)
            tensor = tf.layers.flatten(tensor)

            tensor = tf.stop_gradient(tensor) # dont train yolo

            tensor = tf.concat([tensor, self.input_annot_ph], axis=-1)
            obs = tf.layers.dense(tensor, self.observation_size-self.annotations_size)

            return obs


    def reccurent_network(self, tensor):
        with tf.variable_scope("Reccurent_network"):

            time_steps_tensors = tf.expand_dims(tensor, 0) # shape is (1, time_steps, observation_size)
            # time_steps_tensors = tf.split(tensor, self.time_steps, axis=0)

            def lstm_cell(size):
                return tf.contrib.rnn.BasicLSTMCell(size)

            # self.init_state_placeholders = [tf.placeholder(tf.float32, [2, 1, s]) for s in [50,50,10,8]]
            self.init_state_placeholders = [tf.placeholder(tf.float32, [2, 1, s]) for s in self.lstm_sizes]

            init_states = tuple([tf.nn.rnn_cell.LSTMStateTuple(state_ph[0], state_ph[1]) for state_ph in self.init_state_placeholders])

            # stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell(s) for s in [50,50,10,8]])
            stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell(s) for s in self.lstm_sizes])
            outputs, self.last_states = tf.nn.dynamic_rnn(stacked_lstm, time_steps_tensors, initial_state=init_states, dtype=tf.float32)

            # lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
            # outputs, states = tf.nn.static_rnn(stacked_lstm, time_steps_tensors,  dtype=tf.float32)

            # outputs = tf.concat(
            #     outputs,
            #     axis=0
            # )
            # outputs = tf.Print(outputs, [tf.gradients(outputs, time_steps_tensors)], message="rec_grads:")
            outputs= tf.squeeze(outputs) # one video for now
            outputs = outputs[:, -8:]

            assert_op = tf.assert_equal(tf.shape(outputs), [self.time_steps, 8], data=[tf.shape(outputs)], summarize=4)
            tf.add_to_collection("assert_ops", assert_op)

            return outputs


    def sample_actions(self, means, N):
        # create N random episodes, based on means
        Z = tf.random_uniform(shape=[N, means.shape[0].value, means.shape[1].value]) - 0.5 # N samples
        self.sigmas = tf.get_variable(name="sigmas", shape=self.annotations_size, initializer=tf.constant_initializer(0.001), trainable=False)
        actions = means + Z*(self.sigmas)**2
        return actions

    def calculate_log_probs(self):
        # sum over actions
        # sampled_action = (batch_size, time_steps, annotation_size)
        # means = (batch_size, time_steps, annotation_size)
        # sigmas = (annotation_size)
        # return -tf.reduce_sum((tf.stop_gradient(self.sampled_actions) - self.means)**2, axis=2)
        return -tf.reduce_sum(
            (tf.stop_gradient(self.sampled_actions) - self.means)**2
            / (2*(self.sigmas**2)),
            axis=2)


