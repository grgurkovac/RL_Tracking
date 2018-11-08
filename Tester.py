import tensorflow as tf
import numpy as np
import shutil
from Datasets.vot2017 import VOT2017
import sys
from RLT_model import *
import os
import cv2

def normalize(data, m=0.0, std=1.0):
    mean, var = tf.nn.moments(data, axes=[0])
    n_data = (data - mean) / (var + 1e-8)
    return n_data * (std + 1e-8) + m

def centralize(data, m=0):
    mean, _ = tf.nn.moments(data, axes=[0])
    n_data = (data - mean)
    return n_data + m


class Tester():
    def __init__(self, dataset, model, batch_size, time_steps=None, max_time_steps=None, experiment_name="test"):
        self.dataset = dataset
        self.model = model

        self.time_steps = time_steps
        self.max_time_steps = max_time_steps
        self.batch_size = batch_size

        self.gt_annots_ph = tf.placeholder(shape=[self.time_steps, self.dataset.action_dim], dtype=tf.float32, name="gt_annots")
        self.advantages = self.calculate_advantages(self.model.sampled_actions, self.gt_annots_ph)

        # self.advantages_ph = tf.placeholder(shape=[self.batch_size, self.time_steps], dtype=tf.float32, name="advantages")


        with tf.variable_scope("sq_loss"):
            self.sq_loss = tf.reduce_sum((self.model.means-self.gt_annots_ph)**2) # just for display
            tf.summary.scalar("sq_loss:", self.sq_loss)

        with tf.variable_scope("pseudo_loss"):
            self.pseudo_loss = (1.0/self.batch_size)*tf.reduce_sum(
                self.model.calculate_log_probs()*
                tf.stop_gradient(self.advantages)
            )

        self.global_step = tf.Variable(0, trainable=False)

        self.experiment_name = experiment_name
        self.experiment_dir=os.path.abspath(os.path.join("./", experiment_name))
        self.checkpoint_dir=os.path.abspath(os.path.join("./", experiment_name, "checkpoint/"))

        self.saver = tf.train.Saver()

        # compare: baseline, normalized advantages (advangtages (rtg), advantages diminished (rtg))

    def load(self, session):
        # incijalizacija parametara
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        latest_checkpint_path = tf.train.latest_checkpoint(self.checkpoint_dir)

        assert latest_checkpint_path is not None

        if latest_checkpint_path:
            print("Loading")
            self.saver.restore(session, latest_checkpint_path)
            begin_step = self.global_step.eval(session=session)

        tf.get_default_graph().finalize()
        return begin_step


    def train(self):
        # test_frames, test_annots = self.dataset.get_next_video_frames_and_annots(test=True, time_steps=time_steps) # full video

        # test_input_annot = np.zeros_like(test_annots)
        # test_input_annot[0][0] = test_annots[0][0]

        def resize_inputs(ims):
            def resize_input(im):
                # h, w, c = self.meta['inp_size']
                imsz = cv2.resize(im[0], (416, 416))
                imsz = np.reshape(imsz, [1, 416, 416, 3])
                return imsz
            return np.concatenate(
                tuple(
                    resize_input(im) for im in
                    np.split(ims, indices_or_sections=ims.shape[0], axis=0)), axis=0)


        # session_conf = tf.ConfigProto(
        #     intra_op_parallelism_threads=1,
        #     inter_op_parallelism_threads=1)

        with tf.Session().as_default() as sess:
            self.load(sess)

            test_frames, test_annots = self.dataset.get_next_video_frames_and_annots(test=True)
            test_frames, test_annots, test_padd_size = self.dataset.pad_video(frames=test_frames, annots=test_annots, time_steps=self.time_steps)
            test_frames = resize_inputs(test_frames)

            test_input_annot = np.zeros_like(test_annots)
            test_input_annot[0][0] = test_annots[0][0]

            test_frames_count = test_frames.shape[0]
            test_true_video_lenth = test_frames_count-test_padd_size
            chunk_loss = 0
            chunk_ps_loss = 0
            init_states = [np.zeros([2, 1, s]) for s in self.model.lstm_sizes]
            det_actions = np.empty(shape=(0, *test_annots.shape[1:]))

            for i in range(0, test_frames_count, self.time_steps):
                print(".", end='')
                sys.stdout.flush()
                # get video chunk
                test_frames_chunk = test_frames[i:i+self.time_steps]
                test_annots_chunk = test_annots[i:i+self.time_steps]
                test_input_annot_chunk = test_input_annot[i:i+self.time_steps]

                ps_l, sq_l, det_actions_chunk, global_step_evald, *init_states = sess.run(
                    [
                        self.pseudo_loss,
                        self.sq_loss,
                        self.model.deterministic_actions,
                        self.global_step
                    ]+list(self.model.last_states),
                    feed_dict={
                        **{
                            self.model.frames_ph: test_frames_chunk,
                            self.model.input_annot_ph: test_input_annot_chunk,
                            self.gt_annots_ph: test_annots_chunk,
                        },
                        **dict(zip(self.model.init_state_placeholders, init_states))
                    }
                )
                det_actions = np.concatenate((det_actions, det_actions_chunk), axis=0)

                chunk_loss += sq_l
                chunk_ps_loss += ps_l

            test_loss = chunk_loss/test_true_video_lenth
            test_pseudo_loss = chunk_ps_loss/test_true_video_lenth


            print()
            print("Test sq loss:", test_loss)
            print("Test pseudo loss:", test_pseudo_loss)

            self.dataset.animate(test_frames[:test_true_video_lenth], det_actions[:test_true_video_lenth])
            exit()


    def calculate_rewards(self, sampled_actions, annots):
        with tf.variable_scope("calculate_rewards"):
            # returns np array shape = (batch, time_steps)
            # sum over action space
            distances = tf.abs(sampled_actions-annots)
            # distances = (sampled_actions-annots)**2
            return -tf.reduce_mean(distances, axis=2) - tf.reduce_max(distances, axis=2)

    def calculate_baselines(self, rewards):
        with tf.variable_scope("calculate_baselines"):
            # returns shape(time_steps) from rewards shaped (batch, timesteps)
            return tf.reduce_mean(rewards, axis=0)

    def calculate_cumulative_rewards(self, rewards):
        with tf.variable_scope("calculate_cumulative_rewards"):
            # (batch, time_steps) -> (batch, time_steps) # equal over time steps
            rew = tf.tile(tf.reduce_sum(rewards, keepdims=True, axis=1), [1, self.time_steps])
            return normalize(rew)

    def calculate_rewards_to_go(self, rewards):
        with tf.variable_scope("calculate_rewards_to_go"):
            # (batch, time_steps) -> (batch, time_steps)

            def create_G(p_len, gamma):
                # gammas = [1, g, g**2, g**3 ,....., g**n], g = gamma
                gammas = [gamma ** i for i in range(p_len)]

                # G_ =
                #   [
                #       [1, g, g**2, g**3,,, g**(n-1)],
                #       [0, 1, g, g**2,...., g**(n-2)],
                #       [0, 0, 1, g,......., g**(n-3)],
                #           ..................
                #       [0,......................., 1]
                #   ].T
                #   , g = gamma, G = n x n

                G = tf.constant(np.array([np.pad(gammas, (i, 0), 'constant')[:p_len] for i in range(len(gammas))]).T,dtype=tf.float32)
                print(G)
                return G

        rew = tf.matmul(rewards, create_G(self.time_steps, 0.99))
        return rew

    def calculate_advantages(self, sampled_actions, annots):
        with tf.variable_scope("calculate_advantages"):
            rewards = self.calculate_rewards(sampled_actions, annots)
            # rew = self.calculate_cumulative_rewards(rewards) # batch_size, time_steps
            rew = self.calculate_rewards_to_go(rewards)

            # one baseline per time step
            # baselines = np.mean(cum_rewards, axis=0)  # time_steps

            advantages = normalize(rew)
            # advantages = centralize(rew)  #-baselines
            # advantages = (batch, time_steps)
            return advantages




if __name__ == "__main__":

    assert len(sys.argv) > 1
    experiment_name = sys.argv[1]

    dataset = VOT2017("./Datasets/vot2017/")
    time_steps = 100
    max_time_steps = 1500 # longest video in the dataset (girl)
    batch_size = 5
    model = RLT(max_time_steps=max_time_steps, time_steps=time_steps, batch_size=batch_size)
    trainer = Tester(dataset, model, batch_size=batch_size, time_steps=time_steps, max_time_steps=max_time_steps, experiment_name=experiment_name)

    trainer.train()
