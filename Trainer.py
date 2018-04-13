import tensorflow as tf
import numpy as np
from Datasets.vot2017 import *
from RLT_model import *
import os

def normalize(data, m=0.0, std=1.0):
    mean, var = tf.nn.moments(data, axes=[0])
    n_data = (data - mean) / (var + 1e-8)
    return n_data * (std + 1e-8) + m

class Trainer():
    def __init__(self, dataset, model, batch_size, time_steps, experiment_name="test"):
        self.dataset = dataset
        self.model = model

        self.time_steps = time_steps
        self.batch_size = batch_size

        self.gt_annots_ph = tf.placeholder(shape=[self.time_steps, self.dataset.action_dim], dtype=tf.float32, name="gt_annots")
        self.advantages = self.calculate_advantages(self.model.sampled_actions, self.gt_annots_ph)

        # self.advantages_ph = tf.placeholder(shape=[self.batch_size, self.time_steps], dtype=tf.float32, name="advantages")


        self.sq_loss = tf.reduce_mean((self.model.deterministic_actions-self.gt_annots_ph)**2) # just for display
        tf.summary.scalar("sq_loss:", self.sq_loss)

        self.pseudo_loss = (1.0/self.batch_size)*tf.reduce_sum(
            model.calculate_log_probs()*
            tf.stop_gradient(self.advantages)
        )

        with tf.control_dependencies(tf.get_collection("assert_ops")):
            self.optimize_step = tf.train.AdamOptimizer(0.001).minimize(-self.pseudo_loss)  # maximise

        # todo: setup summaries
        self.experiment_dir=os.path.abspath(os.path.join("./",experiment_name))
        self.train_writer = tf.summary.FileWriter('/'.join([self.experiment_dir,"train"]))
        self.test_writer = tf.summary.FileWriter('/'.join([self.experiment_dir,"test"]))
        self.merged = tf.summary.merge_all()
        # compare: baseline, normalized advantages (advangtages (rtg), advantages diminished (rtg))


    def train(self):
        test_frames, test_annots = self.dataset.get_n_frames_and_annots(self.time_steps, test=True) # full video

        test_input_annot = np.zeros_like(test_annots)
        test_input_annot[0][0] = test_annots[0][0]

        session_conf = tf.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1)

        with tf.Session(config=session_conf).as_default() as sess:
            sess.run(tf.global_variables_initializer())


            train_frames, train_annots = self.dataset.get_n_frames_and_annots(self.time_steps, test=False)
            for j in range(10000):
                print("j:", j)

                if train_frames.shape[0] != self.time_steps:
                    print("skipping sequence of lenth:",train_frames.shape)
                    continue


                train_input_annot = np.zeros_like(train_annots)
                train_input_annot[0][0] = train_annots[0][0]

                _, loss, train_summary = sess.run(
                    [self.optimize_step, self.sq_loss, self.merged],
                    feed_dict={
                        self.model.frames_ph: train_frames,
                        self.model.input_annot_ph: train_input_annot,
                        self.gt_annots_ph: train_annots,
                    }
                )
                self.train_writer.add_summary(train_summary, global_step=j)

                print("Train loss:", loss)
                if loss < 1e-5:
                    break

                test_loss, test_det_actions, loss, test_summary = sess.run(
                    [self.pseudo_loss, self.model.deterministic_actions, self.sq_loss, self.merged],
                    feed_dict={
                        self.model.frames_ph: test_frames,
                        self.model.input_annot_ph: test_input_annot,
                        self.gt_annots_ph: test_annots,
                    }
                )
                self.test_writer.add_summary(test_summary, global_step=j)
                print("Test loss:", loss)
                print()

            self.dataset.animate(test_frames, test_det_actions)

    def calculate_rewards(self, sampled_actions, annots):
        # returns np array shape = (batch, time_steps)
        # sum over action space
        distances = tf.abs(sampled_actions-annots)
        return -tf.reduce_mean(distances, axis=2) - tf.reduce_max(distances, axis=2)

    def calculate_cumulative_rewards(self, rewards):
        # (batch, time_steps) -> (batch, time_steps) # equal over time steps
        rew = tf.tile(tf.reduce_sum(rewards, keepdims=True, axis=1), [1, self.time_steps])
        return normalize(rew)

    def calculate_rewards_to_go(self, rewards):
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

            G = np.array([np.pad(gammas, (i, 0), 'constant')[:p_len] for i in range(len(gammas))]).T
            print(G)
            return G

        rew = np.matmul(rewards, create_G(self.time_steps, 0.99))
        return rew

    def calculate_advantages(self, sampled_actions, annots):
        rewards = self.calculate_rewards(sampled_actions, annots)
        rew = self.calculate_cumulative_rewards(rewards) # batch_size, time_steps
        # rew = self.calculate_rewards_to_go(rewards)

        # one baseline per time step
        # baselines = np.mean(cum_rewards, axis=0)  # time_steps

        advantages = normalize(rew)  #-baselines
        # advantages = (batch, time_steps)
        return advantages




if __name__ == "__main__":
    dataset = VOT2017("./Datasets/vot2017/")
    time_steps = 100
    batch_size = 5
    model = RLT(time_steps=time_steps, batch_size=batch_size)
    trainer = Trainer(dataset, model, batch_size=batch_size, time_steps=time_steps)

    trainer.train()
