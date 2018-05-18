import tensorflow as tf
import numpy as np
import shutil
from functools import lru_cache
import psutil
from Datasets.dataset import Dataset
import sys
from RLT_model import *
import os
import cv2

def normalize(data, m=0.0, std=1.0):
    mean, var = tf.nn.moments(data, axes=[0])
    n_data = (data - mean) / (var + 1e-8)
    return n_data * (std + 1e-8) + m

def centralize(data):
    mean, _ = tf.nn.moments(data, axes=[0])
    c_data = (data - mean)
    return c_data

def center_distance(R1, R2):
    x11, y11, w1, h1 = tf.split(R1, 4, axis=1)
    x1c, y1c = x11 + w1//2, y11 + h1//2

    x21, y21, w2, h2 = tf.split(R2, 4, axis=1)
    x2c, y2c = x21 + w2//2, y21 + h2//2

    C1 = tf.concat((x1c, y1c), axis=1)
    C2 = tf.concat((x2c, y2c), axis=1)

    return tf.reduce_mean(tf.abs(C1-C2))


def IOU(R1, R2):

    # R1 = tf.Print(R1, [R1[0][0]], summarize=10, message="R1")
    # R2 = tf.Print(R2, [R2[0]], summarize=10, message="R2")

    x11, y11, w1, h1 = tf.split(R1, 4, axis=2)
    x12, y12 = x11 + w1, y11 + h1

    x21, y21, w2, h2 = tf.split(R2, 4, axis=1)
    x22, y22 = x21 + w2, y21 + h2

    xI1 = tf.maximum(x11, x21)
    xI2 = tf.minimum(x12, x22)
    wI = tf.maximum(xI2 - xI1, 0)

    yI1 = tf.maximum(y11, y21)
    yI2 = tf.minimum(y12, y22)
    hI = tf.maximum(yI2 - yI1, 0)

    intersection = wI * hI
    union = (w1 * h1 + w2 * h2) - intersection

    IOU = intersection / union

    # remove last dim
    IOU = tf.squeeze(IOU, axis=2)

    return IOU


class Trainer():
    def __init__(self, dataset, model, batch_size, time_steps=None, max_time_steps=None, experiment_name="test"):
        self.dataset = dataset
        self.model = model

        self.time_steps = time_steps
        self.max_time_steps = max_time_steps
        self.batch_size = batch_size

        self.epoch = tf.placeholder(tf.float32, name="epoch")
        tf.summary.scalar("epoch", self.epoch)

        self.gt_annots_ph = tf.placeholder(shape=[self.time_steps, self.dataset.action_dim], dtype=tf.float32, name="gt_annots")
        self.advantages = self.calculate_advantages(self.model.sampled_actions, self.gt_annots_ph)

        # self.advantages_ph = tf.placeholder(shape=[self.batch_size, self.time_steps], dtype=tf.float32, name="advantages")


        with tf.variable_scope("sq_loss"):
            self.sq_loss = tf.reduce_mean((self.model.means-self.gt_annots_ph)**2)  # just for display
            tf.summary.scalar("sq_loss:", self.sq_loss)

        with tf.variable_scope("center_distance"):
            self.center_distance = center_distance(self.model.means, self.gt_annots_ph)  # just for display
            tf.summary.scalar("center_distance:", self.center_distance)


        with tf.variable_scope("pseudo_loss"):
            self.pseudo_loss = (1.0/self.batch_size)*tf.reduce_sum(
                self.model.calculate_log_probs()*
                tf.stop_gradient(self.advantages)
            )

        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.00001
        self.learning_rate = tf.maximum(tf.train.exponential_decay(
            starter_learning_rate, self.global_step, decay_steps=300, decay_rate=0.99, staircase=True), 1e-6)
        tf.summary.scalar("learning rate", self.learning_rate)

        with tf.control_dependencies(tf.get_collection("assert_ops")):
            self.optimize_step = tf.train.AdamOptimizer(self.learning_rate).minimize(-self.pseudo_loss, global_step=self.global_step)  # maximise
            # self.optimize_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.sq_loss, global_step=self.global_step)  # maximise


        self.experiment_name = experiment_name
        self.experiment_dir = os.path.abspath(os.path.join("./experiments/", experiment_name))
        self.checkpoint_dir = os.path.abspath(os.path.join("./experiments/", experiment_name, "checkpoint/"))

        self.train_dir = '/'.join([self.experiment_dir, "summaries/train"])
        self.train_writer = tf.summary.FileWriter(self.train_dir)
        self.train_writer.add_graph(graph=tf.get_default_graph())

        self.test_dir = '/'.join([self.experiment_dir, "summaries/test"])
        self.test_writer = tf.summary.FileWriter(self.test_dir)
        self.merged = tf.summary.merge_all()

        self.saver = tf.train.Saver()

        # compare: baseline, normalized advantages (advangtages (rtg), advantages diminished (rtg))

    def load_or_initialize(self, session):
        # incijalizacija parametara
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        latest_checkpint_path = tf.train.latest_checkpoint(self.checkpoint_dir)

        if latest_checkpint_path and input("load " + latest_checkpint_path + " ? y/n ") == "y":
                print("Loading")
                self.saver.restore(session, latest_checkpint_path)
                begin_step = self.global_step.eval(session=session)

        else:

            session.run(tf.global_variables_initializer())
            begin_step = 0

        tf.get_default_graph().finalize()
        return begin_step


    def train(self):
        # test_frames, test_annots = self.dataset.get_next_video_frames_and_annots(test=True, time_steps=time_steps) # full video

        # test_input_annot = np.zeros_like(test_annots)
        # test_input_annot[0][0] = test_annots[0][0]


        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8

        # train_frames, train_annots = self.dataset.get_next_video_frames_and_annots(test=False)
        with tf.Session(config=config).as_default() as sess:
            self.load_or_initialize(sess)

            for j in range(10000):
                print("j:", j)
                print("epoch:", self.dataset.epoch)

                train_frames, train_annots = self.dataset.get_next_video_frames_and_annots(test=False)
                train_frames = self.model.resize_inputs(train_frames)

                print("memory used:", psutil.virtual_memory().used // 1024 ** 2)

                # initial annotation to be fed to the network
                train_input_annot = np.zeros_like(train_annots)
                train_input_annot[0][0] = train_annots[0][0]

                train_frames_count = len(train_frames)
                # train_true_video_len = train_frames_count-train_padd_size
                train_true_video_len = train_frames_count
                chunk_loss = 0
                init_states = [np.zeros([2, 1, s]) for s in self.model.lstm_sizes]

                for i in range(0, train_frames_count, self.time_steps):
                    # get video chunk
                    train_frames_chunk = np.array(train_frames[i:i+self.time_steps])
                    train_annots_chunk = np.array(train_annots[i:i+self.time_steps])
                    train_input_annot_chunk = np.array(train_input_annot[i:i+self.time_steps])

                    train_frames_chunk, train_annots_chunk, train_input_annot_chunk, train_padd_size = self.dataset.pad_video(
                        frames=train_frames_chunk,
                        annots=train_annots_chunk,
                        input_annot=train_input_annot_chunk,
                        time_steps=self.time_steps
                    )

                    # _, ps_l, sq_l, train_summary, lr, det_actions, global_step_evald, *init_states = sess.run(
                    _, ps_l, sq_l, c_dist, lr, det_actions, global_step_evald, *init_states = sess.run(
                        [
                            self.optimize_step,
                            self.pseudo_loss,
                            self.sq_loss,
                            self.center_distance,
                            # self.merged,
                            self.learning_rate,
                            self.model.deterministic_actions,
                            self.global_step,
                        ]+list(self.model.last_states),
                        feed_dict={
                            **{
                            self.model.frames_ph: train_frames_chunk,
                            self.gt_annots_ph: train_annots_chunk,
                            self.model.input_annot_ph: train_input_annot_chunk,
                            self.epoch: self.dataset.epoch,
                            # self.epoch: j,
                            },
                            **dict(zip(self.model.init_state_placeholders, init_states))
                        }
                    )

                    chunk_loss += sq_l
                    # self.train_writer.add_summary(train_summary, global_step=global_step_evald)



                print()
                loss = chunk_loss/((train_true_video_len // self.time_steps) + 1)
                print("lr:", lr, " loss:", loss)
                print()

                # if (loss) < 0.002:
                #     self.dataset.animate(train_frames_chunk, det_actions)
                #     exit()
                # print("Train loss:", loss, " lr:", lr)


                if j % 10 == 0:

                    # checkpoint
                    print("saving")
                    print(self.checkpoint_dir+self.experiment_name)
                    self.saver.save(sess, self.checkpoint_dir+"/"+self.experiment_name, global_step=self.global_step)

                    test_frames, test_annots = self.dataset.get_next_video_frames_and_annots(test=True)
                    test_frames = self.model.resize_inputs(test_frames)

                    test_input_annot = np.zeros_like(test_annots)
                    test_input_annot[0][0] = test_annots[0][0]

                    test_frames_count = len(test_frames)
                    test_true_video_len = test_frames_count
                    chunk_loss = 0
                    init_states = [np.zeros([2, 1, s]) for s in self.model.lstm_sizes]

                    for i in range(0, test_frames_count, self.time_steps):
                        print(".", end='')
                        sys.stdout.flush()
                        # get video chunk
                        test_frames_chunk = np.array(test_frames[i:i+self.time_steps])
                        test_annots_chunk = np.array(test_annots[i:i+self.time_steps])
                        test_input_annot_chunk = np.array(test_input_annot[i:i+self.time_steps])

                        test_frames_chunk, test_annots_chunk, test_input_annot_chunk, test_padd_size = self.dataset.pad_video(
                            frames=test_frames_chunk, annots=test_annots_chunk, input_annot=test_input_annot_chunk, time_steps=self.time_steps)

                        ps_l, sq_l, c_dist, test_summary, lr, det_actions, global_step_evald, *init_states = sess.run(
                            [
                                self.pseudo_loss,
                                self.sq_loss,
                                self.center_distance,
                                self.merged,
                                self.learning_rate,
                                self.model.deterministic_actions,
                                self.global_step
                            ]+list(self.model.last_states),
                            feed_dict={
                                **{
                                    self.epoch: self.dataset.epoch,
                                    self.model.frames_ph: test_frames_chunk,
                                    self.model.input_annot_ph: test_input_annot_chunk,
                                    self.gt_annots_ph: test_annots_chunk
                                },
                                **dict(zip(self.model.init_state_placeholders, init_states))
                            }
                        )

                        chunk_loss += sq_l

                    test_loss = chunk_loss/(test_true_video_len)
                    self.test_writer.add_summary(test_summary, global_step=global_step_evald)

                    print()
                    print("Test loss:", test_loss)
                    print()


    def calculate_rewards(self, sampled_actions, annots):
        with tf.variable_scope("calculate_rewards"):
            # returns np array shape = (batch, time_steps)
            # sum over action space
            # shapre returned (batch, time_steps)

            def distance_rewards(sampled_actions, annots):
                distances = tf.abs(sampled_actions-annots)
                rews = -tf.reduce_mean(distances, axis=2) - tf.reduce_max(distances, axis=2)
                return rews


            def IOU_rewards(sampled_actions, annots):
                rews = IOU(sampled_actions, annots)
                return rews


            rews = tf.cond(self.epoch < 100, lambda: distance_rewards(sampled_actions, annots), lambda: IOU_rewards(sampled_actions, annots))

            # dr = distance_rewards(sampled_actions, annots)
            # tf.summary.scalar("Dist_rews_mean", tf.reduce_mean(dr))
            #
            # iour = IOU_rewards(sampled_actions, annots)
            # tf.summary.scalar("IOU_mean", tf.reduce_mean(iour))

            tf.summary.scalar("rews_mean", tf.reduce_mean(rews))
            return rews
            # return -tf.reduce_mean(distances, axis=2) - tf.reduce_max(distances, axis=2)

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
            rew = self.calculate_cumulative_rewards(rewards) # batch_size, time_steps
            # rew = self.calculate_rewards_to_go(rewards)

            # one baseline per time step
            # baselines = np.mean(cum_rewards, axis=0)  # time_steps

            advantages = centralize(rew)
            # advantages = centralize(rew)  #-baselines
            # advantages = (batch, time_steps)
            return advantages




if __name__ == "__main__":

    if len(sys.argv) > 1:
        experiment_name = sys.argv[1]
    else:
        experiment_name = "test"

    dataset = Dataset(path="./Datasets/OTB50/")
    time_steps = 10
    max_time_steps = 1500  # longest video in the dataset (girl)
    batch_size = 5
    model = RLT(max_time_steps=max_time_steps, time_steps=time_steps, batch_size=batch_size)
    trainer = Trainer(dataset, model, batch_size=batch_size, time_steps=time_steps, max_time_steps=max_time_steps, experiment_name=experiment_name)

    trainer.train()
