import tensorflow as tf
import numpy as np
import shutil
from functools import lru_cache
import psutil
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


class Trainer():
    def __init__(self, dataset, model, batch_size, time_steps=None, max_time_steps=None, experiment_name="test"):
        self.dataset = dataset
        self.model = model

        self.time_steps = time_steps
        self.max_time_steps = max_time_steps
        self.batch_size = batch_size


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

        def resize_inputs(ims):

            def resize_input(im):
                # h, w, c = self.meta['inp_size']
                imsz = cv2.resize(im[0], (416, 416))
                imsz = np.reshape(imsz, [1, 416, 416, 3])
                return imsz

            resized = np.concatenate([resize_input(im) for im in np.split(ims, indices_or_sections=ims.shape[0], axis=0)], axis=0)
            return resized



        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8

        with tf.Session(config=config).as_default() as sess:
            self.load_or_initialize(sess)

            for j in True:
                print("j:", j)
                print("epoch:", self.dataset.epoch)

                # train_frames, train_annots = self.dataset.get_n_frames_and_annots(self.time_steps, test=False)
                train_frames, train_annots = self.dataset.get_next_video_frames_and_annots(test=False)
                print("memory used:", psutil.virtual_memory().used // 1024 ** 2)
                train_frames = resize_inputs(train_frames)

                # initial annotation to be fed to the network
                train_input_annot = np.zeros_like(train_annots)
                train_input_annot[0][0] = train_annots[0][0]

                train_frames_count = train_frames.shape[0]
                # train_true_video_len = train_frames_count-train_padd_size
                train_true_video_len = train_frames_count
                chunk_loss = 0
                init_states = [np.zeros([2, 1, s]) for s in self.model.lstm_sizes]

                for i in range(0, train_frames_count, self.time_steps):
                    # get video chunk
                    train_frames_chunk = train_frames[i:i+self.time_steps]
                    train_annots_chunk = train_annots[i:i+self.time_steps]
                    train_input_annot_chunk = train_input_annot[i:i+self.time_steps]
                    train_frames_chunk, train_annots_chunk, train_input_annot_chunk, train_padd_size = self.dataset.pad_video(
                        frames=train_frames_chunk, annots=train_annots_chunk, input_annot=train_input_annot_chunk, time_steps=self.time_steps)

                    _, ps_l, sq_l, train_summary, lr, det_actions, global_step_evald, *init_states = sess.run(
                        [
                            self.optimize_step,
                            self.pseudo_loss,
                            self.sq_loss,
                            self.merged,
                            self.learning_rate,
                            self.model.deterministic_actions,
                            self.global_step,
                        ]+list(self.model.last_states),
                        feed_dict={
                            **{
                                self.model.frames_ph: train_frames_chunk,
                                self.model.input_annot_ph: train_input_annot_chunk,
                                self.gt_annots_ph: train_annots_chunk,
                                self.epoch: self.dataset.epoch,
                            },
                            **dict(zip(self.model.init_state_placeholders, init_states))
                        }
                    )

                    chunk_loss += sq_l
                    self.train_writer.add_summary(train_summary, global_step=global_step_evald)



                print()
                loss = chunk_loss/train_true_video_len
                print("lr:", lr, " loss:", loss)
                print()

                # if (loss) < 0.002:
                #     self.dataset.animate(train_frames_chunk, det_actions)
                #     exit()
                # print("Train loss:", loss, " lr:", lr)


                if j % 50 == 0:

                    # checkpoint
                    print("saving")
                    print(self.checkpoint_dir+self.experiment_name)
                    self.saver.save(sess, self.checkpoint_dir+"/"+self.experiment_name, global_step=self.global_step)

                    test_frames, test_annots = self.dataset.get_next_video_frames_and_annots(test=True)
                    test_frames = resize_inputs(test_frames)

                    test_input_annot = np.zeros_like(test_annots)
                    test_input_annot[0][0] = test_annots[0][0]

                    test_frames_count = test_frames.shape[0]
                    test_true_video_len = test_frames_count
                    chunk_loss = 0
                    init_states = [np.zeros([2, 1, s]) for s in self.model.lstm_sizes]

                    for i in range(0, test_frames_count, self.time_steps):
                        print(".", end='')
                        sys.stdout.flush()
                        # get video chunk
                        test_frames_chunk = test_frames[i:i+self.time_steps]
                        test_annots_chunk = test_annots[i:i+self.time_steps]
                        test_input_annot_chunk = test_input_annot[i:i+self.time_steps]

                        test_frames, test_annots, test_input_annot_chunk, test_padd_size = self.dataset.pad_video(
                            frames=test_frames, annots=test_annots, input_annot=test_input_annot_chunk, time_steps=self.time_steps)

                        ps_l, sq_l, test_summary, lr, det_actions, global_step_evald, *init_states = sess.run(
                            [
                                self.pseudo_loss,
                                self.sq_loss,
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
                                    self.gt_annots_ph: test_annots_chunk,
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

            def rews_1(sampled_actions, annots):
                distances = tf.abs(sampled_actions-annots)
                return -tf.reduce_mean(distances, axis=2) - tf.reduce_max(distances, axis=2)

            def rews_2(sampled_actions, annots):
                print("ann:", annots.shape)
                print("sa:", sampled_actions)
                distances = tf.abs(sampled_actions-annots)
                return -tf.reduce_mean(distances, axis=2) - tf.reduce_max(distances, axis=2)

            rews = tf.cond(self.epoch < 300, lambda: rews_1(sampled_actions, annots), lambda: rews_2(sampled_actions, annots))
            print("rw:", rews.shape)
            exit(0)
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
            # rew = self.calculate_cumulative_rewards(rewards) # batch_size, time_steps
            rew = self.calculate_rewards_to_go(rewards)

            # one baseline per time step
            # baselines = np.mean(cum_rewards, axis=0)  # time_steps

            advantages = normalize(rew)
            # advantages = centralize(rew)  #-baselines
            # advantages = (batch, time_steps)
            return advantages




if __name__ == "__main__":

    dataset = VOT2017("./Datasets/vot2017/")
    time_steps = 10
    batch_size = 5
    frames, annots = dataset.get_next_video_frames_and_annots(test=False)
    dataset.animate(frames, annots)

