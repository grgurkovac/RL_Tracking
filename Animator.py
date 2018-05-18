from Datasets.dataset import Dataset
import sys
from RLT_model import RLT

import numpy as np
import tensorflow as tf
import os

class Animator():
    def __init__(self, dataset, model, batch_size, time_steps=None, max_time_steps=None, experiment_name="test"):
        self.dataset = dataset
        self.model = model

        self.time_steps = time_steps
        self.max_time_steps = max_time_steps
        self.batch_size = batch_size

        self.epoch = tf.placeholder(tf.float32, name="epoch")
        tf.summary.scalar("epoch", self.epoch)

        self.gt_annots_ph = tf.placeholder(shape=[self.time_steps, self.dataset.action_dim], dtype=tf.float32, name="gt_annots")

        self.experiment_name = experiment_name
        self.experiment_dir = os.path.abspath(os.path.join("./experiments/", experiment_name))
        self.checkpoint_dir = os.path.abspath(os.path.join("./experiments/", experiment_name, "checkpoint/"))

        self.saver = tf.train.Saver()

    def load(self, session):
        # incijalizacija parametara
        print("checkpoint_dir:", self.checkpoint_dir)
        assert os.path.exists(self.checkpoint_dir)

        latest_checkpint_path = tf.train.latest_checkpoint(self.checkpoint_dir)

        assert latest_checkpint_path

        print("Loading")
        self.saver.restore(session, latest_checkpint_path)

        tf.get_default_graph().finalize()


    def get_det_actions(self, frames, annots):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8

        with tf.Session(config=config).as_default() as sess:
            self.load(sess)


            frames = self.model.resize_inputs(frames)

            input_annot = np.zeros_like(annots)
            input_annot[0][0] = annots[0][0]

            frames_count = len(frames)
            init_states = [np.zeros([2, 1, s]) for s in self.model.lstm_sizes]
            det_actions = np.empty(shape=(0, 4))

            for i in range(0, frames_count, self.time_steps):
                # get video chunk
                frames_chunk = np.array(frames[i:i+self.time_steps])
                annots_chunk = np.array(annots[i:i+self.time_steps])
                input_annot_chunk = np.array(input_annot[i:i+self.time_steps])

                frames_chunk, annots_chunk, input_annot_chunk, padd_size = self.dataset.pad_video(
                    frames=frames_chunk, annots=annots_chunk, input_annot=input_annot_chunk, time_steps=self.time_steps)

                det_actions_chunk, *init_states = sess.run(
                    [
                        self.model.deterministic_actions,
                    ]+list(self.model.last_states),
                    feed_dict={
                        **{
                            self.epoch: self.dataset.epoch,
                            self.model.frames_ph: frames_chunk,
                            self.model.input_annot_ph: input_annot_chunk,
                            self.gt_annots_ph: annots_chunk
                        },
                        **dict(zip(self.model.init_state_placeholders, init_states))
                    }
                )
                det_actions = np.vstack((det_actions, det_actions_chunk))
            # det_actions_list = np.split(det_actions, indices_or_sections=len(det_actions), axis=0)
            return det_actions


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
    animator = Animator(dataset, model, batch_size=batch_size, time_steps=time_steps, max_time_steps=max_time_steps, experiment_name=experiment_name)

    animator.get_det_actions()
