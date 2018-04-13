import tensorflow as tf
import numpy as np

class Dataset:

    def __init__(self, path):
        self.path = path
        self.train_video_index = -1
        self.test_video_index = -1

        self.train_next_frame_index = 0
        self.test_next_frame_index = 0


    def get_train_set(self):
        pass

    def get_test_set(self):
        pass

    def get_next_n_frames_and_annots(self, n):
        pass

    def get_n_frames(self, video_index, frame_index, n, test):
        pass

    def get_n_annots(self, video_index, frame_index, n, test):
        pass
