import tensorflow as tf
import numpy as np
import cv2
import os
from scipy import ndimage
from functools import lru_cache

from Datasets.Dataset import *


class VOT2017(Dataset):

    def __init__(self, *args, test_split="test_list2.txt", train_split="train_list2.txt", **kwargs):
        super().__init__(*args, **kwargs)

        self.action_dim = 8
        self.max_time_steps = 1500  # longest video

        self.train_dict = dict()
        self.train_dict["next_video_index"] = 0
        self.train_dict["split"] = train_split

        self.test_dict = dict()
        self.test_dict["next_video_index"] = 0
        self.test_dict["split"] = test_split

        self.load_videos_list()


        self.load_next_video(test=False)
        self.load_next_video(test=True)

    def load_videos_list(self):
        with open(self.path+self.train_dict["split"]) as f:
            train_video_names = f.readlines()

        self.train_dict["videos_list"] = [v.strip() for v in train_video_names]
        self.train_dict["videos_count"] = len(self.train_dict["videos_list"])


        with open(self.path+self.test_dict["split"]) as f:
            test_video_names = f.readlines()

        self.test_dict["videos_list"] = [v.strip() for v in test_video_names]
        self.test_dict["videos_count"] = len(self.test_dict["videos_list"])

    # @lru_cache(maxsize=1)
    def load_frames_and_annots_for_video(self, test):

        D = self.test_dict if test else self.train_dict

        train_video_frames_list = []
        directory = os.path.join(self.path, D["video"])

        for filename in sorted(os.listdir(directory)):
            if filename.endswith(".jpg"):
                img_path = os.path.join(directory, filename)
                train_video_frames_list.append([ndimage.imread(img_path)])

        train_video_annots_list = []
        gt_path = os.path.join(self.path, D["video"], "groundtruth.txt")

        for line in open(gt_path):
            train_video_annots_list.append(np.array([line.rstrip().split(',')], dtype=np.float32))

        D["video_frames"] = np.concatenate(train_video_frames_list)
        D["video_annots"] = np.concatenate(train_video_annots_list)
        D["next_frame_index"] = 0

        D["frame_height"] = D["video_frames"].shape[1]
        D["frame_width"] = D["video_frames"].shape[2]


        dots_count = D["video_annots"].shape[-1]//2

        D["frame_dimensions"] = np.tile(np.array([D["frame_width"], D["frame_height"]]), [dots_count])
        D["video_annots"] /= D["frame_dimensions"]


    def get_train_set(self):
        pass

    def get_test_set(self):
        pass

    def get_next_video_frames_and_annots(self, test):
        D = self.test_dict if test else self.train_dict

        # todo: use np.save/np.load

        # load frames
        frames = D["video_frames"]

        # load annots
        annots = D["video_annots"]

        self.load_next_video(test=test)

        return (frames, annots)

    def pad_video(self, frames, annots, time_steps):
        n = list(frames.shape)[0]
        roof_n = int(np.ceil(n / time_steps) * time_steps) # first bigger time_steps

        pad_size = roof_n-n
        padded_frames = np.pad(frames, [(0,pad_size), (0,0), (0,0), (0,0)], 'constant', constant_values=(0,))
        padded_annots = np.pad(annots, [(0,pad_size), (0,0)], 'constant', constant_values=(0,))

        return padded_frames, padded_annots, pad_size



    def get_n_frames_and_annots(self, n, test):

        D = self.test_dict if test else self.train_dict

        if D["next_frame_index"]+n > D["video_frames_count"]:
            n = D["video_frames_count"] - D["next_frame_index"]
            load_next_vid=True

        else:
            load_next_vid=False

        frames = D["video_frames"][D["next_frame_index"]:D["next_frame_index"]+n]
        annots = D["video_annots"][D["next_frame_index"]:D["next_frame_index"]+n]
        D["next_frame_index"] = (D["next_frame_index"] + n) % D["video_frames_count"]


        if load_next_vid:
            self.load_next_video(test=test)

        return (frames, annots)



    def load_next_video(self, test):

        D = self.test_dict if test else self.train_dict

        print("Loading next video")
        D["video"] = D["videos_list"][D["next_video_index"]]


        self.load_frames_and_annots_for_video(test=test)
        D["video_frames_count"] = len(D["video_annots"])

        D["next_video_index"] = (D["next_video_index"] + 1) % D["videos_count"]

        print("Loaded next test="+str(test) +" video: ", D["video"])


    def animate(self, frames, coordinates, ground_truth=False):
        cap = cv2.VideoCapture(0)
        # frames = list(map(lambda fr: np.squeeze(fr), np.split(frames, axis=0, indices_or_sections=frames.shape[0])))
        # coordinates = list(map(lambda an: np.squeeze(an), np.split(coordinates, axis=0, indices_or_sections=coordinates.shape[0])))

        winname = "test"
        ndots = coordinates.shape[-1]//2
        scaler = np.tile(np.array([frames.shape[2], frames.shape[1]]), [ndots])
        for i in range(len(frames)):
            frame = frames[i]
            anot = coordinates[i]*scaler

            dots = np.array(
                list(zip(
                    [anot[i] for i in (0, 2, 4, 6)],
                    [anot[i] for i in (1, 3, 5, 7)]
                )
            ))

            cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
            cv2.moveWindow(winname, 5, 5)

            if ground_truth:
                color=(0, 0, 255)
            else:
                color=(255, 0, 0)

            cv2.polylines(frame, np.int32([dots]), 1, color=color, thickness=2)

            cv2.imshow(winname, frame)
            cv2.waitKey(0)

        cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":

    dataset = VOT2017("./vot2017/")
    frames, anots = dataset.get_n_frames_and_annots(10, test=False)

    dataset.animate(frames, anots)
