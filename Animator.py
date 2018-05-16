from Datasets.vot2017 import VOT2017
from RLT_model import RLT

if __name__ == "__main__":

    time_steps = 10
    max_time_steps = 1500 # longest video in the dataset (girl)
    batch_size = 5
    dataset = VOT2017("./Datasets/OTB50/", test_split="Dudek.txt", train_split="Dudek.txt")
    model = RLT(max_time_steps=max_time_steps, time_steps=time_steps, batch_size=batch_size)
    frames, annots = dataset.get_next_video_frames_and_annots(test=False)
    dataset.animate(frames, annots)

