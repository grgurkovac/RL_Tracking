from Datasets.vot2017 import VOT2017

if __name__ == "__main__":

    dataset = VOT2017("./Datasets/vot2017/", test_split="train_list.txt", train_split="train_list.txt")
    time_steps = 10
    batch_size = 5
    frames, annots = dataset.get_next_video_frames_and_annots(test=False)
    dataset.animate(frames, annots)

