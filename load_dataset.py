import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
from torchvision import transforms as T
from PIL import Image


class AVADataset(Dataset):
    def __init__(self, root_directory, frames_len=16):
        self.videos_root = os.path.join(root_directory, 'videos')
        self.classes_root = os.path.join(root_directory, 'classes.csv')
        self.frames_len = frames_len
        # self.classes = os.listdir(self.videos_root)

        classes_cv_tmp = pd.read_csv(self.classes_root, header=None)
        self.classes = []
        for i in range(len(classes_cv_tmp.get(0))):
            if i == 0:
                continue
            self.classes.append(classes_cv_tmp.get(1)[i])

        self.videos = self._load_videos()

        # Define the transformations
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_videos(self):
        videos = []
        for class_idx, class_name in enumerate(self.classes):
            print(f"{class_idx} : {class_name}")
            class_dir = os.path.join(self.videos_root, class_name)
            for index, video_name in enumerate(os.listdir(class_dir)):
                if index % 15 == 0:
                    videos.append((os.path.join(class_dir, video_name), class_idx))
        return videos

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path, class_idx = self.videos[idx]
        frames = self._load_video_frames(video_path)

        if len(frames) < self.frames_len:
            frames += [frames[-1]] * (self.frames_len - len(frames))
        frames = frames[:self.frames_len]

        # Transform each frame
        frames_tensor = torch.stack([self.transform(frame) for frame in frames], dim=0)
        frames_tensor = frames_tensor.permute(1, 0, 2, 3)
        return frames_tensor, class_idx

    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        for _ in range(self.frames_len):
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(frame)
        cap.release()

        # If there were not enough frames, pad with the last frame
        while len(frames) < self.frames_len:
            frames.append(frames[-1])

        return frames
