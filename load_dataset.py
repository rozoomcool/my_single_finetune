import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2

from data_augmentation import transform


class AVADataset(Dataset):
    def __init__(self, videos_root, classes_csv, transform=None):
        """
        Инициализация датасета.
        :param videos_root: Корневая директория с папками видео, где каждая папка соответствует классу.
        :param transform: Опциональные трансформации, которые нужно применить к кадрам видео.
        """
        self.videos_root = videos_root
        self.transform = transform
        self.class_to_idx = pd.read_csv(classes_csv).set_index('class_name').to_dict()['class_index']
        self.classes = sorted(os.listdir(videos_root))  # Список классов
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.videos = self._load_videos()

    def _load_videos(self):
        videos = []
        for class_name in self.classes:
            class_dir = os.path.join(self.videos_root, class_name)
            for video_name in os.listdir(class_dir):
                if video_name.endswith('.ava'):  # Фильтруем только .ava файлы
                    videos.append((os.path.join(class_dir, video_name), self.class_to_idx[class_name]))
        return videos

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path, class_idx = self.videos[idx]
        # Загрузите ваше видео и преобразуйте его в последовательность кадров
        # Это может быть ресурсоемкая операция, в зависимости от размера и длительности видео
        frames = self._load_video_frames(video_path)

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        # Переведите список трансформированных кадров в тензор
        frames_tensor = torch.stack(frames)
        return frames_tensor, class_idx

    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        clips = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

            start_time = time.time()
            image = frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = transform(image=frame)['image']
            clips.append(frame)

        cap.release()
        return frames
