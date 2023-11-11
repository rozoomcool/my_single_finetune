import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import cv2
from torchvision import transforms as T
import torchvision.transforms as transforms


class AVADataset(Dataset):
    def __init__(self, root_directory, transform=None):
        """
        Инициализация датасета.
        :param videos_root: Корневая директория с папками видео, где каждая папка соответствует классу.
        :param transform: Опциональные трансформации, которые нужно применить к кадрам видео.
        """
        self.root_directory = root_directory
        self.videos_root = f'{root_directory}videos'
        self.transform = transform
        # self.class_to_idx = pd.read_csv(f'{root_directory}classes.csv')
        # print(self.class_to_idx)
        self.classes = os.listdir(self.videos_root)
        # self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.videos = self._load_videos()

        self.transform = T.Compose([
            # Ваши трансформации здесь, например:
            T.Resize((128, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_videos(self):
        videos = []
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.videos_root, class_name)
            for video_name in os.listdir(class_dir):
                print(video_name)
                videos.append((os.path.join(class_dir, video_name), class_idx))
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
        clips = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Преобразуем кадр из BGR в RGB и создаем PIL Image
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)

            # Применяем трансформации, ожидающие PIL Images или ndarrays
            if self.transform:
                frame = self.transform(frame)

            # Добавляем преобразованный кадр в список
            clips.append(frame)

        cap.release()

        # Преобразуем список PIL Images или ndarrays в тензоры
        clips_tensor = torch.stack([transforms.ToTensor()(frame) for frame in clips])

        return clips_tensor
