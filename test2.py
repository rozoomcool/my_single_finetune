import torch
from torchvision import transforms as T
from torchvision.transforms import Compose, Resize, CenterCrop
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample,
    ShortSideScale,
)
from pytorchvideo.data import UniformClipSampler, LabeledVideoDataset
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
from torchvision.models.video import r3d_18
import torch.nn as nn

from pytorchvideo.data.labeled_video_paths import LabeledVideoPaths

from load_dataset import AVADataset

# Путь к тестовым данным
test_dataset_path = 'C:/Users/adam/Pictures/rjd/rjdcv/train/train/videos'
test_classes_file = 'C:/Users/adam/Pictures/rjd/rjdcv/train/train/classes.csv'

# Загрузка меток классов
class_names = pd.read_csv(test_classes_file, header=None)
num_classes = len(class_names)

# Создание датасета для тестирования
test_labeled_video_paths = LabeledVideoPaths.from_path(test_dataset_path)
clip_sampler = UniformClipSampler(clip_duration=2.0)

# Преобразования
video_transform = Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_dataset = AVADataset(root_directory='C:/Users/adam/Pictures/rjd/rjdcv/train/train/')

# DataLoader для тестового датасета
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

model = r3d_18(pretrained=False)  # Создаем модель с той же архитектурой
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Изменяем последний слой
model.load_state_dict(torch.load('model_9.pth'))  # Загружаем веса

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

model.eval()  # Переключаем модель в режим оценки


# Включение CUDA ядер


# Тестирование
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for data in test_dataloader:
        inputs, labels = data['video'].to(device), data['label'].to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


print(f'Accuracy of the network on the test videos: {100 * correct / total}%')