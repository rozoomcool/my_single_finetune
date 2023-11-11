import torch
import torchvision.models as models
from torch.utils.data import DataLoader

from data_augmentation import transform
from load_dataset import AVADataset

def execute_finetune():
    num_classes = 5  # Количество ваших кастомных классов
    batch_size = 16  # Размер батча
    learning_rate = 0.001  # Скорость обучения
    num_epochs = 10

    # # Заморозка всех слоёв, кроме последнего
    # for param in model.parameters():
    #     param.requires_grad = False
    # model.fc.requires_grad = True

    train_dataset = AVADataset(videos_root='C:\\Users\\adam\\Pictures\\РЖД Видео\\РЖД CV\\train\\train\\videos',
                               classes_csv='C:\\Users\\adam\\Pictures\\РЖД Видео\\РЖД CV\\train', transform=transform)
    val_dataset = AVADataset(videos_root='path_to_val_data',
                             classes_csv='C:\\Users\\adam\\Pictures\\РЖД Видео\\РЖД CV\\train', transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    model = models.video.r3d_18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()  # Устанавливаем модель в режим обучения
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Прямой проход
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Обратный проход и оптимизация
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Выводим статистику после каждой эпохи
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

        # Валидация модели
        model.eval()  # Устанавливаем модель в режим валидации
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:
                images = images.to(device)

    # if len(clips) == self.frames_len:
    #     with torch.no_grad():
    #         input_frames = np.array(frames)
    #         input_frames = np.expand_dims(input_frames, axis=0)
    #         input_frames = np.transpose(input_frames, (0, 4, 1, 2, 3))
    #         input_frames = torch.tensor(input_frames, dtype=torch.float32)
    #         input_frames = input_frames.to(self.device)
    #         outputs = model(input_frames)
    #         _, preds = torch.max(outputs.data, 1)
    #
    #         label = class_names[preds].strip()

