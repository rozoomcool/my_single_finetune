import datetime
import time

import torch
import torchvision.models as models
from torch import nn
from torch.utils.data import DataLoader

from load_dataset import AVADataset


def execute_finetune():
    # root_dir = 'C:/Users/adam/Pictures/ttt'
    root_dir = 'C:/Users/adam/Pictures/rjd/rjdcv/train/train/'
    classes_dir = 'classes.csv'
    train_root_dir = 'C:/Users/adam/Pictures/rjd/rjdcv/train/newTrain/train'
    val_root_dir = 'C:/Users/adam/Pictures/rjd/rjdcv/train/newTrain/wall'

    num_classes = 24
    batch_size = 2
    learning_rate = 0.001
    num_epochs = 100

    # Initialize the dataset
    train_dataset = AVADataset(videos_directory=train_root_dir, classes_dir=classes_dir, delta=2)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = AVADataset(videos_directory=val_root_dir, classes_dir=classes_dir, delta=1)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = models.video.r3d_18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 24)
    )

    # model.fc = torch.nn.Sequential(
    #     torch.nn.Linear(num_ftrs, 512),
    #     torch.nn.ReLU(),
    #     torch.nn.Dropout(0.5),
    #     torch.nn.Linear(512, 24)
    # )

    # If using a GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("Start on cuda" if torch.cuda.is_available() else " Start on cpu")
    print(device)

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    # Training loop
    for epoch in range(num_epochs):
        start_at = time.time()
        print(
            f'START_EXECUTION: Epoch [{epoch + 1}/{num_epochs}], start at: {datetime.datetime.now().time()}, batch size: {batch_size}')
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        torch.save(model.state_dict(), f'out_models/model_{epoch}.pth')
        ends_at = time.time()
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, ends at: {datetime.datetime.now().time()}, passed: {ends_at - start_at}')
        model.eval()  # Переключение модели в режим оценки
        val_running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)

                # Forward pass
                val_outputs = model(val_images)
                val_loss = criterion(val_outputs, val_labels)
                val_running_loss += val_loss.item()

                # Подсчет точности
                _, predicted = torch.max(val_outputs.data, 1)
                total_preds += val_labels.size(0)
                correct_preds += (predicted == val_labels).sum().item()

        # Средняя потеря и точность на валидационном наборе
        avg_val_loss = val_running_loss / len(val_loader)
        val_accuracy = correct_preds / total_preds

        print(f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        if avg_val_loss < best_val_loss:
            print(f'Saving new best model at epoch {epoch + 1} with validation loss: {avg_val_loss:.4f}')
            best_val_loss = avg_val_loss
            best_model_path = f'out_models/best_model_epoch_{epoch}.pth'
            torch.save(model.state_dict(), best_model_path)


if __name__ == '__main__':
    execute_finetune()
