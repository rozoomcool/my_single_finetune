import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18
from load_dataset import AVADataset

root_directory = 'C:/Users/adam/Pictures/rjd/rjdcv/train/newTrain/testing'

classes_cv_tmp = pd.read_csv('classes.csv', header=None)
classes = []
for i in range(len(classes_cv_tmp.get(0))):
    if i == 0:
        continue
    classes.append(classes_cv_tmp.get(1)[i])

test_dataset = AVADataset(videos_directory=root_directory, classes_dir = 'classes.csv', delta=1)
test_loader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = r3d_18(pretrained=True)
num_ftrs = model.fc.in_features

# model.fc = torch.nn.Linear(num_ftrs, 24)
model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 24)
    )

model.load_state_dict(torch.load('out_models/best_model_epoch_3.pth'))

# num_ftrs = r3d_18(pretrained=True)
# model.fc = torch.nn.Linear(num_ftrs, 24)
model.eval()
model = model.to(device)

# Prepare to collect test results
correct = 0
total = 0

# Disabling gradient calculation is important for inference, it reduces memory usage and speeds up computations
with torch.no_grad():
    for data in test_loader:
        videos, labels = data
        videos, labels = videos.to(device), labels.to(device)

        outputs = model(videos)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Выводите классы для каждого предсказанного метки в батче
        for i in range(labels.size(0)):
            print(f'Video {total - labels.size(0) + i}:')
            print(f'Predicted class: {classes[predicted[i]]}, True class: {classes[labels[i]]}')


# Calculate and print the accuracy of the model on the test set
accuracy = correct / total
print(f'Accuracy of the model on the test videos: {accuracy * 100:.2f}%')