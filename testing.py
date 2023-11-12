import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18
from load_dataset import AVADataset

root_directory = 'C:/Users/adam/Pictures/rjd/rjdcv/train/newTrain/testing'

test_dataset = AVADataset(videos_directory=root_directory, classes_dir = 'classes.csv')
test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = r3d_18(pretrained=True)
num_ftrs = model.fc.in_features

model.fc = torch.nn.Sequential(
        torch.nn.Linear(num_ftrs, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(512, 24)
    )

model.load_state_dict(torch.load('out_models/val_model_3.pth'))

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

        # For each video, the output is the predicted class
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(correct)

# Calculate and print the accuracy of the model on the test set
accuracy = correct / total
print(f'Accuracy of the model on the test videos: {accuracy * 100:.2f}%')