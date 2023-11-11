import torch
import torchvision.models as models
from torch.utils.data import DataLoader

from load_dataset import AVADataset


def execute_finetune():
    root_dir = 'C:/Users/adam/Pictures/ttt'
    # root_dir = 'C:/Users/adam/Pictures/rjd/rjdcv/train/train/'

    num_classes = 24  # Update this to the number of classes in your dataset
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 2

    # Initialize the dataset
    train_dataset = AVADataset(root_directory=root_dir)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    model = models.video.r3d_18()
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)

    # If using a GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("Start on cuda" if torch.cuda.is_available() else " Start on cpu")
    print(device)

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        print(f'START_EXECUTION: Epoch [{epoch + 1}/{num_epochs}]')
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

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.pth')


if __name__ == '__main__':
    execute_finetune()
