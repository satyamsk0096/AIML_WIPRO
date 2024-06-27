import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import pandas as pd
from PIL import Image
import os


# Define your dataset class
class ButterflyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        # Create a label to index mapping
        #self.label_to_index = {label: idx for idx, label in enumerate(self.annotations.iloc[:, 1].unique())}
        if 'label' in self.annotations.columns:
            self.label_to_index = {label: idx for idx, label in enumerate(self.annotations['label'].unique())}
        else:
            self.label_to_index = {}  # Handle cases where labels are not provided

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.annotations.iloc[idx, 1]
        label = self.label_to_index[label]  # Convert label to index
        if self.transform:
            image = self.transform(image)
        return image, label



if __name__=='__main__':
    # Define data augmentation and normalization for training
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(size=128, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load your datasets
    # Correct paths to CSV files and rootdirectories
    trainset = ButterflyDataset(csv_file=r'C:\Users\sittu\Desktop\AIML_Wipro\CNN\butterfly_dataset\Training_set.csv',
                                root_dir=r'C:\Users\sittu\Desktop\AIML_Wipro\CNN\butterfly_dataset\train', transform=transform)
    testset = ButterflyDataset(csv_file=r'C:\Users\sittu\Desktop\AIML_Wipro\CNN\butterfly_dataset\Testing_set.csv',
                               root_dir=r'C:\Users\sittu\Desktop\AIML_Wipro\CNN\butterfly_dataset\test', transform=transform)

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)


    # Define the CNN model
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.dropout = nn.Dropout(p=0.5)
            self.fc1 = nn.Linear(64 * 32 * 32, 512)
            self.fc2 = nn.Linear(512, len(trainset.label_to_index))

        def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = x.view(-1, 64 * 32 * 32)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x


    # Function to train the model
    def train_model(model, trainloader, criterion, optimizer, num_epochs=2):
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 100 == 99:
                    print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.3f}')
                    running_loss = 0.0


    # Instantiate the SimpleCNN model
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, trainloader, criterion, optimizer, num_epochs=2)

    # Save the trained model
    PATH = './butterfly_model.pth'
    torch.save(model.state_dict(), PATH)

    # Load the saved model
    loaded_model = SimpleCNN()
    loaded_model.load_state_dict(torch.load(PATH))

    # Print model architecture to verify successful loading
    print(loaded_model)
'''
Epoch 1, Batch 100, Loss: 6.645
Epoch 1, Batch 200, Loss: 4.318
Epoch 2, Batch 100, Loss: 4.317
Epoch 2, Batch 200, Loss: 4.317
SimpleCNN(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (dropout): Dropout(p=0.5, inplace=False)
  (fc1): Linear(in_features=65536, out_features=512, bias=True)
  (fc2): Linear(in_features=512, out_features=75, bias=True)
)
'''