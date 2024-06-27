import os
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Define custom dataset
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = []
        self.classes = ['dogs', 'cats']

        for label in self.classes:
            label_dir = os.path.join(img_dir, label)
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                self.img_labels.append((img_path, self.classes.index(label)))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except UnidentifiedImageError:
            print(f"Warning: Cannot identify image file {img_path}. Skipping...")
            return None, None
        if self.transform:
            image = self.transform(image)
        return image, label

if __name__ == '__main__':
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset and dataloader
    image_dataset = CustomImageDataset(img_dir='dataset', transform=transform)
    # Set num_workers to 1 or more
    image_dataloader = DataLoader(image_dataset, batch_size=2, shuffle=True, num_workers=2)

    # Iterate through the dataloader and display information
    for batch_idx, (images, labels) in enumerate(image_dataloader):
        if images is None or labels is None:
            continue  # Skip this batch if None was returned
        print(f"Batch {batch_idx + 1}")
        print(f"Images Shape: {images.shape}")
        print(f"Labels: {labels}")
        break  # Just process the first batch for this example
