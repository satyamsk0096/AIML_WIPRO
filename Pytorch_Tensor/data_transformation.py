
import torchvision.transforms as transforms
from PIL import Image

# Load image
image = Image.open(r'C:\Users\sittu\OneDrive\Desktop\AIML_Wipro\Pytorch_Tensor\dog.jpg').convert('RGB')           # Ensure image is in RGB format

# Define transformations
crop_transform = transforms.RandomCrop(128)
flip_transform = transforms.RandomHorizontalFlip(p=1)   # Always flip
rotate_transform = transforms.RandomRotation(45)        # Rotate by 45 degrees
resize_transform = transforms.Resize((256, 256))

# Define normalization transformation
normalize_transform = transforms.Normalize(mean=[1.0, 0.5, 0.5],
                                           std=[0.229, 0.224, 0.225])

# Convert PIL image to Tensor before applying normalization
to_tensor_transform = transforms.ToTensor()

# Apply and save crop transformation
cropped_image = crop_transform(image)
cropped_image.save('cropped_image.png')

# Apply and save flip transformation
flipped_image = flip_transform(image)
flipped_image.save('flipped_image.png')

# Apply and save rotate transformation
rotated_image = rotate_transform(image)
rotated_image.save('rotated_image.png')

# Apply and save resize transformation
resized_image = resize_transform(image)
resized_image.save('resized_image.png')

# Convert to tensor and apply normalization
tensor_image = to_tensor_transform(image)             # type: ignore # Convert image to tensor
normalized_image = normalize_transform(tensor_image)  # Apply normalization

# Convert normalized tensor back to PIL image and save
to_pil_image = transforms.ToPILImage()
normalized_pil_image = to_pil_image(normalized_image)
normalized_pil_image.save('normalized_image.png')

print("Images saved successfully.")
