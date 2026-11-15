import os
from PIL import Image
import torch
from torchvision import transforms

# Path to your downloaded image folder
folder_path = r"D:\anu_research\Pachygyria with Cerebellar Hypoplasia"

# Basic preprocessing transform (resizes images to standard 224x224 for PyTorch)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Find all JPG files in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
print(f"Total images found: {len(image_files)}")

# Test loading the first image
if image_files:
    first_image_path = os.path.join(folder_path, image_files[0])
    img = Image.open(first_image_path).convert('RGB')
    img_tensor = transform(img)
    
    print(f"Successfully loaded: {image_files[0]}")
    print(f"Tensor Shape: {img_tensor.shape}")  # Should print torch.Size([3, 224, 224])
    print("Everything works! Time for sleep 😴")