import os
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore') 

# Path to dataset
image_folder = r"C:\Users\kumar\Desktop\Hackathon\archive (10)\Images"

# Load a sample image
# Get only image files (skip text files)
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
sample_image = image_files[0]  # First image

img = Image.open(os.path.join(image_folder, sample_image))

# Show image
plt.imshow(img)
plt.axis("off")
plt.show()


import torch
import torchvision.transforms as transforms
import torchvision.models as models

# Load Pretrained ResNet18 Model
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove last layer
model.eval()  # Set to evaluation mode

# Image Preprocessing (Resize, Convert to Tensor, Normalize)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Image and Convert to Feature Vector
def extract_features(image_path):
    img = Image.open(image_path)
    img = transform(img).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():  # No gradient computation
        features = model(img).squeeze().numpy()  # Convert to NumPy array
    return features

# Test on a Sample Image
sample_feature = extract_features(os.path.join(image_folder, sample_image))
print("Feature vector shape:", sample_feature.shape)



import numpy as np

# Store Image Names and Features
image_names = []
feature_vectors = []

# Process All Images
for img_name in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img_name)
    try:
        features = extract_features(img_path)
        image_names.append(img_name)
        feature_vectors.append(features)
    except:
        continue  # Skip errors

# Convert to NumPy Arrays
image_names = np.array(image_names)
feature_vectors = np.array(feature_vectors)

# Save Features
np.save("image_names.npy", image_names)
np.save("feature_vectors.npy", feature_vectors)
