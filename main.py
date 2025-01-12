import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import clip
from CLIPAutoencoder import CLIPAutoencoder
from FourierImageDataset import FourierImageDataset
from tqdm import tqdm

print("Initializing autoencoder...")

print("Starting training loop...")

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading CLIP model... using device:", device)

model, preprocess = clip.load("ViT-B/32", device=device)

image_folder = 'train_images'

print("Loading dataset from:", image_folder)

# Load dataset

# Update dataset and dataloader
dataset = FourierImageDataset(image_folder, preprocess)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

print('Initializing autoencoder...')
# Initialize the autoencoder
autoencoder = CLIPAutoencoder(model).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.decoder.parameters(), lr=1e-3)

print('Starting training loop...')

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for images, real_images, imaginary_images in tqdm(dataloader):
        real_images = real_images.to(device)
        imaginary_images = imaginary_images.to(device)
        images = images.to(device)

        # Forward pass
        outputs = autoencoder(real_images, imaginary_images)
        
        # Compute loss
        loss = criterion(outputs, images)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(autoencoder.state_dict(), 'clip_autoencoder.pth')