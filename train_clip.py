import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import clip
import os
import CLIPAutoencoder
import FourierImageDataset


# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load dataset
image_folder = 'path_to_your_image_folder'
dataset = FourierImageDataset(image_folder, preprocess)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Initialize the autoencoder
autoencoder = CLIPAutoencoder(model).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.decoder.parameters(), lr=1e-3)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for images in dataloader:
        images = images.to(device)
        
        # Forward pass
        outputs = autoencoder(images)
        
        # Compute loss
        loss = criterion(outputs, images)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Save the trained model
    torch.save(autoencoder.state_dict(), 'clip_autoencoder.pth')