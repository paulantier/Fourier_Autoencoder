import torch
import torch.nn as nn

# Define the autoencoder
class CLIPAutoencoder(nn.Module):
    def __init__(self, model):
        super(CLIPAutoencoder, self).__init__()
        self.pre_encoder = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True)
        )
        self.encoder = model.encode_image
        
        self.decoder = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 8192),
            nn.ReLU(True),
            nn.Linear(8192, 512*512),
            nn.Sigmoid()
        )

    def forward(self, real, imag):
        with torch.no_grad():
            real_encoded = self.encoder(self.pre_encoder(real))
            imag_encoded = self.encoder(self.pre_encoder(imag))
        x = torch.cat((real_encoded, imag_encoded), dim=1)
        x = x.to(torch.float32)
        x = self.decoder(x)
        x = x.view(-1, 1, 512, 512)
        return x