import os
import cv2
import numpy as np
import torch



def fft2d(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    real_part = np.real(fshift)
    imag_part = np.imag(fshift)
    return real_part, imag_part

def ifft2d(real_part, imag_part):
    fshift = real_part + 1j * imag_part
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back

class FourierImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, preprocess):
        self.image_folder = image_folder
        self.preprocess = preprocess
        self.image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image_reference = image.copy()
        image = cv2.resize(image, (224, 224))
        image_reference = cv2.resize(image_reference, (512, 512))
        real_part, imag_part = fft2d(image)
        real_part = torch.tensor(real_part, dtype=torch.float32).unsqueeze(0)
        imag_part = torch.tensor(imag_part, dtype=torch.float32).unsqueeze(0)
        return image_reference, real_part, imag_part