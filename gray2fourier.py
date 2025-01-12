import os
import numpy as np
from PIL import Image
from numpy.fft import fft2, fftshift

def gray2fourier(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path).convert('L')
            img_array = np.array(img)

            # Perform 2D FFT
            fft_result = fftshift(fft2(img_array))

            # Save the result
            output_filename = os.path.splitext(filename)[0] + '.npy'
            output_path = os.path.join(output_folder, output_filename)
            np.save(output_path, fft_result)

input_folder = 'train'
output_folder = 'train_fourier'
gray2fourier(input_folder, output_folder)