import os
from PIL import Image

def convert_images_to_grayscale(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check if the file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                # Open the image
                img = Image.open(file_path)
                
                # Convert the image to grayscale
                gray_img = img.convert('L')
                
                # Save the grayscale image
                gray_img.save(file_path)
                #print(f"Converted {filename} to grayscale.")
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")

# Specify the folder path
folder_path = 'train'

# Convert all images in the folder to grayscale
convert_images_to_grayscale(folder_path)