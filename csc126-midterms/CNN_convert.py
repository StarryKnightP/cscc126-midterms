import pandas as pd
import math
import os
import numpy as np  
import matplotlib.pyplot as plt
import keras    

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from PIL import Image, ImageOps
 

folder_path = 'input_images'
extensions = ['.jpg', '.jpeg', '.png']

images = []
label = []


if not os.path.exists('CNN_debug_images'):
    print(f"Folder 'CNN_debug_images' does not exist. Creating....")
    os.makedirs('CNN_debug_images')

for file in os.listdir(folder_path):

    if not file.lower().endswith(tuple(extensions)):
        print(f'Skipped Invalid file {file}')
        continue

    try:
        digit = int(file.split('.')[0]).split('_')[0]
    except Exception as e:
        print(f'Error processing file {file}: {e}')
        continue

    label.append(digit)
    img = Image.open(os.path.join(folder_path, file)).convert('L')
    img = img.resize((28, 28))
    img = ImageOps.invert(img)
    arr = np.array(img)
    arr = np.where(arr > 128, 255.0, 0.0)
    
    CNN_debug_path = os.path.join("CNN_debug_images", file)
    img.save(CNN_debug_path)

    arr = arr / 255.0 
    arr = arr.reshape(28, 28, 1)

    images.append(arr)

images_arr = np.array(images, dtype=np.float32)
label_arr = np.array(label, dtype=np.int32)

np.save('CNN_images.npy', images_arr)
np.save('CNN_labels.npy', label_arr)


print("===============================================================================")
print("Convertion Complete. 'CNN_images.npy' & 'CNN_labels.npy' Created")
print(f"Images shape: {images_arr.shape}")
print(f"Labels shape: {label_arr.shape}")
print("===============================================================================")