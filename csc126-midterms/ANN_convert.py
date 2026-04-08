import pandas as pd
import os
import numpy as np    

from PIL import Image, ImageOps


folder_path = "high_accu_dataset" #path for raw images such as written digits
extensions = ['.jpg', '.jpeg', '.png'] #list of valid image file extensions

if not os.path.exists("debug\ANN_debug_images"): ##creates file for a debugger
    os.makedirs("debug\ANN_debug_images")


images = []
label = []

for file in os.listdir(folder_path):

    if not file.lower().endswith(tuple(extensions)):
        print(f"Skipped Invalid File: {file}")
        continue #skips files that do not have a valid image extension

    ## Error hanlder, checks if file is named correctly, else error
    try:
        digit = int(file.split('.')[0].split('_')[0]) #extracts the digit label from the filename, assuming the format is "digit_something.ext"
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        continue

    label.append(digit)
    img = Image.open(os.path.join(folder_path, file)).convert('L') #opens the image and converts it to grayscale
    img = img.resize((28, 28)) 
    img = ImageOps.invert(img) 
    arr = np.array(img, dtype=np.float32)
    arr = np.where(arr > 128, 255.0, 0.0) #binarizes the image by setting pixels above a certain threshold to white and those below to black

    ## Debugger, meant to check if the images are processed as intended
    ANN_image_path = os.path.join("debug\ANN_debug_images", file) 
    img.save(ANN_image_path) 

    #arr = arr / 255.0 #delete this para ma increase ang accuracy, kay na normalize na ni siya pag predict og process
    arr = arr.flatten()
    images.append(arr)


df = pd.DataFrame(images)
df.insert(0, 'label', label) 
df.to_csv("ANN.csv", index=False)#saves data into a CSV file named "ANN.csv" for training

print("===============================================================================")
print("Process complete. 'ANN.csv' created.")
print(df.head())
print("===============================================================================")