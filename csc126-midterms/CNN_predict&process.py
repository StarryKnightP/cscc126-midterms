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


try: #test first if file exists or valid to open
    input_data = pd.read_csv('train.csv')
    Y = input_data['label']
    X = input_data.drop('label', axis=1) / 255.0
    Y = to_categorical(Y, num_classes = 10)
    X = X.values.reshape(-1, 28, 28, 1)
except Exception as e:
    print("Error: ", e)
    exit()

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))


model.compile(optimizer='adam',
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
model.summary()
model.fit(X, Y, epochs = 10, batch_size = 128)
model.save('CNN_model.h5')


try:
    CNN_test_data = np.load('CNN_images.npy')
    CNN_labels = np.load('CNN_labels.npy')
except Exception as e:
    print('ERROR!, FAILED TO READ/OPEN FILE: ', e)
    exit()

predictions = model.predict(CNN_test_data)
y_pred = np.argmax(predictions, axis=1)
accuracy = np.mean(y_pred == CNN_labels)

print("CNN labels: ", CNN_labels)
print("Predicted labels: ", y_pred)
print(f"Accuracy: {accuracy}")

plt.figure(figsize=(15, 6))
shown_digits = []
subplot_index = 1

for i in range(len(CNN_labels)):
    if CNN_labels[i] not in shown_digits:
        plt.subplot(2, 5, subplot_index)
        plt.imshow(CNN_test_data[i].reshape(28, 28), cmap='gray')
        plt.title(f"True: {CNN_labels[i]}, Pred: {y_pred[i]}")
        plt.axis('off')
        shown_digits.append(CNN_labels[i])
        subplot_index += 1

    if subplot_index > 10:
        break

plt.tight_layout()
plt.show()