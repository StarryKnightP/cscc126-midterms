import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import keras    

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.utils import to_categorical


try: #test first if file exists or valid to open
    input_data = pd.read_csv('train.csv')
    Y = input_data['label']
    X = input_data.drop('label', axis=1) / 255.0
    Y = to_categorical(Y, num_classes = 10)
    X = X.values.reshape(-1, 784)
except Exception as e:
    print("Error: ", e)
    exit()

model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax'),
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
model.summary()
model.fit(X, Y, epochs = 30, batch_size = 128, validation_split = 0.1)
model.save('ANN_model.h5')

try:
    ANN_test_data = pd.read_csv('ANN.csv')
    ANN_labels = ANN_test_data['label']
    X_test = ANN_test_data.drop('label', axis=1) / 255.0
    X_test = X_test.values.reshape(-1, 784)
except Exception as e:
    print('ERROR!, FAILED TO READ/OPEN FILE: ', e)
    exit()


predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)
accuracy = np.mean(y_pred == ANN_labels)


print("ANN labels: ", ANN_labels)
print("Predicted labels: ", y_pred)
print(f"Accuracy: {accuracy}")


plt.figure(figsize=(15, 6))
shown_digits  = []
subplot_index = 1
 
for i in range(len(ANN_labels)):
    if ANN_labels.iloc[i] not in shown_digits:
        text_color = "green" if y_pred[i] == ANN_labels.iloc[i] else "red"
 
        plt.subplot(2, 5, subplot_index)
        plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
        plt.title(f"True: {ANN_labels.iloc[i]}, Pred: {y_pred[i]}",
                  color=text_color)
        plt.axis('off')
 
        shown_digits.append(ANN_labels.iloc[i])
        subplot_index += 1
 
    if subplot_index > 10:
        break
 
plt.tight_layout()
plt.show()