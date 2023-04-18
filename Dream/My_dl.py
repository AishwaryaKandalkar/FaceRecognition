import numpy as np
import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import random

IMG_SIZE=64
TRAIN_TEST_SPLIT=70


data = []
for img in os.listdir("data"):
    path = os.path.join("data", img)
    img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img_data = cv2.resize(img_data, (64, 64))
    name = img.split('.')
    if len(name) < 3:
        continue
    name = name[1] 
    data.append([np.array(img_data), np.array(int(name))])
    
random.shuffle(data)
Till = 70/len(data)
Till*=100
train = data[:Till]
test = data[Till:]

X_train = np.array([i[0] for i in train]).reshape(-1, 64, 64, 1)
y_train = np.array([i[1] for i in train])
X_test = np.array([i[0] for i in test]).reshape(-1, 64, 64, 1)
y_test = np.array([i[1] for i in test])

model = keras.Sequential([
    keras.layers.Conv2D(16, kernel_size=(3, 3), activation="relu", input_shape=(64, 64, 1)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation="relu"),
    keras.layers.Dense(4, activation="softmax"),
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

y_train = keras.utils.to_categorical(y_train, num_classes=4)
y_test = keras.utils.to_categorical(y_test, num_classes=4)

new_model = model.fit(X_train, y_train, epochs=6, validation_data=(X_test, y_test))
model.save('model.h5')
# pred_class=[1,2,3,4]

y_pred_batch = model.predict(X_test)
# predicted_classes = np.argmax(y_pred_batch, axis=1)
# print(predicted_classes)

