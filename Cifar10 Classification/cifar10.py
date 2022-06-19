# Importing the dependencies

import os
import json
import random
import requests
import subprocess
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

# Dataset preprocessing

# Loading the dataset

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

class_names = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

# Image Normalization

X_train = X_train / 255.0

X_test = X_test / 255.0

print(X_train.shape)

# Defining the model

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[32, 32, 3]))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=256, activation="relu"))
model.add(tf.keras.layers.Dense(units=10, activation="softmax"))

# Compiling the model

model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training the model

model.fit(X_train, y_train, batch_size=100, epochs=100, verbose=1)

# Model evalutaion

test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Test accuracy is {}".format(test_accuracy))

# Creating the directory for the model

MODEL_DIR = "model/"
version = 1

export_path = os.path.join(MODEL_DIR, str(version))

# Saving the model for Tensorflow Serving

model.save('cifar10_model.h5')