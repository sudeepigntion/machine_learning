import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import warnings
from tensorflow.keras.preprocessing import image

warnings.simplefilter(action='ignore', category=FutureWarning)

train_path = './train'

valid_path = './valid'

test_path = './test'

train_batches = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory(train_path, target_size = (28, 28), classes=['dog', 'cat'], batch_size = 10)

valid_batches = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory(valid_path, target_size = (28, 28), classes=['dog', 'cat'], batch_size = 4)

test_batches = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory(test_path, target_size = (28, 28), classes=['dog', 'cat'], batch_size = 10)

print(train_batches.class_indices)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[28, 28, 3]))
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
model.add(tf.keras.layers.Dense(units=2, activation="softmax"))

# Compiling the model

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model

model.fit_generator(train_batches, steps_per_epoch = 4,
	validation_data = valid_batches, validation_steps = 4, epochs = 5, verbose = 2)

predictions = model.predict_generator(test_batches, steps = 1, verbose = 0)

cm_plot_labels = ["dog", "cat"]

# print(predictions)

test_image = image.load_img("./test/dog/Beautiful-Dogs-Photos-and-Wallpapers-Free-Download.jpg", target_size=(28, 28))

test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

print(test_image.shape)
print(test_image)

result = model.predict(test_image)

print("Dog" if result[0][-1] == 1.0 else "Cat")


model.save('image_classifier.h5')