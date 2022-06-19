# Importing Dependencies

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
import numpy as np

model = VGG16()

img_path = './uploads/elephant.jpg'
x = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(x)
x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
# x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

prediction = model.predict(x)

label = decode_predictions(prediction)

label = label[0][0]
# print the classification
print('%s (%.2f%%)' % (label[1], label[2]*100))
