import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import decode_predictions
import numpy as np

base_model = VGG19()

model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

img_path = './models/elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
x = preprocess_input(x)

prediction = model.predict(x)

label = decode_predictions(prediction)

label = label[0][0]
# print the classification
print('%s (%.2f%%)' % (label[1], label[2]*100))

