# Stage 1 Import project dependencies

import os
import requests
import numpy as np
import tensorflow as tf
from scipy.misc import imsave, imread
from flask import Flask, request, jsonify

# Stage 2 Load the pretrained model

# Loading model structure

model = tf.keras.models.load_model('fashion_model.h5')

# Loading model weights

model.load_weights('fashion_model.h5')

# Stage 3 Create the Flask Api

# Defining flask application

app = Flask(__name__)

# Defining the classify image function

@app.route("/api/v1/<string:img_name>", methods=["POST"])
def classify_image(img_name):

    print(img_name)

    upload_dir = "uploads/"

    image = imread(upload_dir + img_name)

    print(image.shape)

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle Boot"
    ]

    prediction = model.predict([image.reshape(1, 28*28)])

    return jsonify({"object_detected": classes[np.argmax(prediction[0])]})

# Stage 4 Start the Flask Api and make predictions

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=3000, debug=False)
