import os
import cv2
import numpy as np
import tensorflow as tf
import face_recognition
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import custom_object_scope

class L1Dist(Layer):

    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(tf.convert_to_tensor(input_embedding) - tf.convert_to_tensor(validation_embedding))

facetracker = load_model("facetracker.h5") #cnn model
facerecognition = tf.keras.models.load_model('SiameseModel.h5', custom_objects={'L1Dist': L1Dist, 'BinaryCrossentropy': tf.losses.BinaryCrossentropy})


def detect_face(image):
    """
    Детекція обличчя. Повертає вирізане обличчя та координати.
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120, 120))
    y_hat = facetracker.predict(np.expand_dims(resized / 255.0, axis=0))

    if y_hat[0] < 0.5:
        return None, None

    sample_coords = y_hat[1][0]
    scaled_coords = np.multiply(
        sample_coords,
        [image.shape[1], image.shape[0], image.shape[1], image.shape[0]]
    ).astype(int)
    x1, y1, x2, y2 = scaled_coords
    face_img = image[y1:y2, x1:x2]

    if face_img.size == 0:
        return None, None

    return face_img, (x1, y1, x2, y2)


def get_face_embedding(face_img):

    rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

    # Отримуємо ембедінги (як правило, один об'єкт на зображенні)
    embeddings = face_recognition.face_encodings(rgb_face)

    if len(embeddings) == 0:
        return None

    return embeddings[0]