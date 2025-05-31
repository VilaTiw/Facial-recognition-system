import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)

    img = tf.image.resize(img, (100, 100))
    img = img / 255.0
    return img

def preprocess_twin(input_img, validation_img, label):
    return (preprocess(input_img), preprocess(validation_img), label)


#Siamese L1 Distance class
class L1Dist(Layer):

    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(tf.convert_to_tensor(input_embedding) - tf.convert_to_tensor(validation_embedding))

# Reload model
model = tf.keras.models.load_model('models/SiameseModel_FineTuned.h5', custom_objects={'L1Dist': L1Dist, 'BinaryCrossentropy': tf.losses.BinaryCrossentropy})
print(model.summary())

# Verification function
def verify(model, detection_threshold, verification_threshhold):
    results = []
    for image in os.listdir(os.path.join("data_all/application_data", "verification_images")):
        input_img = preprocess(os.path.join("data_all/application_data", "input_images", "input_image.jpg"))
        validation_img = preprocess(os.path.join("data_all/application_data", "verification_images", image))

        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)

    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(os.listdir(os.path.join("data_all/application_data", "verification_images")))
    verified = verification > verification_threshhold

    return results, verified

# Opencv again
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    frame = frame[100:100 + 250, 515:515 + 250, :]

    cv2.imshow("Verification", frame)

    if cv2.waitKey(10) & 0xFF == ord("v"):
        cv2.imwrite(os.path.join("data_all/application_data", "input_images", "input_image.jpg"), frame)
        results, verified = verify(model, 0.9, 0.7) #0.9, 0.7 (0.5 - 0.5)v
        print(verified)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()