import os
import cv2
import uuid
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

facetracker = load_model("facetracker.h5")

folder = os.path.join("../Siamese Model/data_all/data/testing")
for image in os.listdir(folder):
    if image.startswith("."):
        print("skipped hidden file: ", image)
        continue
    # img = cv2.imread(os.path.join(folder, image))
    # img = img[100:100+250, 515:515+250, :]
    # print(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    byte_img = tf.io.read_file(os.path.join(folder, image))
    img = tf.io.decode_jpeg(byte_img)
    img = img.numpy()
    #img = cv2.cvtColor(img.numpy(), cv2.COLOR_GRAY2RGB)
    resized = tf.image.resize(tf.convert_to_tensor(img), (120, 120))

    y_hat = facetracker.predict(np.expand_dims(resized / 255, 0))
    sample_coords = y_hat[1][0]
    scaled_coords = np.multiply(sample_coords, [250, 250, 250, 250]).astype(int)

    if y_hat[0] > 0.5:
        x1, y1, x2, y2 = scaled_coords
        detected_face = img[y1 + 2:y2 - 1, x1 + 2:x2 - 1]
        detected_face = cv2.resize(detected_face, (250, 250))
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)

        if detected_face.size > 0:
            cv2.imwrite(f"../Siamese Model/data_all/data/testing_face/{uuid.uuid1()}.jpg", detected_face)