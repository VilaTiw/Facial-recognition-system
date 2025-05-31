import os
import cv2
import time
import numpy as np
import tensorflow as tf
import face_recognition
from tensorflow.keras.models import load_model
from layers import L1Dist

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import custom_object_scope
class DistanceLayer(Layer):
    """
    This layer is responsible for computing the distance
    between the embeddings
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, compare):
        sum_squared = K.sum(K.square(anchor - compare), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_squared, K.epsilon()))

def face_detection(image):
    facetracker = load_model("facetracker.h5")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120, 120))

    y_hat = facetracker.predict(np.expand_dims(resized / 255, 0))
    sample_coords = y_hat[1][0]
    scaled_coords = np.multiply(sample_coords, [250, 250, 250, 250]).astype(int)
    x1, y1, x2, y2 = scaled_coords

    face_img = image[y1:y2, x1:x2]
    if face_img.size > 0:
        face_img = cv2.resize(face_img, (100, 100))
        cv2.imwrite("Detected_face.jpg", face_img)
        return face_img, scaled_coords, y_hat
    else:
        return 0, 0, 0

def preprocess(file_path):
    try:
        byte_img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(byte_img)
    except Exception as e:
        print(f"Some problem with file: {file_path}: {e}")
        raise
    # img = tf.image.resize(img, (100, 100))
    img = tf.image.resize(img, (64, 64))
    img = img / 255.0
    return img

def verify_(input_path, verification_folder, threshold=0.6):
    input_image = face_recognition.load_image_file(input_path)
    input_encodings = face_recognition.face_encodings(input_image)

    if not input_encodings:
        print("Обличчя не знайдено в вхідному зображенні.")
        return [], False

    input_encoding = input_encodings[0]
    matches = []

    for file in os.listdir(os.path.join("application_data", verification_folder)):
        if file.startswith("."):
            continue
        val_path = os.path.join("application_data", verification_folder, file)
        val_image = face_recognition.load_image_file(val_path)
        val_encodings = face_recognition.face_encodings(val_image)

        if not val_encodings:
            continue

        val_encoding = val_encodings[0]
        distance = np.linalg.norm(input_encoding - val_encoding)
        is_match = distance < threshold
        matches.append(is_match)

    verification = sum(matches) / len(matches) if matches else 0
    verified = verification > 0.2

    print("Dlib verification:", "Verified" if verified else "Unverified")
    print("Match ratio:", verification)

    return matches, verified

def verify(self, *args):
    # Specify thresholds
    detection_threshold = 0.75 #0.5
    verification_threshold = 0.2 #0.8
    verification_folder = "verification_images_gray_face"
    #verification_folder = "valid_images"

    results = []
    for image in os.listdir(os.path.join("application_data", verification_folder)):
        if image.startswith("."):
            continue
        input_img = preprocess(SAVE_PATH)
        validation_img = preprocess(os.path.join("application_data", verification_folder, image))
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)

    filtered_files = [image for image in os.listdir(os.path.join("application_data", verification_folder)) if not image.startswith(".")]
    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(filtered_files)
    verified = verification > verification_threshold

    verification_text = "Verified" if verified == True else "Unverified"

    print(verification_text)
    print(detection)
    print(verification)

    return results, verified

facetracker = load_model("facetracker.h5")
#model = tf.keras.models.load_model("SiameseModel.h5", custom_objects={"L1Dist": L1Dist})
with custom_object_scope({"DistanceLayer": DistanceLayer}):
    # model = tf.keras.models.load_model("../Another Siamese Model/SiameseModel_another_tuned.h5")
    model = tf.keras.models.load_model("../Siamese Model/models/SiameseModel_another_tuned.h5")
    #model = tf.keras.models.load_model("SiameseModel_more_layers.h5")

SAVE_PATH = os.path.join("application_data", "input_images", "input_image.jpg")
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[135:585, 415:865]  # Обрізаємо початковий кадр
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120, 120))

    # Прогноз координат обличчя
    y_hat = facetracker.predict(np.expand_dims(resized / 255, 0))
    sample_coords = y_hat[1][0]
    scaled_coords = np.multiply(sample_coords, [450, 450, 450, 450]).astype(int)

    if y_hat[0] > 0.5:
        x1, y1, x2, y2 = scaled_coords  # Отримуємо координати прямокутника

        # Основний прямокутник
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # Прямокутник для мітки
        cv2.rectangle(frame, (x1, y1 - 30), (x1 + 80, y1), (255, 0, 0), -1)
        # Текстова мітка
        cv2.putText(frame, "Face", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("FaceTracker", frame)
    if cv2.waitKey(1) & 0xFF == ord("v"):
        detected_face = frame[y1+2:y2-1, x1+2:x2-1]
        detected_face = cv2.resize(detected_face, (250, 250))
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
        if detected_face.size > 0:
            cv2.imwrite("Detected_Face.jpg", frame)
            cv2.imwrite(SAVE_PATH, detected_face)
            #results, verified = verify(detected_face)
            results, verified = verify_(SAVE_PATH, "verification_images_gray_face")
            print(results)
            print(max(results))
            print(verified)
            time.sleep(2)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()