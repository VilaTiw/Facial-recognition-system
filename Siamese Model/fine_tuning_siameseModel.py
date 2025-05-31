import cv2
import os
import uuid
import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.metrics import Precision, Recall

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


def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)

    # img = tf.image.resize(img, (100, 100))
    img = tf.image.resize(img, (64, 64))
    img = img / 255.0
    return img

def preprocess_twin(input_img, validation_img, label):
    return (preprocess(input_img), preprocess(validation_img), label)

# Custom L1 Distance layer module (its needed to load the custom model
class L1Dist(Layer):

    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(tf.convert_to_tensor(input_embedding) - tf.convert_to_tensor(validation_embedding))


# POS_PATH = os.path.join('data_all/data', 'new_positive')
# NEG_PATH = os.path.join('data_all/data', 'new_negative')
# ANC_PATH = os.path.join('data_all/data', 'anchor')

POS_PATH = os.path.join('data_all/data', 'new_positive_gray')
NEG_PATH = os.path.join('data_all/data', 'new_negative_gray')
ANC_PATH = os.path.join('data_all/data', 'anchor_gray')

# Завантажуємо попередньо навчену модель
# siamese_model = tf.keras.models.load_model("App_kivy/SiameseModel_LargerDataSet.h5", custom_objects={"L1Dist": L1Dist})
with custom_object_scope({"DistanceLayer": DistanceLayer}):
    siamese_model = tf.keras.models.load_model("../Another Siamese Model/SiameseModel_another.h5")


# Отримуємо списки файлів для навчання
anchor = tf.data.Dataset.list_files(ANC_PATH + '/*.jpg').take(500)
positive = tf.data.Dataset.list_files(POS_PATH + '/*.jpg').take(500)
negative = tf.data.Dataset.list_files(NEG_PATH + '/*.jpg').take(500)

# Оновлюємо датасет
positive_pairs = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negative_pairs = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))

# Об'єднуємо всі пари
data = positive_pairs.concatenate(negative_pairs)

# Обробляємо датасет
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=10000)

# Розділяємо на навчальний і тестовий набори
train_data = data.take(round(len(data)*.7)).batch(16).prefetch(8)
#train_data = train_data.repeat(15)
test_data = data.skip(round(len(data)*.7)).take(round(len(data)*.3)).batch(16).prefetch(8)

# Використовуємо менший learning rate для fine-tuning
opt = tf.keras.optimizers.Adam(1e-6)  # Було 1e-4, тепер 1e-5

# Визначаємо функцію втрат
binary_cross_loss = tf.losses.BinaryCrossentropy()

# Функція навчання
def train_step(batch):
    with tf.GradientTape() as tape:
        X = batch[:2]
        y = batch[2]

        # Передбачення
        yhat = siamese_model(X, training=True)
        y = tf.reshape(y, tf.shape(yhat))

        # Обчислення втрат
        loss = binary_cross_loss(y, yhat)

    # Обчислення градієнтів
    grad = tape.gradient(loss, siamese_model.trainable_variables)

    # Оновлення ваг моделі
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))

    return loss

def fine_tune(data, EPOCHS):
    for epoch in range(1, EPOCHS+1):
        print("\n Epoch {}/{}".format(epoch, EPOCHS))
        steps_per_epoch = len(train_data) // 16
        progbar = tf.keras.utils.Progbar(steps_per_epoch)
        # progbar = tf.keras.utils.Progbar(len(data))

        for idx, batch in enumerate(data):
            loss = train_step(batch)
            progbar.update(idx+1)

        # Зберігаємо модель після кожних 5 епох
        if epoch % 5 == 0:
            siamese_model.save("SiameseModel_another_tuned.h5")

# Запускаємо fine-tuning
fine_tune(train_data, EPOCHS=25)

recall_metric = Recall()
precision_metric = Precision()

for test_input, test_validation, y_true in test_data.as_numpy_iterator():
    y_hat = siamese_model.predict([test_input, test_validation])
    recall_metric.update_state(y_true, y_hat)
    precision_metric.update_state(y_true, y_hat)

recall = recall_metric.result().numpy()
precision = precision_metric.result().numpy()

print(f"📊 Recall: {recall}, Precision: {precision}")