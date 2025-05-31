# Import Standard Dependencies
import cv2
import os
import uuid
import random
import numpy as np
import matplotlib.pyplot as plt

# Import tensorflow dependencies - Functional API
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten

# Setup paths
POS_PATH = os.path.join('data_all/data', 'positive')
NEG_PATH = os.path.join('data_all/data', 'negative')
ANC_PATH = os.path.join('data_all/data', 'anchor')

# Get image directories
anchor = tf.data.Dataset.list_files(ANC_PATH+'/*.jpg').take(3000)
positive = tf.data.Dataset.list_files(POS_PATH+'/*.jpg').take(3000)
negative = tf.data.Dataset.list_files(NEG_PATH+'/*.jpg').take(3000)

def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)

    img = tf.image.resize(img, (100, 100))
    img = img / 255.0
    return img

def preprocess_twin(input_img, validation_img, label):
    return (preprocess(input_img), preprocess(validation_img), label)


positive = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negative = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positive.concatenate(negative)

# Build dataLoader pipeline
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=10000)

# Training partition
train_data = data.take(round(len(data)*.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

# Testing partition
test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)

# Building a siamese neural network
# def make_embedding():
#     inp = Input(shape=(100, 100, 3), name="input_image")
#
#     c1 = Conv2D(64, (10, 10), activation="relu")(inp)
#     m1 = MaxPooling2D(64, (2, 2), padding="same")(c1)
#
#     c2 = Conv2D(128, (7, 7), activation="relu")(m1)
#     m2 = MaxPooling2D(64, (2, 2), padding="same")(c2)
#
#     c3 = Conv2D(128, (4, 4), activation="relu")(m2)
#     m3 = MaxPooling2D(64, (2, 2), padding="same")(c3)
#
#     c4 = Conv2D(256, (4, 4), activation="relu")(m3)
#     f1 = Flatten()(c4)
#     d1 = Dense(4096, activation="sigmoid")(f1)
#
#     return Model(inputs=[inp], outputs=[d1], name="embedding")

def make_embedding():
    inp = Input(shape=(100, 100, 3), name="input_image")

    c1 = Conv2D(64, (10, 10), activation="relu")(inp)
    m1 = MaxPooling2D(64, (2, 2), padding="same")(c1)

    c2 = Conv2D(128, (7, 7), activation="relu")(m1)
    m2 = MaxPooling2D(64, (2, 2), padding="same")(c2)

    c3 = Conv2D(128, (4, 4), activation="relu")(m2)
    m3 = MaxPooling2D(64, (2, 2), padding="same")(c3)

    c4 = Conv2D(256, (4, 4), activation="relu")(m3)
    m4 = MaxPooling2D(64, (2, 2), padding="same")(c4)

    c5 = Conv2D(512, (2, 2), activation="relu")(m4)
    f1 = Flatten()(c5)
    d1 = Dense(4096, activation="sigmoid")(f1)

    return Model(inputs=[inp], outputs=[d1], name="embedding")

embedding = make_embedding()
embedding.summary()

# Siamese L1 Distance class
class L1Dist(Layer):

    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(tf.convert_to_tensor(input_embedding) - tf.convert_to_tensor(validation_embedding))

def make_siamese_model():
    # Anchor image input in the network
    input_image = Input(name="input_img", shape=(100, 100, 3))

    # Validation image input in the network
    validation_image = Input(name="validation_img", shape=(100, 100, 3))

    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = "distance"
    distances = siamese_layer(embedding(input_image), embedding(validation_image))

    # Classification layer
    classifier = Dense(1, activation="sigmoid")(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name="SiameseNeuralNetwork")

siamese_model = make_siamese_model()
print(siamese_model.summary())

# Setup Loss and Optimizer
binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4) # 0.0001

# Establish checkpoints
checkpoint_dir = 'training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

@tf.function
def train_step(batch):
    # Record all of our operations
    with tf.GradientTape() as tape:
        # Get anchor and pos/neg image
        X = batch[:2]
        # Get label
        y = batch[2]

        # Forward pass
        yhat = siamese_model(X, training=True)
        y = tf.reshape(y, tf.shape(yhat))
        # Calculate loss
        loss = binary_cross_loss(y, yhat)

    print(loss)

    # Calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)

    # Calculate updated weight and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))

    return loss

def train(data, EPOCHS):
    # Loop through epochs
    for epoch in range(1, EPOCHS+1):
        print("\n Epoch {}/{}".format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))

        r = Recall()
        p = Precision()

        # Loop through each batch
        for idx, batch in enumerate(data):
            loss = train_step(batch)
            yhat = siamese_model.predict(batch[:2])
            r.update_state(batch[2], yhat)
            p.update_state(batch[2], yhat)
            progbar.update(idx+1)
        print(loss.numpy(), r.result().numpy(), p.result().numpy())

        # Save checkpoints
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

# Train the model
EPOCHS = 70
train(train_data, EPOCHS)

recall_metric = Recall()
precision_metric = Precision()
count = 0

for test_input, test_validation, y_true in test_data.as_numpy_iterator():
    y_hat = siamese_model.predict([test_input, test_validation])

    recall_metric.update_state(y_true, y_hat)
    precision_metric.update_state(y_true, y_hat)

    for i in range(len(test_input)):
        plt.figure(figsize=(10, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(test_input[i])
        plt.title("Input Image")

        plt.subplot(1, 2, 2)
        plt.imshow(test_validation[i])
        plt.title("Validation Image")

        plt.savefig(f"visualization/visualization_{count}.png")
        plt.show()

        count += 1

        if count >= 10:
            break
    if count >= 10:
        break

recall = recall_metric.result().numpy()
precision = precision_metric.result().numpy()

print(f"Recall: {recall}")
print(f"Precision: {precision}")

# Save model
siamese_model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
siamese_model.save('SiameseModel_more_layers.h5')
