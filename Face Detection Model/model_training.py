import os
import cv2
import json
import time
import numpy as np
import tensorflow as tf
import albumentations as alb
import matplotlib.pyplot as plt
from data_processing import load_image, load_labels
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.applications import VGG16

def build_model():
    input_layer = Input(shape=(120, 120, 3))

    vgg = VGG16(include_top=False)(input_layer)

    #Classification branch (class)
    f1 = GlobalMaxPooling2D()(vgg)
    class1 = Dense(2048, activation="relu")(f1)
    class2 = Dense(1, activation="sigmoid")(class1)

    #Regression branch (Bounding box)
    f2 = GlobalMaxPooling2D()(vgg)
    regress1 = Dense(2048, activation="relu")(f2)
    regress2 = Dense(4, activation="sigmoid")(regress1)

    facetracker = Model(inputs=input_layer, outputs=[class2, regress2])

    return facetracker

train_images = tf.data.Dataset.list_files("aug_data/train/images/*.jpg", shuffle=False)
train_images = train_images.map(load_image)
train_images = train_images.map(lambda x: tf.image.resize(x, (120, 120)))
train_images = train_images.map(lambda x: x/255)

test_images = tf.data.Dataset.list_files("aug_data/test/images/*.jpg", shuffle=False)
test_images = test_images.map(load_image)
test_images = test_images.map(lambda x: tf.image.resize(x, (120, 120)))
test_images = test_images.map(lambda x: x/255)

val_images = tf.data.Dataset.list_files("aug_data/val/images/*.jpg", shuffle=False)
val_images = val_images.map(load_image)
val_images = val_images.map(lambda x: tf.image.resize(x, (120, 120)))
val_images = val_images.map(lambda x: x/255)

train_labels = tf.data.Dataset.list_files("aug_data/train/labels/*.json", shuffle=False)
train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

test_labels = tf.data.Dataset.list_files("aug_data/test/labels/*.json", shuffle=False)
test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

val_labels = tf.data.Dataset.list_files("aug_data/val/labels/*.json", shuffle=False)
val_labels = val_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

for dataset in [train_images, test_images, val_images, train_labels, test_labels, val_labels]:
    print(len(dataset))

train = tf.data.Dataset.zip((train_images, train_labels))
train = train.shuffle(5000).batch(8).prefetch(4)

test = tf.data.Dataset.zip((test_images, test_labels))
test = test.shuffle(1300).batch(8).prefetch(4)

val = tf.data.Dataset.zip((val_images, val_labels))
val = val.shuffle(1300).batch(8).prefetch(4)

# data_samples = train.as_numpy_iterator()
# res = data_samples.next()
#
# fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
# for idx in range(4):
#     sample_image = res[0][idx]
#     sample_coords = res[1][1][idx]
#     sample_image = sample_image.copy()
#     cv2.rectangle(sample_image,
#                   tuple(np.multiply(sample_coords[:2], [120, 120]).astype(int)),
#                   tuple(np.multiply(sample_coords[2:], [120, 120]).astype(int)),
#                   (255, 0, 0), 2)
#     ax[idx].imshow(sample_image)
# plt.savefig("some.jpg")
# plt.show()


#Define Losses and Optimizers
batches_per_epoch = len(train)
lr_decay = (1./0.75 -1) / batches_per_epoch
opt = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=lr_decay)

#Localization Loss and Classification Loss
def localization_loss(y_true, y_hat):
    delta_coord = tf.reduce_sum(tf.square(y_true[:, :2] - y_hat[:, :2]))

    h_true = y_true[:, 3] - y_true[:, 1]
    w_true = y_true[:, 2] - y_true[:, 0]

    h_pred = y_hat[:, 3] - y_hat[:, 1]
    w_pred = y_hat[:, 2] - y_hat[:, 0]

    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true - h_pred))

    return delta_coord + delta_size

#Train Neural Network

#Custom Model Class
class FaceTracker(Model):
    def __init__(self, facetracker, **kwargs):
        super().__init__(**kwargs)
        self.model = facetracker

    def compile(self, opt, class_loss, localization_loss, **kwargs):
        super().compile(**kwargs)
        self.c_loss = class_loss
        self.l_loss = localization_loss
        self.opt = opt

    def train_step(self, batch, **kwargs):
        print(f"Type of batch: {type(batch)}")  # має бути tuple
        print(f"Batch length: {len(batch)}")  # має бути 2
        print(f"Type of X: {type(batch[0])}, Shape: {batch[0].shape}, or: {tf.shape(batch[0])}")
        print(f"Type of y: {type(batch[1])}, Shape: {batch[1].shape if isinstance(batch[1], tf.Tensor) else 'Not a tensor'}")
        print(f"Type of y_0: {type(batch[1][0])}, Shape: {batch[1][0].shape}, or: {tf.shape(batch[1][0])}")
        print(f"Type of y_1: {type(batch[1][1])}, Shape: {batch[1][1].shape}, or: {tf.shape(batch[1][1])}")

        X, y = batch
        y_class = b

        with tf.GradientTape() as tape:
            classes, coords = self.model(X, training=True)

            batch_classloss = self.c_loss(y_class, classes)
            batch_regressloss = self.l_loss(tf.cast(y[1], tf.float32), coords)

            total_loss = batch_regressloss + 0.5 * batch_classloss

            grad = tape.gradient(total_loss, self.model.trainable_variables)

        opt.apply_gradients(zip(grad, self.model.trainable_variables))

        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_regressloss}

    def test_step(self, batch, **kwargs):
        X, y = batch
        y_class = tf.reshape(y[0], [-1, 1])

        classes, coords = self.model(X, training=False)

        batch_classloss = self.c_loss(y_class, classes)
        batch_regressloss = self.l_loss(tf.cast(y[1], tf.float32), coords)
        total_loss = batch_regressloss + 0.5 * batch_classloss

        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_regressloss}

    def call(self, X, **kwargs):
        return self.model(X, **kwargs)

class_loss = tf.keras.losses.BinaryCrossentropy()
regress_loss = localization_loss

facetracker = build_model()

X, y = train.as_numpy_iterator().next()
classes, coords = facetracker.predict(X)
print(classes, coords)
print(localization_loss(y[1], coords))
print(class_loss(y[0], classes))

model = FaceTracker(facetracker)
model.compile(opt, class_loss, regress_loss)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
hist = model.fit(train.take(100), epochs=40, validation_data=val, callbacks=[tensorboard_callback])

#Training
logdir = "logs"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
for batch in train.take(1):
    X, y = batch
    print(f"X shape: {X.shape}")
    print(f"Type of y: {type(y)}")
    print(f"Shape of y[0]: {y[0].shape if isinstance(y[0], tf.Tensor) else 'Not a tensor'}")
    print(f"Shape of y[1]: {y[1].shape if isinstance(y[1], tf.Tensor) else 'Not a tensor'}")

hist = model.fit(train.take(100), epochs=40, validation_data=val, callbacks=[tensorboard_callback])

print(hist.history)

#Plot Performance
fig, ax = plt.subplots(ncols=3, figsize=(20, 5))
ax[0].plot(hist.history["total_loss"], color="teal", label="loss")
ax[0].plot(hist.history["val_total_loss"], color="orange", label="val_loss")
ax[0].title.set_text("Loss")
ax[0].legend()

ax[1].plot(hist.history["class_loss"], color="teal", label="class_loss")
ax[1].plot(hist.history["val_class_loss"], color="orange", label="val_class_loss")
ax[1].title.set_text("Classification Loss")
ax[1].legend()

ax[2].plot(hist.history["regress_loss"], color="teal", label="regress_loss")
ax[2].plot(hist.history["val_regress_loss"], color="orange", label="val_regress_loss")
ax[2].title.set_text("Regression Loss")
ax[2].legend()

plt.savefig("Loss_Function.jpg")
plt.show()


test_data = test.as_numpy_iterator()
test_sample = test_data.next()
y_hat = facetracker.predict(test_sample[0])

fix, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx in range(4):
    sample_image = test_sample[0][idx]
    sample_coords = y_hat[1][idx]

    if y_hat[0][idx] > 0.5: #0.9
        cv2.rectangle(sample_image,
                      tuple(np.multiply(sample_coords[:2], [120, 120]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [120, 120]).astype(int)),
                      (255, 0, 0), 2)
    ax[idx].imshow(sample_image)
plt.savefig("some_testing.jpg")
plt.show()

#Saving the model
facetracker.save('facetracker.h5')
