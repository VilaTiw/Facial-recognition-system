import os
import cv2
import uuid
import numpy as np
import tensorflow as tf

def data_aug(img):
    data = []
    img = tf.convert_to_tensor(img, dtype=tf.float32) / 255.0
    for i in range(9):
        img = tf.image.stateless_random_brightness(img, max_delta=0.02, seed=(1, 2))
        img = tf.image.stateless_random_contrast(img, lower=0.6, upper=1, seed=(1, 3))
        img = tf.image.stateless_random_flip_left_right(img, seed=(np.random.randint(100), np.random.randint(100)))
        img = tf.image.stateless_random_jpeg_quality(img, min_jpeg_quality=90, max_jpeg_quality=100, seed=(np.random.randint(100), np.random.randint(100)))
        img = tf.image.stateless_random_saturation(img, lower=0.9, upper=1, seed=(np.random.randint(100), np.random.randint(100)))
        data.append(img)
    return data

# Setup paths
#POS_PATH = os.path.join('data_all/data', 'positive')
#NEG_PATH = os.path.join('data_all/data', 'negative')
#ANC_PATH = os.path.join('data_all/data', 'anchor')

# Make th directories
# os.makedirs(POS_PATH, exist_ok=True)
# os.makedirs(NEG_PATH, exist_ok=True)
# os.makedirs(ANC_PATH, exist_ok=True)

# Move LFW Images to the following repository data/negative
if os.path.isdir("lfw"):
    for directory in os.listdir('lfw'):
        dir_path = os.path.join('lfw', directory)
        if os.path.isdir(dir_path):
            for file in os.listdir(dir_path):
                EX_PATH = os.path.join('lfw', directory, file)
                NEW_PATH = os.path.join(NEG_PATH, file)
                os.replace(EX_PATH, NEW_PATH)

#Collect Positive and Anchor Classes
#Establish a connection to the webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    #height, width, _ = frame.shape

    # Cut down frame to 250x250px
    frame = frame[100:100+250, 515:515+250, :]

    # Collect anchors
    if cv2.waitKey(1) & 0XFF == ord('a'):
        print("a")
        # Create unique file path
        img_name = os.path.join(NEG_PATH, "{}.jpg".format(uuid.uuid1()))
        # Write out anchor image
        cv2.imwrite(img_name, frame)

    # Collect positives
    if cv2.waitKey(1) & 0XFF == ord('p'):
        print("p")
        # Create unique file path
        img_name = os.path.join(POS_PATH, "{}.jpg".format(uuid.uuid1()))
        # Write out positive image
        cv2.imwrite(img_name, frame)


    # Show images back to screen
    cv2.imshow("Image Collection", frame)

    # Breaking gracefully
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
# ReLease the webcam
cap.release()
# Close the image show frame
cv2.destroyAllWindows()

# Data augmentation
# folder_name = POS_PATH
folder_name = os.path.join("../App_kivy/application_data", "valid_image")
for file_name in os.listdir(os.path.join(folder_name)):
    img_path = os.path.join(folder_name, file_name)
    if file_name.startswith('.'):
        print(f"Skipping hidden file: {file_name}")
        continue
    img = cv2.imread(img_path)
    augmented_images = data_aug(img)

    for image in augmented_images:
        img_out = (image.numpy() * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(folder_name, "{}.jpg".format(uuid.uuid1())), img_out)
