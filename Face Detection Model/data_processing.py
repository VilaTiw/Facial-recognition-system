import os
import cv2
import json
import time
import numpy as np
import tensorflow as tf
import albumentations as alb
import matplotlib.pyplot as plt

def load_image(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    return img
def load_labels(label_path):
    try:
        path = label_path.numpy() if isinstance(label_path, tf.Tensor) else label_path
        with open(path, "r", encoding="utf-8") as f:
            if f.read(1):
                f.seek(0)
                label = json.load(f)
                #print("Label: ", label)
            else:
                return [0], [0, 0, 0, 0]
        return [label.get("class", 0)], label.get("bbox", [0, 0, 0, 0])
    except Exception as e:
        print(f"Error loading labels from {label_path}: {e}")

# def load_labels(label_path):
#     with open(label_path.numpy(), "r", encoding="utf-8") as f:
#         label = json.load(f)
#     return [label["class"]], label["bbox"]

# images = tf.data.Dataset.list_files("data/images/*.jpg", shuffle=False)
# images = images.map(load_image)
# 
# #View raw images with Matplotlib
# image_generator = images.batch(4).as_numpy_iterator()
# plot_images = image_generator.next()
#
# fig, ax = plt.subplots(ncols=4, figsize=(20, 2))
# for idx, image in enumerate(plot_images):
#     ax[idx].imshow(image)
# plt.savefig("example")
# plt.show()

#Move the matching labels
# for folder in ["train", "test", "val"]:
#     for file in os.listdir(os.path.join("data", folder, "images")):
#         filename = file.split(".")[0]+".json"
#         existing_filepath = os.path.join("data", "labels", filename)
#         if os.path.exists(existing_filepath):
#             new_filepath = os.path.join("data", folder, "labels", filename)
#             os.replace(existing_filepath, new_filepath)

# #Setup Albumentations Transform Pipeline
augmentor = alb.Compose([alb.RandomCrop(width=450, height=450),
                         alb.HorizontalFlip(p=0.5),
                         alb.RandomBrightnessContrast(p=0.2),
                         alb.RandomGamma(p=0.2),
                         alb.RGBShift(p=0.2),
                         alb.VerticalFlip(p=0.5)],
                         bbox_params=alb.BboxParams(format="albumentations", label_fields=["class_labels"]))

# img = cv2.imread(os.path.join("data", "train", "images", "1a73bbac-dbd8-11ef-a406-1e7acbd0b66d.jpg"))
# with open(os.path.join("data", "train", "labels", "1a73bbac-dbd8-11ef-a406-1e7acbd0b66d.json"), "r") as f:
#     label = json.load(f)
#     print(label["shapes"][0]["points"])
a#     print(coords)
#     coords = list(np.divide(coords, [1280, 720, 1280, 720]))
#     print(coords)
#     augmented_img = augmentor(image=img, bboxes=[coords], class_labels=["Face"])
#     print(type(augmented_img))
#     print(augmented_img.keys())
#     print(augmented_img["class_labels"])
#     new_img = cv2.rectangle(augmented_img["image"],
#                             tuple(np.multiply(augmented_img["bboxes"][0][:2], [450, 450]).astype(int)),
#                             tuple(np.multiply(augmented_img["bboxes"][0][2:], [450, 450]).astype(int)),
#                             (250, 0, 0), 2)
#     plt.imshow(augmented_img["image"])
#     plt.savefig("new")

#Run Augmentation Pipeline
# for partition in ["train", "test", "val"]:
#     for image in os.listdir(os.path.join("data", partition, "images")):
#         if image.startswith("."):
#             print(f"Skipping hidden file: {image}")
#             continue
#         img = cv2.imread(os.path.join("data", partition, "images", image))
#         coords = [0, 0, 0.00001, 0.00001]
#         label_path = os.path.join("data", partition, "labels", f"{image.split('.')[0]+'.json'}")
#
#         if os.path.exists(label_path):
#             with open(label_path, "r") as f:
#                 label = json.load(f)
#             coords = [*label["shapes"][0]["points"][0], *label["shapes"][0]["points"][1]]
#             coords = list(np.divide(coords, [1280, 720, 1280, 720]))
#
#         try:
#             for x in range(60):
#                 augmented = augmentor(image=img, bboxes=[coords], class_labels=["Face"])
#                 cv2.imwrite(os.path.join("aug_data", partition, "images", f"{image.split('.')[0]}.{x}.jpg"), augmented["image"])
#
#                 annotation = {}
#                 annotation["image"] = image
#
#                 if os.path.exists(label_path)
#                     if len(augmented["bboxes"]) == 0:
#                         annotation["bbox"] = [0, 0, 0, 0]
#                         annotation["class"] = 0
#                     else:
#                         annotation["bbox"] = augmented["bboxes"][0]
#                         annotation["class"] = 1
#                 else:
#                     annotation["bbox"] = [0, 0, 0, 0]
#                     annotation["class"] = 0
#
#                 with open(os.path.join("aug_data", partition, "labels", f"{image.split('.')[0]}.{x}.json"), "w") as f:
#                     json.dump(annotation, f)
#         except Exception as e:
#             print(e)

