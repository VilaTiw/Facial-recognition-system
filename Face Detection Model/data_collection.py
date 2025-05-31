import os
import time
import uuid
import cv2

IMAGES_PATH = os.path.join("..", "Siamese Model", "data_all", "data", "testing")
number_images = 60

cap = cv2.VideoCapture(0)
for imgNumber in range(number_images):
    print("Collecting image {}".format(imgNumber))
    ret, frame = cap.read()
    imgName = os.path.join(IMAGES_PATH, "{}.jpg".format(uuid.uuid1()))
    cv2.imwrite(imgName, frame)
    cv2.imshow("frame", frame)
    time.sleep(1)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()