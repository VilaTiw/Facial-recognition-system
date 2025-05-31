import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

facetracker = load_model("../Face Auth App/facetracker.h5")

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[135:585, 415:865]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120, 120))

    y_hat = facetracker.predict(np.expand_dims(resized/255, 0))
    sample_coords = y_hat[1][0]
    scaled_coords = np.multiply(sample_coords, [450, 450, 450, 450]).astype(int)

    if y_hat[0] > 0.5:
        #Main rectangle
        cv2.rectangle(frame,
                      tuple(scaled_coords[:2]),
                      tuple(scaled_coords[2:]),
                      (255, 0, 0), 2)
        #Label rectangle
        cv2.rectangle(frame,
                      tuple(np.add(scaled_coords[:2], [0, -30])),
                      tuple(np.add(scaled_coords[:2], [80, 0])),
                      (255, 0, 0), -1)
        #Label
        cv2.putText(frame, "Face", tuple(np.add(scaled_coords[:2], [0, -5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("FaceTracker", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.imwrite("Tracked_Face.jpg", frame)
        break

cap.release()
cv2.destroyAllWindows()

