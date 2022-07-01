import cv2
import os
import numpy as np
import tensorflow as tf
from skimage.transform import resize
import winsound

frequency = 1500
duration = 500

font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN
Xception = tf.keras.models.load_model('drivesafe/xceptionNew.h5')

cascPathface = os.path.dirname(cv2.__file__) + \
    "/data/haarcascade_frontalface_alt2.xml"
# cascPatheyes = os.path.dirname(cv2.__file__) + \
#     "/data/haarcascade_eye_tree_eyeglasses.xml"

faceCascade = cv2.CascadeClassifier(cascPathface)
eyeCascade = cv2.CascadeClassifier(
    'drivesafe/haarcascade_eye_tree_eyeglasses.xml')

status = ['Anger', 'Closed_Eyes', 'Happy', 'Neutral', 'Open_Eyes', 'Yawn']


def image_pred(image):
    current_status = ""
    current_status_eye = ""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        faceROI = image[y:y+h, x:x+w]
        resized_face = resize(faceROI, (224, 224, 3))
        final_face = np.expand_dims(resized_face, axis=0)
        pred_face = Xception.predict(final_face)
        current_status = status[np.argmax(pred_face)]
        cv2.putText(image, current_status, (100, 150),
                    font, 3, (0, 0, 255), 2, cv2.LINE_4)
        eyes = eyeCascade.detectMultiScale(faceROI)

        for (x2, y2, w2, h2) in eyes:
            cv2.rectangle(faceROI, (x2, y2),
                          (x2 + w2, y2 + h2), (0, 255, 0), 2)
            eyeROI = faceROI[y2:y2+h2, x2:x2+w2]
            resized_face_eye = resize(eyeROI, (224, 224, 3))
            final_face_eye = np.expand_dims(resized_face_eye, axis=0)
            pred_face_eye = Xception.predict(final_face_eye)
            current_status_eye = status[np.argmax(pred_face_eye)]
            cv2.putText(faceROI, current_status_eye, (100, 150),
                        font, 3, (0, 0, 255), 2, cv2.LINE_4)

    _, jpeg = cv2.imencode('.jpg', image)
    return jpeg.tobytes(), current_status, current_status_eye
