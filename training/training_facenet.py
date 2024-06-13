import os
from os import listdir
from PIL import Image
from numpy import asarray, expand_dims
from matplotlib import pyplot
from keras.models import load_model
import numpy as np
import tensorflow as tf
import pickle
import cv2
import mediapipe as mp
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

mp_face_detection = mp.solutions.face_detection
MyFaceNet = FaceNet()

curr_path = os.getcwd()
data_base_path = 'database/'

filenames = []
for path, subdirs, files in os.walk(data_base_path):
    for name in files:
        filenames.append(os.path.join(path, name))

face_embeddings = []
face_names = []

for (i, filename) in enumerate(filenames):
    if filename == ".DS_Store" or filename == "foto/.jpg":
        continue

    image = cv2.imread(filename)

    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                x1, y1, width, height = bbox
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height

                gbr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                gbr = Image.fromarray(gbr)  # konversi dari OpenCV ke PIL
                gbr_array = asarray(gbr)

                face = gbr_array[y1:y2, x1:x2]

                face = Image.fromarray(face)
                face = face.resize((160, 160))
                face = asarray(face)

                face = expand_dims(face, axis=0)
                signature = MyFaceNet.embeddings(face)

                name = filename.split(os.path.sep)[-2]
                face_embeddings.append(signature)
                face_names.append(name)

data = {"embeddings": face_embeddings, "names": face_names}

le = LabelEncoder()
labels = le.fit_transform(data["names"])

recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(np.concatenate(face_embeddings), labels)

# Save the recognizer model
with open('model/facenet/recognizer_facenet.pickle', "wb") as f:
    f.write(pickle.dumps(recognizer))

# Save the label encoder
with open('model/facenet/le_facenet.pickle', "wb") as f:
    f.write(pickle.dumps(le))

f.close()
