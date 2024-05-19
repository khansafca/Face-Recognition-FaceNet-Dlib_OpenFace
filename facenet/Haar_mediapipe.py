import os
from PIL import Image
from numpy import asarray, expand_dims
from keras.models import load_model
import numpy as np
import tensorflow as tf
import pickle
import cv2
import mediapipe as mp
from keras_facenet import FaceNet

directory = "/Users/khansafca/Documents/gui_fixed/facenet_recog"

mp_face_detection = mp.solutions.face_detection
MyFaceNet = FaceNet()

folder = '/Users/khansafca/Documents/gui_fixed/database'
database = {}

# Walk through the directory tree
for path, subdirs, files in os.walk(folder):
    for filename in files:
        if filename == ".DS_Store":
            continue
        file_path = os.path.join(path, filename)
        if not os.path.isfile(file_path):
            print(f"Warning: Image not found - {file_path}")
            continue
        gbr1 = cv2.imread(file_path)

        mp_face_detection = mp.solutions.face_detection
        mp_drawing = mp.solutions.drawing_utils
        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(cv2.cvtColor(gbr1, cv2.COLOR_BGR2RGB))

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = gbr1.shape
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                    x1, y1, width, height = bbox
                    x1, y1 = abs(x1), abs(y1)
                    x2, y2 = x1 + width, y1 + height

                    gbr = cv2.cvtColor(gbr1, cv2.COLOR_BGR2RGB)
                    gbr = Image.fromarray(gbr)  # konversi dari OpenCV ke PIL
                    gbr_array = asarray(gbr)

                    face = gbr_array[y1:y2, x1:x2]

                    face = Image.fromarray(face)
                    face = face.resize((160, 160))
                    face = asarray(face)

                    face = expand_dims(face, axis=0)
                    signature = MyFaceNet.embeddings(face)

                    database[os.path.splitext(filename)[0]] = signature


file_path = os.path.join(directory, "encode_mediapipe_facenet.pkl")
# Open the file and save the data
with open(file_path, "wb") as myfile:
    pickle.dump(database, myfile)

database
