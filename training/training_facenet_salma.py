import os
import argparse
import cv2
import mediapipe as mp
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import numpy as np
import pickle
from PIL import Image
from numpy import asarray, expand_dims

mp_face_detection = mp.solutions.face_detection
MyFaceNet = FaceNet()

def extract_embeddings(filename):
    # Function to extract face embeddings from images
    image = cv2.imread(filename)

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
                gbr = Image.fromarray(gbr)  # convert from OpenCV to PIL
                gbr_array = asarray(gbr)

                face = gbr_array[y1:y2, x1:x2]

                face = Image.fromarray(face)
                face = face.resize((160, 160))
                face = asarray(face)

                face = expand_dims(face, axis=0)
                signature = MyFaceNet.embeddings(face)

                return signature

    return None

def main(data_path):
    filenames = []
    for path, subdirs, files in os.walk(data_path):
        for name in files:
            if name == ".DS_Store" or name == "foto/.jpg":
                continue
            filenames.append(os.path.join(path, name))

    face_embeddings = []
    face_names = []

    for filename in filenames:
        signature = extract_embeddings(filename)
        if signature is not None:
            name = filename.split(os.path.sep)[-2]
            face_embeddings.append(signature)
            face_names.append(name)

    data = {"embeddings": face_embeddings, "names": face_names}

    le = LabelEncoder()
    labels = le.fit_transform(data["names"])

    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(np.concatenate(face_embeddings), labels)

    # Save the recognizer model
    with open('./model/facenet/recognizer_facenet.pickle', "wb") as f:
        pickle.dump(recognizer, f)

    # Save the label encoder
    with open('./model/facenet/le_facenet.pickle', "wb") as f:
        pickle.dump(le, f)

    print("Finish Training FaceNet")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data directory")
    args = parser.parse_args()

    main(args.data_path)
