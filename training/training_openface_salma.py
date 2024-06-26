import argparse
import cv2
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import face_recognition

# Load face detection model
proto_path = './model/openface/deploy.prototxt'
model_path = './model/openface/res10_300x300_ssd_iter_140000.caffemodel'
face_detector = cv2.dnn.readNetFromCaffe(prototxt=proto_path, caffeModel=model_path)

# Load face recognition model
recognition_model = './model/openface/openface_nn4.small2.v1.t7'
face_recognizer = cv2.dnn.readNetFromTorch(model=recognition_model)

# Function to extract face embeddings using face_recognition library
def extract_embeddings(image_path):
    image = cv2.imread(image_path)

    # Convert the frame to grayscale and apply histogram equalization
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    equalized = cv2.equalizeHist(gray)
    equalized_bgr = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

    image_blob = cv2.dnn.blobFromImage(cv2.resize(equalized_bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)

    # Resize face to 96x96 (required input size for openface model)
    # face_resized = cv2.resize(face, (96, 96))

    # # Preprocess the face for the face_recognizer
    # face_blob = cv2.dnn.blobFromImage(face_resized, 1.0/255, (96, 96), (0, 0, 0), swapRB=True, crop=False)

    return image, image_blob

def main(data_path):
    filenames = []
    for path, subdirs, files in os.walk(data_path):
        for name in files:
            if name == ".DS_Store" or name == "dataset/.jpg":
                continue
            filenames.append(os.path.join(path, name))

    face_embeddings = []
    face_names = []

    for filename in filenames:
        print("Processing image {}".format(filename))

        # Extract embeddings using face_recognition and face_recognizer models
        image, face_blob = extract_embeddings(filename)
        if face_blob is not None:
            face_detector.setInput(face_blob)
            face_detections = face_detector.forward()

            if face_detections.shape[2] == 0:
                continue

            i = np.argmax(face_detections[0, 0, :, 2])
            confidence = face_detections[0, 0, i, 2]
            height, width, _ = image.shape

            if confidence >= 0.65:
                box = face_detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")

                # Ensure the ROI is within image bounds
                if startX < 0 or startY < 0 or endX > width or endY > height:
                    continue

                face = image[startY:endY, startX:endX]

                if face.size == 0:  # Check if face extraction failed
                    continue

                # Resize the face to 96x96 for the model
                face_resized = cv2.resize(face, (96, 96))

                # Preprocess the face for the face_recognizer
                face_blob = cv2.dnn.blobFromImage(face_resized, 1.0/255, (96, 96), (0, 0, 0), swapRB=True, crop=False)

                face_recognizer.setInput(face_blob)
                face_recognitions = face_recognizer.forward()

                name = filename.split(os.path.sep)[-2]

                face_embeddings.append(face_recognitions.flatten())
                face_names.append(name)

    data = {"embeddings": face_embeddings, "names": face_names}

    le = LabelEncoder()
    labels = le.fit_transform(data["names"])

    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    # Save the recognizer model
    with open('./model/openface/recognizer_openface.pickle', "wb") as f:
        pickle.dump(recognizer, f)

    # Save the label encoder
    with open('./model/openface/le_openface.pickle', "wb") as f:
        pickle.dump(le, f)

    print("Finish Training OpenFace")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data directory")
    args = parser.parse_args()

    main(args.data_path)

