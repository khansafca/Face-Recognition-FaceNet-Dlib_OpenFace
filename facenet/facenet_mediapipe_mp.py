import cv2
import mediapipe as mp
from PIL import Image
import time
import os
import sys
import datetime
from os import listdir
import numpy as np
from numpy import asarray, expand_dims
from keras_facenet import FaceNet
import pickle
import mysql.connector
from flask import Flask, render_template, Response
import re

app = Flask(__name__)

# Load MediaPipe face detection model
mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.7)

# Load FaceNet model
MyFaceNet = FaceNet()

recognizer_path = 'model/recognizer_facenet.pickle'
with open(recognizer_path, 'rb') as f:
    recognizer = pickle.load(f)

le_path = 'model/le_facenet.pickle'
with open(le_path, 'rb') as f:
    le = pickle.load(f)

def Attendance(emp_id, database_name, timestamp_now, max_pred):
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            port='3308',
            database=database_name
        )

        cursor = connection.cursor()
        cursor.execute("SELECT name, FaceNet FROM users WHERE id = %s", (emp_id,))
        result = cursor.fetchone()

        if result:
            emp_name, timestamp = result
            # Always update the timestamp for demonstration
            if max_pred > 0.83:
                new_timestamp_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                cursor.execute("UPDATE users SET FaceNet = %s WHERE id = %s", (new_timestamp_now, emp_id))
                connection.commit()
                print(f"Updated timestamp for {emp_name} to {new_timestamp_now}")

            cursor.close()
            connection.close()
            return emp_name
        else:
            print(f"ID {emp_id} not found.")
            cursor.close()
            connection.close()
            return None
    except Exception as e:
        print("Error:", e)
        return None



def FaceNet_recog(frame):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert BGR frame to RGB
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Perform face detection
    results = mp_face_detection.process(rgb_frame)

    if not results.detections:
        return frame, None

    max_pred = 0
    max_bbox = None
    max_name = None

    for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = frame.shape
        bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

        # Extract face region
        face = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]

        # Check if face is not empty
        if face.size == 0:
            continue

        # Preprocess the face for FaceNet
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = Image.fromarray(face)
        face = face.resize((160, 160))
        face = asarray(face)
        face = expand_dims(face, axis=0)

        # Get face signature using FaceNet
        signature = MyFaceNet.embeddings(face)

        preds = recognizer.predict_proba(signature)
        name = le.inverse_transform([np.argmax(preds)])

        # Check if this prediction is the highest
        if preds[0][np.argmax(preds)] > max_pred:
            max_pred = preds[0][np.argmax(preds)]
            max_bbox = bbox
            max_name = name

    # Draw the bounding box of the face with the highest prediction
    if max_bbox is not None:
        cv2.rectangle(frame, max_bbox, (0, 255, 0), 2)
        max_name = Attendance(max_name[0], 'absen', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), max_pred)
        if max_name:  # Ensure max_name is not None before formatting text
            text = "{}: {:.2f}%".format(max_name, max_pred * 100)
            cv2.putText(frame, text, (max_bbox[0], max_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    return frame, max_name, max_pred


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    time.sleep(3)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        now = datetime.datetime.now()
        dt_string = now.strftime("%Y-%m-%d:%H.%M.%S")
        result, max_name, max_pred = FaceNet_recog(frame)
        cv2.imshow('FaceNet', result)

        if max_name is not None and max_pred > 0.85:
            folder_path = f'Capture/{max_name}'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            file_path = f'{folder_path}/{dt_string}.jpg'
            try:
                cv2.imwrite(file_path, frame)
                print(f'Saved: {file_path}')
            except Exception as e:
                print(f'Error saving file: {e}')

            time.sleep(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()