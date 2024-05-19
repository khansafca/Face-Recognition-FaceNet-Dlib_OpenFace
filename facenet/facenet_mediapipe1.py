import cv2
import mediapipe as mp
from datetime import datetime
from PIL import Image
import numpy as np
from numpy import asarray, expand_dims
from keras_facenet import FaceNet
import pickle
import mysql.connector
import re
import os

# Load MediaPipe face detection model
mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.7)

# Load FaceNet model
MyFaceNet = FaceNet()

# Load face embeddings database
# Update the path to the file here
file_path = "/Users/khansafca/Documents/3Face_Recognition_/facenet_recog/encode_mediapipe_facenet.pkl"
with open(file_path, "rb") as myfile:
    database = pickle.load(myfile)

# Function to handle attendance and database operations
def Attendance(emp_id, database_name, timestamp_now):
    try:
        # Establishing connection to the database
        connection = mysql.connector.connect(
            host='172.20.10.11',
            user='root',
            password='',  # Replace with actual password
            database=database_name
        )

        # Creating a cursor to execute SQL commands
        cursor = connection.cursor()

        # Check if employee exists
        cursor.execute("SELECT EmpName, Timestamp FROM employee WHERE EmpID = %s", (emp_id,))
        result = cursor.fetchone()

        if result:
            emp_name, timestamp = result

            # Update timestamp if not already present
            if not timestamp:
                cursor.execute("UPDATE employee SET Timestamp = %s WHERE EmpID = %s", (timestamp_now, emp_id))
                connection.commit()

            cursor.close()
            connection.close()
            return emp_name
        else:
            print("Employee ID not found.")
            cursor.close()
            connection.close()
            return None
    except Exception as e:
        print("Error:", e)
        return None

def FaceNet_recog(frame):
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Convert BGR frame to RGB
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Perform face detection
    results = mp_face_detection.process(rgb_frame)

    if not results.detections:
        return frame, None

    identity = "Unknown"
    detection = results.detections[0]  # Only consider the first detected face

    bboxC = detection.location_data.relative_bounding_box
    ih, iw, _ = frame.shape
    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
    cv2.rectangle(frame, bbox, (0, 255, 0), 2)

    # Extract face region
    face = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]

    # Check if face is not empty
    if face.size == 0:
        return frame, None

    # Preprocess the face for FaceNet
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = Image.fromarray(face)
    face = face.resize((160, 160))
    face = asarray(face)
    face = expand_dims(face, axis=0)

    # Get face signature using FaceNet
    signature = MyFaceNet.embeddings(face)

    # Recognize the face
    min_dist = 100
    identity = 'Unknown'

    for key, value in database.items():
        dist = np.linalg.norm(value - signature)
        if dist < min_dist:
            min_dist = dist
            identity = key
    
    # Display the recognized identity and similarity distance on the frame
    similarity = round((1 - min_dist) * 100, 2)
    if similarity < 10:
        identity = 'Unknown'

    if identity != 'Unknown':
        timestamp_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        matches = re.findall(r'\d+(?=\s*\()', identity)
        identity = matches[0] if matches else None
        #identity = Attendance(identity, 'attendance_system', timestamp_now)

    text = f"{identity} - similarity: {similarity}%"
    cv2.putText(frame, text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return frame, identity


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        result, text = FaceNet_recog(frame)
        cv2.imshow('FaceNet', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # change to waitKey(1)
            break

    cap.release()
    cv2.destroyAllWindows()