from flask import Flask, Response, render_template, request, jsonify
import mediapipe as mp
import argparse
import cv2
from PIL import Image
import numpy as np
from numpy import asarray, expand_dims
from keras_facenet import FaceNet
import pickle
import mysql.connector
import datetime
import time
import os

app = Flask(__name__)

# Load MediaPipe face detection model
mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.7)

# Load FaceNet model
MyFaceNet = FaceNet()

# Load FaceNet recognizer and label encoder
with open('/Users/khansafca/Documents/gui_fixed/facenet_recog/recognizer_facenet.pickle', 'rb') as f:
    recognizer = pickle.load(f)

with open('/Users/khansafca/Documents/gui_fixed/facenet_recog/le_facenet.pickle', 'rb') as f:
    le = pickle.load(f)

# List to store wrong recognition messages
wrong_recognition_messages = []

def Attendance(emp_id, database_name, timestamp_now, max_pred):
    try:
        if isinstance(emp_id, str) and emp_id.isdigit():
            emp_id = int(emp_id)
        elif not isinstance(emp_id, int):
            print(f"Invalid emp_id: {emp_id} of type {type(emp_id)}")
            return None
        
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
            if max_pred > 0.9:
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

def FaceNet_recog_mp(frame):
    small_frame = frame.copy()

    # Convert BGR frame to RGB
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Perform face detection
    results = mp_face_detection.process(rgb_frame)

    if not results.detections:
        return frame, None, None

    max_pred = 0
    max_bbox = None
    max_name = None
    text = 'unknown'

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

    if max_bbox is not None:
        cv2.rectangle(frame, max_bbox, (0, 255, 0), 2)
        max_name = Attendance(max_name[0], 'absen', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), max_pred)
        if max_name:  # Ensure max_name is not None before formatting text
            text = "{}: {:.2f}%".format(max_name, max_pred * 100)
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            text_offset_x = max_bbox[0]
            text_offset_y = max_bbox[1] - 10

            if text_offset_y - text_height < 0:
                text_offset_y = max_bbox[1] + max_bbox[3] + text_height

            box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width, text_offset_y - text_height))
            cv2.rectangle(frame, box_coords[0], box_coords[1], (0, 0, 0), cv2.FILLED)
            cv2.putText(frame, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frame, text, None

def gen_frames():
    cap = cv2.VideoCapture(url)
    time.sleep(3)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result, max_name, max_pred = FaceNet_recog_mp(frame)

        ret, buffer = cv2.imencode('.jpg', result)
        if ret:
            frame = buffer.tobytes()  # Convert numpy array to string
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('show.html', wrong_recognition_messages=wrong_recognition_messages)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/wrong_recognition', methods=['POST'])
def wrong_recognition(emp_id):
    true_name = request.form.get('true_name')
    mistaken_name = request.form.get('mistaken_name')
    timestamp_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    message = f"Wrong recognition for {true_name} on {timestamp_now} mistaken by {mistaken_name}"
    
    # Append the new message to the list
    wrong_recognition_messages.append(message)
    
    # Update the MySQL database with the new wrong recognition details
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            port='3308',
            database='your_database_name'  # Replace with your actual database name
        )
        cursor = connection.cursor()
        cursor.execute(
            "UPDATE wrong SET nama_asli = %s, nama_salah = %s, timestamp_salah = %s WHERE id = %s",
            (true_name, mistaken_name, timestamp_now, emp_id)  # Replace 'your_condition_id' with your actual condition for updating
        )
        connection.commit()
        cursor.close()
        connection.close()
    except Exception as e:
        print(f"Error updating database: {e}")

    # Return the entire list of messages as JSON
    return jsonify(messages=wrong_recognition_messages)

#@app.route('/get_wrong_recognition_messages', methods=['GET'])
#def get_wrong_recognition_messages():
    #return jsonify(messages=wrong_recognition_messages)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face recognition using Facenet and Mediapipe')
    parser.add_argument('--url', type=str, required=True, help='URL of the video stream')
    args = parser.parse_args()
    url = args.url
    app.run(host='0.0.0.0', port=5100, debug=True)
