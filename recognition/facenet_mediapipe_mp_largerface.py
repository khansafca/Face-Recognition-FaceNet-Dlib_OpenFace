import argparse
from collections import Counter
import shutil
import cv2
from flask import Flask, Response, render_template, request, jsonify
import json
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
import re
from collections import defaultdict

import requests

# Initialize
names_probs = defaultdict(list)
app = Flask(__name__)

# Load MediaPipe face detection model
mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.8)

# Load FaceNet model
MyFaceNet = FaceNet()
face_timer = None
last_recognized_name = None
last_recognized_prob = None
last_recognized_nameid = None
capture_active = True

# Load FaceNet recognizer and label encoder
with open('./training/recognizer_facenet.pickle', 'rb') as f:
    recognizer = pickle.load(f)

with open('./training/le_facenet.pickle', 'rb') as f:
    le = pickle.load(f)

def Attendance(emp_id, database_name, timestamp_now, max_pred):
    """Records the attendance of an employee in the database."""
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            port='3306',
            database=database_name
        )

        emp_id = int(emp_id)
        cursor = connection.cursor()
        cursor.execute("SELECT name FROM employee WHERE id = %s", (emp_id,))
        result = cursor.fetchone()

        if result:
            emp_name = result[0]

            # Always update the timestamp for demonstration
            if max_pred > 0.8:
                new_timestamp_now = str(timestamp_now) + '_' + str(max_pred)
                cursor.execute("UPDATE primer_data SET FaceNet = %s WHERE EmpID = %s", (new_timestamp_now, emp_id))
                connection.commit()
                print(f"Updated FaceNet for {emp_name} to {new_timestamp_now}")

            cursor.close()
            connection.close()
            return emp_name
        
    except Exception as e:
        print("Error:", e)
        return None

def log_error(database_name, name, prob, emp_id, real_name):
    """Logs the error details to the data_sys table."""
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            port='3306',
            database=database_name
        )
        
        cursor = connection.cursor()
        timestamp_now = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        emp_id = int(emp_id)
        # real_name = str(real_name)
        error_message = f'FaceNet_{timestamp_now}_{str(real_name)}_bukan_{name}'

        query = f"SELECT Error FROM primer_data WHERE EmpID = %s"
        cursor.execute(query, (emp_id,))
        result = cursor.fetchall()

        if result == [('',)]:
            cursor.execute("UPDATE primer_data SET Error = %s WHERE EmpID = %s", (error_message, emp_id))
        else:
            cursor.execute("UPDATE primer_data SET Error = CONCAT(Error, ', ', %s) WHERE EmpID = %s", (error_message, emp_id))
        
        connection.commit()
        cursor.close()
        connection.close()
        return True
    
    except Exception as e:
        print("Error logging to database:", e)
        return False

def FaceNet_recog_mp(frame, now):
    global face_timer, last_recognized_name, last_recognized_prob, last_recognized_nameid, capture_active
    if not capture_active:
        return frame, None, None

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face_detection.process(rgb_frame)
    prob = 0
    max_prob = 0

    if not results.detections:
        face_timer = None
        names_probs.clear()
        return frame, None, None

    if face_timer is None:
        face_timer = datetime.datetime.now()
    
    height, width, _ = frame.shape
    x, y = int(0.25*width), int(0.8*height)
    w, h = int(0.5*width), int(0.08*height)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,0), -1)

    # Inisiasi teks
    text_color, font, font_scale, thickness = (255, 255, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    most_common_name, max_face_area, largest_face_bbox = 'Unknown', 0, None

    for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = frame.shape
        bbox = [
            max(int(bboxC.xmin * iw), 0),
            max(int(bboxC.ymin * ih), 0),
            int(bboxC.width * iw),
            int(bboxC.height * ih)]
        
        face_area = bbox[2] * bbox[3]
        
        if face_area > max_face_area:
            max_face_area = face_area
            largest_face_bbox = bbox

    if largest_face_bbox:
        bbox = largest_face_bbox
        face = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        if face.size < 75000:
            # print('Empty face size')
            face_timer = datetime.datetime.now()
            names_probs.clear()
            return frame, None, None
        
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2) # kotak wajah
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,0), -1) # kotak background text
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = Image.fromarray(face)
        face = face.resize((160, 160))
        face = asarray(face)
        face = expand_dims(face, axis=0)

        signature = MyFaceNet.embeddings(face)
        preds = recognizer.predict_proba(signature)
        name = le.inverse_transform([np.argmax(preds)])[0]
        prob = round(preds[0][np.argmax(preds)] * 100, 2)

        if prob < 80:
            name = 'Unknown'
        else:
            names_probs[name].append(prob)

        if (datetime.datetime.now() - face_timer).total_seconds() > 2.5:
            if names_probs:
                most_common_nameid = max(names_probs, key=lambda k: (len(names_probs[k]), max(names_probs[k])))
                max_prob = max(names_probs[most_common_nameid])
                
                if most_common_nameid != 'Unknown':  # Check if a valid name is detected
                    timestamp_now = now.strftime('%Y-%m-%d_%H:%M:%S')
                    most_common_name = Attendance(most_common_nameid, 'attendance', timestamp_now, max_prob)

                    # Construct folder and file names only if most_common_name is not None
                    if most_common_name:
                        folder_name = os.path.join('Capture_Result', most_common_name)
                        os.makedirs(folder_name, exist_ok=True)
                        photo_name = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_{most_common_name}_{max_prob}.jpg"
                        cv2.imwrite(os.path.join(folder_name, photo_name), frame)

                else:
                    names_probs.clear()

                # print('names_probs = ', names_probs)
                last_recognized_prob = max_prob
                last_recognized_name = most_common_name
                last_recognized_nameid = most_common_nameid

                text = "{} - {}%".format(most_common_name, max_prob)
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                text_x = x + (w - text_width) // 2
                text_y = y + (h + text_height) // 2

                cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, thickness)

                # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), -1)
                # cv2.putText(frame, text, (x, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        elif (datetime.datetime.now() - face_timer).total_seconds() < 2:
            text = str(datetime.datetime.now() - face_timer)
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = x + (w - text_width) // 2
            text_y = y + (h + text_height) // 2

            cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, thickness)
            # cv2.putText(frame, str(datetime.datetime.now() - face_timer), (50, 50), font, font_scale, text_color, thickness)

    return frame, most_common_name, max_prob

def notify_gui_of_error(message):
    try:
        requests.post('http://localhost:5001/report_error', json={'error_message': message})
    except Exception as e:
        print(f"Failed to notify GUI: {e}")

@app.route('/report_error', methods=['POST'])
def report_error():
    global capture_active, last_recognized_name, last_recognized_prob, last_recognized_nameid, real_name
    data = request.get_json()
    real_name = data.get('real_name')
    
    if last_recognized_name and last_recognized_prob and last_recognized_nameid:
        capture_active = False  # Pause capture
        success = log_error('attendance', last_recognized_name, last_recognized_prob, last_recognized_nameid, real_name)
        if success:
            notify_gui_of_error(f"Error reported: {real_name} was not {last_recognized_name}")
        capture_active = True  # Reactivate the capture promptly
        return jsonify({'success': success})
    return jsonify({'success': False})

def gen_frames():
    cap = cv2.VideoCapture(url)
    time.sleep(3)  # Initial delay for the camera to warm up

    folder_path = 'Capture_Result'
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

    os.makedirs(folder_path, exist_ok=True)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if capture_active:  # Only process the frame if capture is active
            now = datetime.datetime.now()
            result, max_name, max_pred = FaceNet_recog_mp(frame, now)

            if max_name and max_pred and max_pred > 80:
                timestamp_now = now.strftime('%Y-%m-%d_%H-%M-%S')
                folder_path = 'Capture_Data'
                os.makedirs(folder_path, exist_ok=True)

                folder_name = os.path.join(folder_path, max_name)
                photo_name = f"{timestamp_now}_{max_name}_{max_pred}.jpg"
                cv2.imwrite(os.path.join(folder_name, photo_name), frame)

        else:
            result = frame  # Display the same frame when capture is not active
            timestamp_now = now.strftime('%Y-%m-%d_%H-%M-%S')
            folder_path = 'Capture_Error'
            os.makedirs(folder_path, exist_ok=True)

            folder_name = os.path.join(folder_path, max_name)
            photo_name = f"Error_{timestamp_now}_{real_name}_bukan_{max_name}_.jpg"
            cv2.imwrite(os.path.join(folder_name, photo_name), frame)

        ret, buffer = cv2.imencode('.jpg', result)

        if ret:
            frame_html = buffer.tobytes()  # Convert numpy array to string
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_html + b'\r\n')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('show.html')

if __name__ == "__main__":
    url = 1
    app.run(host='0.0.0.0', port=5000, debug=True)

# def gen_frames():
#     """Generates frames from the webcam and processes them for face and gesture recognition."""
#     cap = cv2.VideoCapture(1)
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         result, max_name, max_pred = FaceNet_recog_mp(frame, names_probs)
#         cv2.imshow("result", result)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     gen_frames()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Face recognition using Facenet and Mediapipe')
#     parser.add_argument('--url', type=str, required=True, help='URL of the video stream')
#     args = parser.parse_args()
#     url = args.url
#     # url = 1
#     # CAMERA_URL = 'http://192.168.14.214:4747/video' # android salma
#     app.run(host='0.0.0.0', port=5000, debug=True)