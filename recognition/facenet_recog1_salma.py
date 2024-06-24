import argparse
from collections import Counter
import shutil
import cv2
from flask import Flask, Response, render_template, request, jsonify
from flask_socketio import SocketIO, emit
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
import threading

import requests

# Initialize
names_probs = defaultdict(list)
app = Flask(__name__)
socketio = SocketIO(app)

# Load MediaPipe face detection model
mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.8)

# Load FaceNet model
MyFaceNet = FaceNet()
face_timer = None
last_recognized_name = None
last_recognized_prob = None
last_recognized_nameid = None
recognition_paused = False

# Load FaceNet recognizer and label encoder
with open('./training/recognizer_facenet.pickle', 'rb') as f:
    recognizer = pickle.load(f)

with open('./training/le_facenet.pickle', 'rb') as f:
    le = pickle.load(f)

def get_next_no():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            port='3306',
            database='absen'
        )
        cursor = connection.cursor()
        cursor.execute("SELECT MAX(No) FROM presence")
        result = cursor.fetchone()
        next_no = result[0] + 1 if result[0] else 1
        cursor.close()
        connection.close()
        return next_no
    except Exception as e:
        print("Error:", e)
        return None

def Name(emp_id, database_name):
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
        cursor.execute("SELECT Nama FROM siswa WHERE Id = %s", (emp_id,))
        result = cursor.fetchone()
        cursor.close()
        connection.close()

        if result:
            return result[0]
        else:
            return None
    except Exception as e:
        print("Error fetching name from database:", e)
        return None


def Attendance(emp_id, database_name, timestamp_now, max_pred):
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
        cursor.execute("SELECT Nama FROM siswa WHERE Id = %s", (emp_id,))
        result = cursor.fetchone()

        if result:
            emp_name = result[0]
            next_no = get_next_no()
            cursor.execute("""
                INSERT INTO presence (No, Timestamp, Id_Camera, Id, Flag, Engine, Error)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (next_no, timestamp_now, camera_id, emp_id, 'True', 'FaceNet', '0'))
            connection.commit()
            cursor.close()
            connection.close()
            return emp_name
        
    except Exception as e:
        print("Error:", e)
        return None

def log_error(emp_id, timestamp_now, real_name, operator_code):
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            port='3306',
            database='absen'
        )

        emp_id = int(emp_id)
        cursor = connection.cursor()
        cursor.execute("SELECT MAX(Error) FROM presence")
        result = cursor.fetchone()
        error_no = int(result[0]) + 1 if result[0] else 1
        
        next_no = get_next_no()
        cursor.execute("""
            INSERT INTO presence (No, Timestamp, Id_Camera, Id, Flag, Engine, Error)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (next_no, timestamp_now, camera_id, emp_id, 'False', 'FaceNet', error_no))
        connection.commit()

        cursor.execute("""
            INSERT INTO wrong (Error, Timestamp, Id_Camera, Id_Salah, Id_Benar, Engine, Kode_Operator)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (error_no, timestamp_now, camera_id, emp_id, real_name, 'FaceNet', operator_code))
        connection.commit()

        cursor.close()
        connection.close()
        return True

    except Exception as e:
        print("Error logging to database:", e)
        return False

def FaceNet_recog_mp(frame, now):
    global face_timer, last_recognized_name, last_recognized_prob, last_recognized_nameid, recognition_paused, names_probs

    height, width, _ = frame.shape
    x, y = int(0.25*width), int(0.8*height)
    w, h = int(0.5*width), int(0.08*height)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), -1)

    if recognition_paused:
        text = "{} - {}%".format(last_recognized_name, last_recognized_prob)
        (text_width, text_height), baseline = cv2.getTextSize(text,  cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        text_x = x + (w - text_width) // 2
        text_y = y + (h + text_height) // 2
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
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

    # Initialize text
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
            face_timer = datetime.datetime.now()
            names_probs.clear()
            return frame, None, None
        
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2) # face rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), -1) # text background rectangle
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

        if (datetime.datetime.now() - face_timer).total_seconds() < 5:
            text = str(datetime.datetime.now() - face_timer)
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = x + (w - text_width) // 2
            text_y = y + (h + text_height) // 2
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, thickness)

        else:
            if names_probs:
                most_common_nameid = max(names_probs, key=lambda k: (len(names_probs[k]), max(names_probs[k])))
                max_prob = max(names_probs[most_common_nameid])
                
                if most_common_nameid != 'Unknown':  # Check if a valid name is detected
                    timestamp_now = now.strftime('%Y-%m-%d_%H:%M:%S')
                    most_common_name = Name(most_common_nameid, 'absen')

                    # Construct folder and file names only if most_common_name is not None
                    if most_common_name:
                        folder_name = os.path.join('Capture_Result', most_common_name)
                        os.makedirs(folder_name, exist_ok=True)
                        photo_name = f"{timestamp_now}_{most_common_name}_{max_prob}.jpg"
                        cv2.imwrite(os.path.join(folder_name, photo_name), frame)

                else:
                    names_probs.clear()

                last_recognized_prob = max_prob
                last_recognized_name = most_common_name
                last_recognized_nameid = most_common_nameid

                text = "{} - {}%".format(most_common_name, max_prob)
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                text_x = x + (w - text_width) // 2
                text_y = y + (h + text_height) // 2

                cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, thickness)

        print(names_probs)

        # elif (datetime.datetime.now() - face_timer).total_seconds() > 6:
        #     text1 = str(datetime.datetime.now() - face_timer)
        #     (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        #     text_x = x + (w - text_width) // 2
        #     text_y = y + (h + text_height) // 2

        #     cv2.putText(frame, text1, (text_x, text_y), font, font_scale, text_color, thickness)
        
        # elif (datetime.datetime.now() - face_timer).total_seconds() > 10:
        #     text2 = "Waiting for next recognition"
        #     (text_width, text_height), baseline = cv2.getTextSize(text2, font, font_scale, thickness)
        #     text_x = x + (w - text_width) // 2
        #     text_y = y + (h + text_height) // 2

        #     cv2.putText(frame, text2, (text_x, text_y), font, font_scale, text_color, thickness)
   
    return frame, most_common_name, prob

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('show.html')

def notify_gui_of_error(message):
    try:
        requests.post('http://localhost:5001/report_error', json={'error_message': message})
    except Exception as e:
        print(f"Failed to notify GUI: {e}")

@app.route('/wrong_recognition', methods=['POST'])
def wrong_recognition():
    global last_recognized_nameid, last_recognized_name, last_recognized_prob

    if not last_recognized_nameid or not last_recognized_name:
        return jsonify(success=False, message="No recognized name to correct")

    data = request.get_json()
    real_name = data.get('real_name')
    operator_code = data.get('operator_code')

    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            port='3306',
            database='absen'
        )

        cursor = connection.cursor()
        cursor.execute("SELECT Id FROM siswa WHERE Nama = %s", (real_name,))
        result = cursor.fetchone()

        if not result:
            return jsonify(success=False, message="Name not found in database")

        real_name_id = result[0]

        # Ensure the ID values are integers and within the acceptable range
        try:
            real_name_id = int(real_name_id)
            last_recognized_nameid = int(last_recognized_nameid)
        except ValueError:
            return jsonify(success=False, message="Invalid ID format")

        timestamp_now = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        if log_error(last_recognized_nameid, timestamp_now, real_name_id, operator_code):
            last_recognized_nameid, last_recognized_name, last_recognized_prob = None, None, None
            notify_gui_of_error(f"Error: {real_name} was not {last_recognized_name}")
            return jsonify(success=True, message="Successfully corrected real identity")
        else:
            return jsonify(success=False, message="Error correcting identity")

    except Exception as e:
        print("Error:", e)
        return jsonify(success=False, message="Database error")

@app.route('/pause_recognition', methods=['POST'])
def pause_recognition():
    global recognition_paused
    recognition_paused = True
    return jsonify(success=True, message="System paused temporarily")

@app.route('/resume_recognition', methods=['POST'])
def resume_recognition():
    global recognition_paused, names_probs
    names_probs.clear()
    recognition_paused = False
    return jsonify(success=True, message="Real identities successfully entered. System resumed")

def gen_frames():
    global last_recognized_name, last_recognized_prob, last_recognized_nameid
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

        now = datetime.datetime.now()
        result, max_name, max_pred = FaceNet_recog_mp(frame, now)

        if max_name and max_pred and max_pred > 80:
            timestamp_now = now.strftime('%Y-%m-%d_%H-%M-%S')
            folder_path = 'Capture_Data'
            os.makedirs(folder_path, exist_ok=True)

            folder_name = os.path.join(folder_path, max_name)
            photo_name = f"{timestamp_now}_{max_name}_{max_pred}.jpg"
            cv2.imwrite(os.path.join(folder_name, photo_name), frame)

        ret, buffer = cv2.imencode('.jpg', result)

        if ret:
            frame_html = buffer.tobytes()  # Convert numpy array to string
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_html + b'\r\n')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/present_now', methods=['POST'])
def present_now():
    global last_recognized_name, last_recognized_prob
    if not last_recognized_name:
        return jsonify(success=False, message="No recognized face to record attendance")

    timestamp_now = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    most_common_name = Attendance(last_recognized_nameid, 'absen', timestamp_now, last_recognized_prob)

    if most_common_name:
        # Send success message to the server
        socketio.emit('message', {"message": f"{most_common_name} successfully recorded at {timestamp_now}"})
        return jsonify(success=True, message=f"{most_common_name} successfully recorded at {timestamp_now}")
    else:
        return jsonify(success=False, message="Error recording attendance")

if __name__ == "__main__":
    global camera_id
    parser = argparse.ArgumentParser(description='Face recognition using Facenet and Mediapipe')
    parser.add_argument('--camera_id', type=str, required=True, help='Camera ID')
    parser.add_argument('--url', type=str, required=True, help='URL of the video stream')
    args = parser.parse_args()
    url = 1 # args.url
    camera_id = args.camera_id
    socketio.run(app, host='0.0.0.0', port=5100, debug=True)

# if __name__ == "__main__":
#     global camera_id
#     url = 1
#     camera_id = 2
#     socketio.run(app, host='0.0.0.0', port=5100, debug=True)