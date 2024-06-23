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
with open('/Users/khansafca/Documents/gui_fixed/facenet_recog/recognizer_facenet.pickle', 'rb') as f:
    recognizer = pickle.load(f)

with open('/Users/khansafca/Documents/gui_fixed/facenet_recog/le_facenet.pickle', 'rb') as f:
    le = pickle.load(f)

def get_next_no():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            port='3308',
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
        cursor.execute("SELECT Nama FROM siswa WHERE Id = %s", (emp_id,))
        result = cursor.fetchone()

        if result:
            emp_name = result[0]
            next_no = get_next_no()
            cursor.execute("""
                INSERT INTO presence (No, Timestamp, Id_Camera, Id, Flag, Engine, Error)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (next_no, timestamp_now, '1', emp_id, 'True', 'FaceNet', '0'))
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
            port='3308',
            database='absen'
        )
        
        cursor = connection.cursor()
        cursor.execute("SELECT MAX(Error) FROM presence WHERE Id = %s", (emp_id,))
        result = cursor.fetchone()
        error_no = result[0] if result[0] else 0

        cursor.execute("UPDATE presence SET Error = %s WHERE Id = %s AND Error = %s", (error_no + 1, emp_id, error_no))
        cursor.execute("""
            INSERT INTO wrong (Error, Timestamp, Id_Camera, Id_Salah, Id_Benar, Engine, Kode_Operator)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (error_no + 1, timestamp_now, '1', emp_id, real_name, 'FaceNet', operator_code))
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
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), -1)

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
            #print('Empty face size')
            face_timer = datetime.datetime.now()
            names_probs.clear()
            return frame, None, None
        
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2) # kotak wajah
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), -1) # kotak background text
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
                    most_common_name = Attendance(most_common_nameid, 'absen', timestamp_now, max_prob)

                    # Construct folder and file names only if most_common_name is not None
                    if most_common_name:
                        folder_name = os.path.join('Capture_Result', most_common_name)
                        os.makedirs(folder_name, exist_ok=True)
                        photo_name = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_{most_common_name}_{max_prob}.jpg"
                        cv2.imwrite(os.path.join(folder_name, photo_name), frame)

                else:
                    names_probs.clear()

                #print('names_probs = ', names_probs)
                last_recognized_prob = max_prob
                last_recognized_name = most_common_name
                last_recognized_nameid = most_common_nameid

                text = "{} - {}%".format(most_common_name, max_prob)
                #print('Displaying text:', text)
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                text_x = x + (w - text_width) // 2
                text_y = y + (h + text_height) // 2

                cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, thickness)

        elif (datetime.datetime.now() - face_timer).total_seconds() > 5:
            text = str(datetime.datetime.now() - face_timer)
            #print('Displaying timer text:', text)
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = x + (w - text_width) // 2
            text_y = y + (h + text_height) // 2

            cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, thickness)

    return frame, most_common_name, prob

@app.route('/')
def index():
    return render_template('show.html')

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            now = datetime.datetime.now()
            frame, most_common_name, prob = FaceNet_recog_mp(frame, now)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/wrong_recognition', methods=['POST'])
def wrong_recognition():
    global last_recognized_nameid, last_recognized_name, last_recognized_prob
    if not last_recognized_nameid or not last_recognized_name:
        return jsonify(success=False, message="No recognized name to correct")

    data = request.get_json()
    real_name = data.get('real_name')
    kode_operator = data.get('kode_operator')
    
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            port='3308',
            database='absen'
        )

        cursor = connection.cursor()
        cursor.execute("SELECT Id FROM siswa WHERE Nama = %s", (real_name,))
        result = cursor.fetchone()
        
        if not result:
            return jsonify(success=False, message="Real name not found in profile database")
        
        real_name_id = result[0]
        timestamp_now = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        
        if log_error(last_recognized_nameid, timestamp_now, real_name_id, kode_operator):
            last_recognized_nameid, last_recognized_name, last_recognized_prob = None, None, None
            face_timer = None
            return jsonify(success=True, message="Correction submitted successfully")
        else:
            return jsonify(success=False, message="Error logging correction")
    
    except Exception as e:
        print("Error:", e)
        return jsonify(success=False, message="Database error")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)