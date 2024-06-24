import argparse
from flask import Flask, Response, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import shutil
import numpy as np
import pickle
import os
import datetime
import mysql.connector
import re
from collections import defaultdict
import cv2
import time
import sys
import imutils

# Initialize
names_probs = defaultdict(list)
app = Flask(__name__)
socketio = SocketIO(app)

print("Loading face detection model")
proto_path = '/Users/khansafca/Documents/gui_fixed/model/deploy.prototxt'
model_path = '/Users/khansafca/Documents/gui_fixed/model/res10_300x300_ssd_iter_140000.caffemodel'
face_detector = cv2.dnn.readNetFromCaffe(prototxt=proto_path, caffeModel=model_path)

print("Loading face recognition model")
recognition_model = '/Users/khansafca/Documents/gui_fixed/model/openface_nn4.small2.v1.t7'
face_recognizer = cv2.dnn.readNetFromTorch(model=recognition_model)

face_timer = None
last_recognized_name = None
last_recognized_prob = None
last_recognized_nameid = None
capture_active = True
recognition_paused = False

recognizer_path = '/Users/khansafca/Documents/gui_fixed/openface/recognizer_openface.pickle'
with open(recognizer_path, 'rb') as f:
    recognizer = pickle.load(f)

le_path = '/Users/khansafca/Documents/gui_fixed/openface/le_openface.pickle'
with open(le_path, 'rb') as f:
    le = pickle.load(f)

print("Starting test video file")

# List to store wrong recognition messages
#wrong_recognition_messages = []

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

def Name(emp_id, database_name):
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
            """, (next_no, timestamp_now, '1', emp_id, 'True', 'OpenFace', '0'))
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
        error_no = result[0] + 1 if result[0] else 1

        next_no = get_next_no()
        cursor.execute("""
            INSERT INTO presence (No, Timestamp, Id_Camera, Id, Flag, Engine, Error)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (next_no, timestamp_now, '1', emp_id, 'False', 'OpenFace', error_no))
        connection.commit()

        cursor.execute("""
            INSERT INTO wrong (Error, Timestamp, Id_Camera, Id_Salah, Id_Benar, Engine, Kode_Operator)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (error_no, timestamp_now, '1', emp_id, real_name, 'OpenFace', operator_code))
        connection.commit()

        cursor.close()
        connection.close()
        return True

    except Exception as e:
        print("Error logging to database:", e)
        return False


def openface_recog(frame, now):
    global face_timer, last_recognized_name, last_recognized_prob, last_recognized_nameid, capture_active, recognition_paused

    if recognition_paused:
        return frame, None, None

    emp_name = None
    proba = 0.0
    (h, w) = frame.shape[:2]

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization
    equalized = cv2.equalizeHist(gray)

    # Convert back to BGR for dnn.blobFromImage
    equalized_bgr = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

    image_blob = cv2.dnn.blobFromImage(cv2.resize(equalized_bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)

    face_detector.setInput(image_blob)
    face_detections = face_detector.forward()

    if face_timer is None:
        face_timer = datetime.datetime.now()

    names_probs = {}

    for i in range(0, face_detections.shape[2]):
        confidence = face_detections[0, 0, i, 2]

        if confidence >= 0.15:
            box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue

            (fH, fW) = face.shape[:2]

            face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), True, False)

            face_recognizer.setInput(face_blob)
            vec = face_recognizer.forward()

            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            emp_id = le.classes_[j]

            if proba >= 0.30:
                if emp_id not in names_probs:
                    names_probs[emp_id] = []
                names_probs[emp_id].append(proba)
                emp_name = Name(emp_id, 'absen')

    most_common_name = 'Unknown'
    max_prob = 0
    most_common_emp_id = None  # Initialize most_common_emp_id

    if (datetime.datetime.now() - face_timer).total_seconds() > 5.5:
        if names_probs:
            most_common_emp_id = max(names_probs, key=lambda k: (len(names_probs[k]), max(names_probs[k])))
            max_prob = max(names_probs[most_common_emp_id])

            if most_common_emp_id != 'Unknown':
                emp_name = Name(most_common_emp_id, 'absen')
                most_common_name = emp_name

    last_recognized_prob = max_prob
    last_recognized_name = most_common_name
    last_recognized_nameid = most_common_emp_id if most_common_emp_id != 'Unknown' else None

    if emp_name:
        text = "{} - {}%".format(emp_name, max_prob * 100)
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1.0
        text_color = (255, 255, 255)
        thickness = 2
        
        # Calculate text width, height, and baseline
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Calculate text position
        text_x = int((frame.shape[1] - text_width) / 2)
        text_y = frame.shape[0] - text_height - 10
        
        # Draw rectangle and put text
        cv2.rectangle(frame, (text_x, text_y - text_height - 10), (text_x + text_width, text_y + baseline - 10), (0, 0, 0), -1)
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, thickness)

    return frame, last_recognized_name, max_prob

@app.route('/')
def index():
    return render_template('show.html')

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

        # Ensure the ID values are integers and within the acceptable range
        try:
            real_name_id = int(real_name_id)
            last_recognized_nameid = int(last_recognized_nameid)
        except ValueError:
            return jsonify(success=False, message="Invalid ID format")

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


@app.route('/pause_recognition', methods=['POST'])
def pause_recognition():
    global recognition_paused
    recognition_paused = True
    return jsonify(success=True, message="Recognition paused")

@app.route('/resume_recognition', methods=['POST'])
def resume_recognition():
    global recognition_paused
    recognition_paused = False
    return jsonify(success=True, message="Recognition resumed")

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

        now = datetime.datetime.now()

        if capture_active:  # Only process the frame if capture is active
            result, max_name, max_pred = openface_recog(frame, now)

            if max_name and max_pred and max_pred > 80:
                timestamp_now = now.strftime('%Y-%m-%d_%H-%M-%S')
                folder_path = 'Capture_Data'
                os.makedirs(folder_path, exist_ok=True)

                folder_name = os.path.join(folder_path, max_name)
                photo_name = f"{timestamp_now}_{max_name}_{max_pred}.jpg"
                cv2.imwrite(os.path.join(folder_name, photo_name), frame)
        else:
            result = frame  # Display the same frame when capture is not active

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


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    url = 0
    socketio.run(app, host='0.0.0.0', port=5300, debug=True)
