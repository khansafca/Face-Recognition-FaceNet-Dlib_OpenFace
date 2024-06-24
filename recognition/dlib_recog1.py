import argparse
import face_recognition
import cv2
import time
import shutil
from os import listdir
import datetime
from flask import Flask, Response, render_template, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import numpy as np
from collections import defaultdict
import datetime
import mysql.connector
import re
import os
import sys
import pickle
import dlib

# Initialize
names_probs = defaultdict(list)
app = Flask(__name__)
socketio = SocketIO(app)

# Initialize dlib's face detector and the face recognition model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/Users/khansafca/Documents/gui_fixed/dlib_recog/model/shape_predictor_68_face_landmarks.dat')
face_rec_model = dlib.face_recognition_model_v1('/Users/khansafca/Documents/gui_fixed/dlib_recog/model/dlib_face_recognition_resnet_model_v1.dat')
data = {'encodings': [], 'names': []}  # This should be filled with actual face encodings and corresponding names

# Assuming these are initialized similarly to FaceNet_recog_mp
names_probs = defaultdict(list)
face_timer = None
last_recognized_name = None
last_recognized_prob = None
last_recognized_nameid = None
capture_active = True
recognition_paused = False

# Load known faces and embeddings
with open("/Users/khansafca/Documents/gui_fixed/dlib_recog/encodings_dlib.pickle", "rb") as f:
    data = pickle.load(f)

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
            """, (next_no, timestamp_now, '1', emp_id, 'True', 'Dlib', '0'))
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
        """, (next_no, timestamp_now, '1', emp_id, 'False', 'Dlib', error_no))
        connection.commit()

        cursor.execute("""
            INSERT INTO wrong (Error, Timestamp, Id_Camera, Id_Salah, Id_Benar, Engine, Kode_Operator)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (error_no, timestamp_now, '1', emp_id, real_name, 'Dib', operator_code))
        connection.commit()

        cursor.close()
        connection.close()
        return True

    except Exception as e:
        print("Error logging to database:", e)
        return False


def dlib_recog(frame, now):
    global face_timer, last_recognized_name, last_recognized_prob, last_recognized_nameid, capture_active, recognition_paused
    if recognition_paused:
        return frame, None, None

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = detector(rgb_small_frame, 1)

    if not face_locations:
        face_timer = None
        names_probs.clear()
        return frame, None, None

    if face_timer is None:
        face_timer = datetime.datetime.now()

    name = "Unknown"
    accuracy = 0

    for face in face_locations:
        shape = predictor(rgb_small_frame, face)
        face_encoding = np.array(face_rec_model.compute_face_descriptor(rgb_small_frame, shape, 1))

        matches = face_recognition.compare_faces(data["encodings"], face_encoding, 0.6)
        face_distances = face_recognition.face_distance(data["encodings"], face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = data["names"][best_match_index]
            accuracy = 1 - face_distances[best_match_index]

        names_probs[name].append(accuracy)

    if (datetime.datetime.now() - face_timer).total_seconds() > 5.5:
        if names_probs:
            most_common_name = max(names_probs, key=lambda k: (len(names_probs[k]), max(names_probs[k])))
            max_prob = max(names_probs[most_common_name])

            if most_common_name != 'Unknown':  # Check if a valid name is detected
                timestamp_now = now.strftime('%Y-%m-%d_%H:%M:%S')
                emp_name = Name(most_common_name, 'absen')
                if emp_name:
                    folder_name = os.path.join('Capture_Result', emp_name)
                    os.makedirs(folder_name, exist_ok=True)
                    photo_name = f"{timestamp_now}_{emp_name}_{max_prob}.jpg"
                    cv2.imwrite(os.path.join(folder_name, photo_name), frame)

            else:
                names_probs.clear()

            last_recognized_prob = max_prob
            last_recognized_name = most_common_name
            last_recognized_nameid = most_common_name

            text = "{} - {}%".format(emp_name, max_prob * 100)
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
            text_x = int((frame.shape[1] - text_width) / 2)
            text_y = frame.shape[0] - text_height - 10

            # Draw face bounding box
            for face in face_locations:
                x, y, w, h = (face.left() * 4, face.top() * 4, face.width() * 4, face.height() * 4)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw background rectangle for text
            cv2.rectangle(frame, (text_x, text_y - text_height - 10), (text_x + text_width, text_y + baseline - 10), (0, 0, 0), -1)

            # Draw text
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

    return frame, last_recognized_name, accuracy


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
            result, max_name, max_pred = dlib_recog(frame, now)

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
    socketio.run(app, host='0.0.0.0', port=5400, debug=True)
