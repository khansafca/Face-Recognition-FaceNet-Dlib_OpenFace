import argparse
import shutil
import cv2
from flask import Flask, Response, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import mediapipe as mp
from PIL import Image
import time
import os
import datetime
from os import listdir
import numpy as np
from numpy import asarray, expand_dims
import pickle
import mysql.connector
from collections import defaultdict
import requests

# Initialize
names_probs = defaultdict(list)
app = Flask(__name__)
socketio = SocketIO(app)

print("Loading face detection model")
proto_path = './model/openface/deploy.prototxt'
model_path = './model/openface/res10_300x300_ssd_iter_140000.caffemodel'
face_detector = cv2.dnn.readNetFromCaffe(prototxt=proto_path, caffeModel=model_path)

print("Loading face recognition model")
recognition_model = './model/openface/openface_nn4.small2.v1.t7'
face_recognizer = cv2.dnn.readNetFromTorch(model=recognition_model)

names_probs = defaultdict(list)
face_timer = None
last_recognized_name = None
last_recognized_prob = None
last_recognized_nameid = None
recognition_paused = False

recognizer_path = './model/openface/recognizer_openface.pickle'
with open(recognizer_path, 'rb') as f:
    recognizer = pickle.load(f)

le_path = './model/openface/le_openface.pickle'
with open(le_path, 'rb') as f:
    le = pickle.load(f)

print("Starting test video file")

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
            """, (next_no, timestamp_now, camera_id, emp_id, 'True', 'OpenFace', '0'))
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
        """, (next_no, timestamp_now, camera_id, emp_id, 'False', 'OpenFace', error_no))
        connection.commit()

        cursor.execute("""
            INSERT INTO wrong (Error, Timestamp, Id_Camera, Id_Salah, Id_Benar, Engine, Kode_Operator)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (error_no, timestamp_now, camera_id, emp_id, real_name, 'OpenFace', operator_code))
        connection.commit()

        cursor.close()
        connection.close()
        return True

    except Exception as e:
        print("Error logging to database:", e)
        return False

def openface_recog(frame, now):
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

    emp_name = None
    prob = 0
    max_prob = 0
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

    if face_detections.shape[2] == 0:
        face_timer = None
        names_probs.clear()
        return frame, None, None

    if face_timer is None:
        face_timer = datetime.datetime.now()

    text_color, font, font_scale, thickness = (255, 255, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    most_common_name, max_face_area, largest_face_bbox = 'Unknown', 0, None

    for i in range(0, face_detections.shape[2]):
        confidence = face_detections[0, 0, i, 2]
        if confidence < 0.65:
            continue

        box = face_detections[0, 0, i, 3:7] * np.array([width, height, width, height])
        (startX, startY, endX, endY) = box.astype("int")
        face_area = (endX - startX) * (endY - startY)

        if face_area > max_face_area:
            max_face_area = face_area
            largest_face_bbox = (startX, startY, endX, endY)
    
    if largest_face_bbox:
        (startX, startY, endX, endY) = largest_face_bbox
        face = frame[startY:endY, startX:endX]

        if face.size == 0:
            face_timer = datetime.datetime.now()
            names_probs.clear()
            return frame, None, None

        (fH, fW) = face.shape[:2]

        face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), True, False)
        face_recognizer.setInput(face_blob)
        vec = face_recognizer.forward()
        preds = recognizer.predict_proba(vec)[0]
        emp_id = le.classes_[np.argmax(preds)]
        prob = round(preds[np.argmax(preds)] * 100, 2)

        if prob >= 0.30:
            if emp_id not in names_probs:
                names_probs[emp_id] = []
            names_probs[emp_id].append(prob)
            emp_name = Name(emp_id, 'absen')

    # print("names_probs =", names_probs)

    most_common_name = 'Unknown'
    max_prob = 0
    most_common_emp_id = None  # Initialize most_common_emp_id

    if (datetime.datetime.now() - face_timer).total_seconds() < 5:
            text = str(datetime.datetime.now() - face_timer)
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = x + (w - text_width) // 2
            text_y = y + (h + text_height) // 2
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, thickness)

    else:
        if names_probs:
            most_common_emp_id = max(names_probs, key=lambda k: (len(names_probs[k]), max(names_probs[k])))
            max_prob = max(names_probs[most_common_emp_id])

            if most_common_emp_id != 'Unknown':  # Check if a valid name is detected
                timestamp_now = now.strftime('%Y-%m-%d_%H:%M:%S')
                emp_name = Name(most_common_emp_id, 'absen')
                most_common_name = emp_name

                # Construct folder and file names only if most_common_name is not None
                if most_common_name:
                    folder_name = os.path.join('Capture_Result', most_common_name)
                    os.makedirs(folder_name, exist_ok=True)
                    photo_name = f"{timestamp_now}_{most_common_name}_OpenFace.jpg"
                    cv2.imwrite(os.path.join(folder_name, photo_name), frame)

            else:
                names_probs.clear()
    
            last_recognized_prob = max_prob
            last_recognized_name = most_common_name
            last_recognized_nameid = most_common_emp_id if most_common_emp_id != 'Unknown' else None

            text = "{} - {}%".format(most_common_name, max_prob)
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = x + (w - text_width) // 2
            text_y = y + (h + text_height) // 2

            cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, thickness)

    return frame, last_recognized_name, max_prob

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
    global last_recognized_nameid, last_recognized_name, last_recognized_prob, operator_code

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
            notify_gui_of_error(f"Error: {real_name}-{real_name_id} was not {last_recognized_name}-{last_recognized_nameid}")
            last_recognized_nameid, last_recognized_name, last_recognized_prob = None, None, None
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
        result, max_name, max_pred = openface_recog(frame, now)

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
    parser = argparse.ArgumentParser(description='Face recognition using OpenFace')
    parser.add_argument('--camera_id', type=str, required=True, help='Camera ID')
    parser.add_argument('--url', type=str, required=True, help='URL of the video stream')
    args = parser.parse_args()
    url = args.url
    camera_id = args.camera_id

    if len(url) == 1:
        url = int(url)

    socketio.run(app, host='0.0.0.0', port=5100, debug=True)

# if __name__ == "__main__":
#     global camera_id
#     url = 0
#     camera_id = 2
#     socketio.run(app, host='0.0.0.0', port=5100, debug=True)