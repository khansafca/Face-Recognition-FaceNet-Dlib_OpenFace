import argparse
import face_recognition
import cv2
import time
from os import listdir
import datetime
from flask import Flask, Response, render_template, render_template, request, jsonify
import numpy as np
import datetime
import mysql.connector
import re
import os
import sys
import pickle
import dlib

app = Flask(__name__)

# Initialize dlib's face detector and the face recognition model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/khansafca/Documents/gui_fixed/dlib_recog/model/shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("/Users/khansafca/Documents/gui_fixed/dlib_recog/model/dlib_face_recognition_resnet_model_v1.dat")

# Load known faces and embeddings
with open("/Users/khansafca/Documents/gui_fixed/dlib_recog/encodings_dlib.pickle", "rb") as f:
    data = pickle.load(f)

# List to store wrong recognition messages
wrong_recognition_messages = []

def Attendance(emp_id, database_name, timestamp_now, accuracy):
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            port='3308',
            database=database_name
        )

        cursor = connection.cursor()
        cursor.execute("SELECT name, Dlib FROM users WHERE id = %s", (emp_id,))
        result = cursor.fetchone()

        if result:
            emp_name, timestamp = result
            if accuracy > 0.6:
                new_timestamp_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                cursor.execute("UPDATE users SET Dlib = %s WHERE id = %s", (new_timestamp_now, emp_id))
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

def dlib_recog(frame):
    face_locations = []
    face_encodings = []
    face_names = []

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = detector(rgb_small_frame, 1)

    name = "Unknown"
    accuracy = 0
    emp_name = None

    for face in face_locations:
        shape = predictor(rgb_small_frame, face)
        face_encoding = np.array(face_rec_model.compute_face_descriptor(rgb_small_frame, shape, 1))

        matches = face_recognition.compare_faces(data["encodings"], face_encoding, 0.6)
        face_distances = face_recognition.face_distance(data["encodings"], face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = data["names"][best_match_index]
            accuracy = 1 - face_distances[best_match_index]
        
        face_names.append((name, accuracy))

    for (i, rect), (name, accuracy) in zip(enumerate(face_locations), face_names):
        top, right, bottom, left = rect.top(), rect.right(), rect.bottom(), rect.left()
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX

        emp_name = Attendance(name, 'absen', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), accuracy)
        if emp_name:
            text = f'{emp_name} - {accuracy * 100:.2f}%'
            cv2.putText(frame, text, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    return frame, emp_name, accuracy

def gen_frames():
    cap = cv2.VideoCapture(url)
    time.sleep(3)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result, emp_name, accuracy = dlib_recog(frame)

        ret, buffer = cv2.imencode('.jpg', result)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        if emp_name is not None and accuracy > 0.7:
            now = datetime.datetime.now()
            dt_string = now.strftime("%Y%m%d%H%M%S")
            folder_path = 'Capture'
            folder_name = os.path.join(folder_path, emp_name)
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            try:
                # Save the 'result' frame (which is a numpy array) to an image file
                if frame is not None:
                    photo_name = f'{emp_name}_{dt_string}.jpg'
                    cv2.imwrite(os.path.join(folder_name, photo_name), frame)
                    print(f'Photo saved as {photo_name}')
                else:
                    print(f'Error: Result frame is None or empty, not saving the file.')
            except Exception as e:
                print(f'Error saving file: {e}')

            time.sleep(1)

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
def wrong_recognition():
    text = request.form.get('text')
    mistaken_name = request.form.get('mistaken_name')
    timestamp_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    message = f"Wrong recognition for {text} on {timestamp_now} mistaken by {mistaken_name}"
    
    # Append the new message to the list
    wrong_recognition_messages.append(message)
    
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
    # url = 'http://192.168.19.31:5000/video'
    app.run(host='0.0.0.0', port=5400, debug=True)
