import argparse
from flask import Flask, Response, render_template, request, jsonify
import numpy as np
import pickle
import os
import datetime
import mysql.connector
import re
import cv2
import time
import sys
import imutils

app = Flask(__name__)

print("Loading face detection model")
proto_path = './model/deploy.prototxt'
model_path = './model/res10_300x300_ssd_iter_140000.caffemodel'
face_detector = cv2.dnn.readNetFromCaffe(prototxt=proto_path, caffeModel=model_path)

print("Loading face recognition model")
recognition_model = './model/openface_nn4.small2.v1.t7'
face_recognizer = cv2.dnn.readNetFromTorch(model=recognition_model)

recognizer_path = '/Users/khansafca/Documents/gui_fixed/openface/recognizer_openface.pickle'
with open(recognizer_path, 'rb') as f:
    recognizer = pickle.load(f)

le_path = '/Users/khansafca/Documents/gui_fixed/openface/le_openface.pickle'
with open(le_path, 'rb') as f:
    le = pickle.load(f)

print("Starting test video file")

# List to store wrong recognition messages
wrong_recognition_messages = []

def Attendance(emp_id, database_name, timestamp_now, proba):
    try:
        # Ensure emp_id is an integer (or convert it if necessary)
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
        cursor.execute("SELECT name, OpenFace FROM users WHERE id = %s", (emp_id,))
        result = cursor.fetchone()

        if result:
            emp_name, timestamp = result
            if proba > 0.7:
                new_timestamp_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                cursor.execute("UPDATE users SET OpenFace = %s WHERE id = %s", (new_timestamp_now, emp_id))
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


def openface_recog(frame):
    emp_name = None  # Initialize name
    proba = 0.0  # Initialize proba
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

    for i in range(0, face_detections.shape[2]):
        confidence = face_detections[0, 0, i, 2]

        if confidence >= 0.65:
            box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue

            (fH, fW) = face.shape[:2]

            face_blob = cv2.dnn.blobFromImage(face, 1.0/255, (96, 96), (0, 0, 0), True, False)

            face_recognizer.setInput(face_blob)
            vec = face_recognizer.forward()

            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]
            emp_name= name
            emp_name = Attendance(name, 'absen', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), proba)
            if emp_name:
                text = "{}: {:.2f}".format(emp_name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    return frame, emp_name, proba

def gen_frames():
    cap = cv2.VideoCapture(url)
    time.sleep(3)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result, emp_name, proba = openface_recog(frame)
        cv2.imshow('OpenFace', result)
        ret, buffer = cv2.imencode('.jpg', result)
        if ret:
            frame = buffer.tobytes()  # Convert numpy array to string
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        
        if emp_name is not None and proba > 0:
            now = datetime.datetime.now()
            dt_string = now.strftime("%Y-%m-%d:%H.%M.%S")
            folder_path = f'Capture/{emp_name}'
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


@app.route('/')
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
    app.run(host='0.0.0.0', port=5300, debug=True)