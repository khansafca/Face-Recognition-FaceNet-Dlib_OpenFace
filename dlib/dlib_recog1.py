import fa
import cv2
import time
from os import listdir
import numpy as np
import datetime
import os
import sys
import pickle

def dlib_recog(frame):         
    face_locations = []
    face_encodings = []
    face_names = []

    # Specify the full path to the file here
    file_path = "/Users/khansafca/Documents/gui_fixed/dlib_recog/encodings_dlib.pickle"
    data = pickle.loads(open(file_path, "rb").read())
    
    # Resize and gray scale frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    image_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    face_locations = face_recognition.face_locations(image_gray)  # get location by hog using grayscale image
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)  # get encodings by rgb images
    
    name = "Unknown"  # Initialize name here
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(data["encodings"], face_encoding, 0.6)
        accuracy = 0
        # check to see if we have found a match
        if True in matches:
            matchIdx = [i for (i, b) in enumerate(matches) if b]  # only get the True index of matches
            counts = {}
            for i in matchIdx:
                name = data["names"][i]  # get the name at index i
                counts[name] = counts.get(name, 0) + 1  # put it into a dictionary
            # get the fist max index, there would be some wrong cases
            name = max(counts, key=counts.get)  # get the name with max idx

            # Calculate accuracy based on the number of matching faces
            accuracy = counts[name] / len(matchIdx) * 100

        face_names.append((name, accuracy))  # get the face_names

    # Display the results with the condition of 1 face
    for (top, right, bottom, left), (name, accuracy) in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        text = f'{name} - {accuracy:.2f}%'
        cv2.putText(frame, text, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    return frame, name, accuracy


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    time.sleep(3)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        result, name, accuracy = dlib_recog(frame)
        cv2.imshow('Dlib', result)

        if name is not None and accuracy > 0.88:
            now = datetime.datetime.now()
            dt_string = now.strftime("%d-%m-%Y:%H.%M.%S")
            folder_path = f'Capture/{name}'
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
