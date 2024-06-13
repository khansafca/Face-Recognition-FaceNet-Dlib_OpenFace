import numpy as np
import pickle
import os
import cv2
import time
import datetime
import imutils

curr_path = os.path.dirname(os.path.abspath(__file__))

print("Loading face detection model")
proto_path = os.path.join(curr_path,  'model', 'deploy.prototxt')
model_path = os.path.join(curr_path, 'model', 'res10_300x300_ssd_iter_140000.caffemodel')
face_detector = cv2.dnn.readNetFromCaffe(prototxt=proto_path, caffeModel=model_path)

print("Loading face recognition model")
recognition_model = os.path.join(curr_path, 'model', 'openface_nn4.small2.v1.t7')
face_recognizer = cv2.dnn.readNetFromTorch(model=recognition_model)

recognizer = pickle.loads(open(os.path.join(curr_path, 'recognizer_openface.pickle'), "rb").read())
le = pickle.loads(open(os.path.join(curr_path, 'le_openface.pickle'), "rb").read())

print("Starting test video file")

def openface_recog(frame):
    name = "Unknown"
    while True:
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

            if confidence >= 0.95:
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

                text = "{}: {:.2f}".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        return frame, name

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        result, text = openface_recog(frame)
        cv2.imshow('OpenFace', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # change to waitKey(1)
            break


    cap.release()
    cv2.destroyAllWindows()