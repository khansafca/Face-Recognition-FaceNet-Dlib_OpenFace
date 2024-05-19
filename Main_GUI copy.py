import tkinter as tk
import subprocess
from PIL import Image, ImageTk
import cv2
import os
import time
import mediapipe as mp

# Create a directory to save the images if it doesn't exist
if not os.path.exists('database'):
    os.makedirs('database')

class CameraGUI:
    def __init__(self, master):
        self.master = master

        # Load the background image, resize it, and set it as the background
        img = Image.open("5.jpeg")
        self.tk_img = ImageTk.PhotoImage(img)
        label = tk.Label(master, image=self.tk_img)
        label.pack()

        self.register_button = tk.Button(master, text="Register", command=self.register)
        self.register_button.pack()
        self.register_button.place(x=115, y=350)

        self.camera_button = tk.Button(master, text="Open Camera", command=self.open_camera)
        self.camera_button.pack()
        self.camera_button.place(x=100, y=495)

        # Counter for the number of photos taken
        self.photo_count = 0

        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    def register(self):
        self.name_label = tk.Label(self.master, text="Enter NIM:")
        self.name_label.pack()
        self.name_label.place(x=123, y=395)
        
        self.name_entry = tk.Entry(self.master)
        self.name_entry.pack()
        self.name_entry.place(x=55, y=420)

        self.name_button = tk.Button(self.master, text="Set NIM", command=self.set_name)
        self.name_button.pack()
        self.name_button.place(x=115, y=450)

    def set_name(self):
        self.name = self.name_entry.get()
        print("NIM set to:", self.name)

    def open_camera(self):
            self.cap = cv2.VideoCapture(0)
            cv2.namedWindow("Camera Feed")

            # Create a directory with the given name if it doesn't exist
            folder_name = f'database/{self.name}'
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            # Set the start time
            start_time = time.time()

            # Set the time interval (0.7 seconds)
            interval = 0.7

            # Add a flag for capturing
            capturing = False

            while True:
                ret, frame = self.cap.read()

                # Display the photo count on the frame
                cv2.putText(frame, f'Count: {self.photo_count}', (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                cv2.imshow("Camera Feed", frame)

                key = cv2.waitKey(1) & 0xFF

                # Press 'space' to start/stop capturing
                if key == ord(' '):
                    if capturing:
                        self.photo_count = 0
                        break
                    else:
                        capturing = True

                # Get the current time
                current_time = time.time()

                # Check if the time interval has passed and if capturing is True
                if capturing and current_time - start_time >= interval:
                    # Increase the photo count
                    self.photo_count += 1
                    date_time = time.strftime("%Y%m%d%H%M%S")

                    # Save the frame as an image file
                    photo_name = f'{self.name}_{date_time}.jpg'
                    cv2.imwrite(os.path.join(folder_name, photo_name), frame)
                    
                    print(f'Photo saved as {photo_name}')

                    # Update the start time
                    start_time = current_time

                # Press 'q' to quit
                if key == ord('q'):
                    break

            self.cap.release()
            cv2.destroyAllWindows()


def train_facenet():
    subprocess.run(['python3', '/Users/khansafca/Documents/gui_fixed/facenet_recog/training_facenet_mp.py'])

def train_openface():
    subprocess.run(['python3', '/Users/khansafca/Documents/gui_fixed/openface/training.py'])

def train_dlib():
    subprocess.run(['python3', '/Users/khansafca/Documents/gui_fixed/dlib_recog/preprocess.py'])

def facenet():
    subprocess.run(['python3', '/Users/khansafca/Documents/gui_fixed/facenet_recog/facenet_mediapipe_mp.py'])

def openface():
    subprocess.run(['python3', '/Users/khansafca/Documents/gui_fixed/openface/openface_recog1.py'])

def dlib():
    subprocess.run(['python3', '//Users/khansafca/Documents/gui_fixed/dlib_recog/dlib_recog1.py'])

def create_train_buttons():
    button_facenet = tk.Button(root, text="Facenet Train", command=train_facenet)
    button_facenet.pack()
    button_facenet.place(x=340, y=415)

    button_openface = tk.Button(root, text="Openface Train", command=train_openface)
    button_openface.pack()
    button_openface.place(x=335, y=455)

    button_dlib = tk.Button(root, text="Dlib Train", command=train_dlib)
    button_dlib.pack()
    button_dlib.place(x=350, y=495)

def create_face_recognition():
    button_facenet = tk.Button(root, text="Facenet", command=facenet)
    button_facenet.pack()
    button_facenet.place(x=605, y=415)

    button_openface = tk.Button(root, text="Openface", command=openface)
    button_openface.pack()
    button_openface.place(x=600, y=455)

    button_dlib = tk.Button(root, text="Dlib", command=dlib)
    button_dlib.pack()
    button_dlib.place(x=620, y=495)

root = tk.Tk()

# Set the window size to 800x500
root.geometry("800x600")

app = CameraGUI(root)

button_train = tk.Button(root, text="Train The Data", command=create_train_buttons)
button_train.pack()
button_train.place(x=335, y=350)

button_train = tk.Button(root, text="Start Recognize", command=create_face_recognition)
button_train.pack()
button_train.place(x=580, y=350)

root.mainloop()
