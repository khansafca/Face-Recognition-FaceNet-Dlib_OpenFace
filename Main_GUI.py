import tkinter as tk
import subprocess
from PIL import Image, ImageTk
import cv2
import os
import time
import mediapipe as mp
import signal

# Constants
DATABASE_DIR = 'database'
BACKGROUND_IMAGE_PATH = 'main gui.jpg'
CAMERA_URL = 'http://172.20.10.14:5000/video_feed'
WINDOW_SIZE = "1000x700"

# Create a directory to save the images if it doesn't exist
os.makedirs(DATABASE_DIR, exist_ok=True)

recognize_process = None  # Global variable to store the face recognition process

class CameraGUI:
    def __init__(self, master):
        self.master = master
        self.setup_gui()
        self.photo_count = 0
        self.name = ''
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    def setup_gui(self):
        self.master.configure(bg="#242424")
        self.master.geometry(WINDOW_SIZE)

        self.main_frame = tk.Frame(self.master, bg="#242424")
        self.main_frame.pack(expand=True, fill="both")

        try:
            img = Image.open(BACKGROUND_IMAGE_PATH)
            self.tk_img = ImageTk.PhotoImage(img)
            label = tk.Label(self.main_frame, image=self.tk_img, bg="#242424")
            label.grid(row=0, column=0, columnspan=2, sticky="nsew")
        except Exception as e:
            print(f"Error loading background image: {e}")

        self.register_button = tk.Button(self.master, text="Register", command=self.register, font=("Arial", 10), height=1, width=8)
        self.register_button.place(x=180, y=430)

        self.camera_button = tk.Button(self.master, text="Open Camera", command=lambda: self.open_camera(CAMERA_URL), font=("Arial", 10), height=1, width=10)
        self.camera_button.place(x=170, y=575)

    def register(self):
        self.name_label = tk.Label(self.master, text="Enter NIM:", font=("Arial", 10), height=1, width=8)
        self.name_label.place(x=182, y=470)
        
        self.name_entry = tk.Entry(self.master, font=("Arial", 10, 'bold'), justify="center")
        self.name_entry.place(x=142, y=500)

        self.name_button = tk.Button(self.master, text="Set NIM", command=self.set_name, font=("Arial", 10), height=1, width=8)
        self.name_button.place(x=180, y=530)

    def set_name(self):
        self.name = self.name_entry.get()
        print(f"NIM set to: {self.name}")

    def open_camera(self, url):

        if not self.name:
            print("NIM not set. Please register first.")
            return

        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Failed to open video stream.")
            return

        cv2.namedWindow("Camera Feed")

        folder_name = os.path.join(DATABASE_DIR, self.name)
        os.makedirs(folder_name, exist_ok=True)

        start_time = time.time()
        interval = 0.6
        capturing = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.putText(frame, f'Count: {self.photo_count}', (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Camera Feed", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                capturing = not capturing
                if not capturing:
                    print("Paused capturing.")
                else:
                    print("Resumed capturing.")
                start_time = time.time()  # Reset start_time to avoid immediate capture on resume

            if capturing and (time.time() - start_time) >= interval:
                self.photo_count += 1
                date_time = time.strftime("%Y%m%d%H%M%S")
                photo_name = f'{self.name}_{date_time}.jpg'
                cv2.imwrite(os.path.join(folder_name, photo_name), frame)
                print(f'Photo saved as {photo_name}')
                start_time = time.time()

            if key == ord('q'):
                print("Exiting camera feed.")
                self.photo_count = 0
                break

        cap.release()
        cv2.destroyAllWindows()

def train_model(script_name):
    subprocess.run(['python3', f'training/{script_name}'])

def recognize_model(script_name, url):
    global recognize_process
    recognize_process = subprocess.Popen(['python3', f'recognition/{script_name}', '--url', url])

def stop_face_recognition():
    global recognize_process
    if recognize_process:
        if os.name == 'nt':  # Check if running on Windows
            recognize_process.send_signal(signal.CTRL_C_EVENT)
        else:  # Unix-like systems
            recognize_process.send_signal(signal.SIGINT)
        recognize_process = None

def create_train_buttons(root):
    buttons = [
        ("Facenet Train", lambda: train_model('training_facenet.py')),
        ("Openface Train", lambda: train_model('training_openface.py')),
        ("Dlib Train", lambda: train_model('training_dlib.py')),
    ]
    y_position = 480
    button_width = 14
    for text, command in buttons:
        button = tk.Button(root, text=text, command=command, font=("Arial", 10, "bold"), width=button_width)
        button.place(x=440, y=y_position)
        y_position += 50

def create_face_recognition(root, url):
    buttons = [
        ("Facenet", lambda: recognize_model('facenet_mediapipe_mp.py', url)),
        ("Openface", lambda: recognize_model('openface_recog1.py', url)),
        ("Dlib", lambda: recognize_model('dlib_recog1.py', url)),
    ]
    y_position = 480
    button_width = 12
    for text, command in buttons:
        button = tk.Button(root, text=text, command=command, font=("Arial", 10, "bold"), width=button_width)
        button.place(x=737, y=y_position)
        y_position += 50

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x600")
    app = CameraGUI(root)

    tk.Button(root, text="Train The Data", command=lambda: create_train_buttons(root), font=("Arial", 10), height=1, width=12).place(x=450, y=430)
    tk.Button(root, text="Start Recognize", command=lambda: create_face_recognition(root, CAMERA_URL), font=("Arial", 10), height=1, width=14).place(x=730, y=430)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        # Handle Ctrl+C or window close event
        stop_face_recognition()  # Stop the face recognition process if running
