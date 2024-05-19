import face_recognition
import pickle
import os
import cv2
import time

known_encodings = []
known_names = []
path = "/Users/khansafca/Documents/gui_fixed/database"
count = 0

directory = "/Users/khansafca/Documents/gui_fixed/dlib_recog"

# Start the timer
start_time = time.time()

# Walk through the directory tree
for dirpath, dirnames, filenames in os.walk(path):
    for filename in filenames:
        if filename == ".DS_Store":
            continue
        full_file_path = os.path.join(dirpath, filename)
        image = cv2.imread(full_file_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale

        boxes = face_recognition.face_locations(gray)

        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(image, boxes)

        # loop over the encodings
        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(os.path.basename(dirpath))  # use the directory name as the label

        count += 1

# create a data to store encodings and names
data = {"encodings": known_encodings, "names": known_names}
# Specify the full path to the file
file_path = os.path.join(directory, "encodings_dlib.pickle")

# Open the file and save the data
with open(file_path, "wb") as f:
    pickle.dump(data, f)

# Calculate the elapsed time
elapsed_time = time.time() - start_time

print(f"{count} faces trained in {elapsed_time} seconds. Thank you for waiting. The data is saved at {file_path}")

