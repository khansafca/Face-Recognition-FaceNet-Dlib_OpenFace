import argparse
import face_recognition
import pickle
import os
import cv2
import dlib
import numpy as np

def extract_embeddings(image_path):
    # Function to extract face embeddings from a single image
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./model/dlib/shape_predictor_68_face_landmarks.dat')
    face_rec_model = dlib.face_recognition_model_v1('./model/dlib/dlib_face_recognition_resnet_model_v1.dat')
                                                
    image = cv2.imread(image_path)
    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector(rgb_frame)
    
    # Assume there is only one face in the image for simplicity
    if len(faces) == 1:
        shape = predictor(rgb_frame, faces[0])
        face_encoding = np.array(face_rec_model.compute_face_descriptor(rgb_frame, shape))
        return [face_encoding]  # Return as a list for consistency with multiple encodings

    return []

def main(data_path):
    known_encodings = []
    known_names = []
    count = 0

    # Loop over the image paths
    for folder in os.listdir(data_path):
        if folder == ".DS_Store" or folder == "dataset/.jpg":
            continue
        image_folder = os.path.join(data_path, folder)
        count += 1
        print(image_folder)
        for file in os.listdir(image_folder):
            
            full_file_path = os.path.join(image_folder, file)
            
            encodings = extract_embeddings(full_file_path)
            if encodings:
                known_encodings.extend(encodings)
                known_names.extend([folder] * len(encodings))

    # Create a data dictionary to store encodings and names
    data = {"encodings": known_encodings, "names": known_names}

    # Save the encodings and names into a pickle file
    with open("./model/dlib/encoding_dlib.pickle", "wb") as f:
        pickle.dump(data, f)

    print(f"{count} faces trained. Finish Training Dlib")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data directory")
    args = parser.parse_args()

    main(args.data_path)
