import cv2
import face_recognition
import os
import pickle

def encode_faces():
    known_encodings = []
    known_names = []

    for person_name in os.listdir('dataset'):
        person_dir = os.path.join('dataset', person_name)

        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(person_name)

    print(f"Number of encodings collected: {len(known_encodings)}")
    print(f"Number of names collected: {len(known_names)}")

    data = {"encodings": known_encodings, "names": known_names}
    with open('encodings.pickle', 'wb') as f:
        f.write(pickle.dumps(data))

if __name__ == "__main__":
    encode_faces()
