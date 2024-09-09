import cv2
import face_recognition
import pickle
import numpy as np

# Define the threshold for confidence
CONFIDENCE_THRESHOLD = 0.95

def load_model():
    with open('face_recognition_model.pickle', 'rb') as f:
        model = pickle.load(f)
    return model

def recognize_faces():
    model = load_model()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_face)

            if encodings:
                encoding = encodings[0]
                # Predict the class and probabilities
                predictions = model.predict_proba([encoding])
                predicted_class = model.predict([encoding])[0]
                confidence = np.max(predictions)

                # Check if confidence is above the threshold
                if confidence >= CONFIDENCE_THRESHOLD:
                    name = predicted_class
                else:
                    name = "Unknown"

                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()
