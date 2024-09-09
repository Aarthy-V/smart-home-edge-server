import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

def train_model():
    with open('encodings.pickle', 'rb') as f:
        data = pickle.load(f)

    if not data["encodings"]:
        print("No encodings found. Check your dataset.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        data["encodings"], data["names"], test_size=0.2, random_state=42
    )

    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)

    with open('face_recognition_model.pickle', 'wb') as f:
        f.write(pickle.dumps(clf))

    print(f"Model trained and saved to 'face_recognition_model.pickle'.")

if __name__ == "__main__":
    train_model()
