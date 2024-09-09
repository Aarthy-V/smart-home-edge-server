import cv2
import os

def capture_images(person_name):
    save_dir = f'dataset/{person_name}'
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        # Save every 10th frame to reduce redundancy
        if count % 10 == 0:
            img_path = os.path.join(save_dir, f'{person_name}_{count}.jpg')
            cv2.imwrite(img_path, frame)
            print(f"Captured image {img_path}")

        count += 1

        if count >= 1000:  
            break

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    persons = ['Aarthy','Senthooran']  

    for person_name in persons:
        capture_images(person_name)
