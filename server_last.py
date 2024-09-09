import cv2
import socket
import struct
import pickle
import numpy as np
import face_recognition
import requests
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template_string
import threading
import queue


CONFIDENCE_THRESHOLD = 0.95

ACTION_WAIT_TIME_SECONDS = 10
UNKNOWN_ACTION_WAIT_TIME_SECONDS = 10


server_ip = '192.168.8.101'  
server_port = 9997

esp32_ip = '192.168.8.100'  
esp32_port = 80  

app = Flask(__name__)

#queue handling
data_queue = queue.Queue()
command_queue = queue.Queue()

received_data_list = []

# HTML for local server
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Face Recognition Data</title>
</head>
<body>
    <h1>Face Recognition Data</h1>
    {% if received_data_list %}
        <ul>
        {% for data in received_data_list %}
            <li><strong>Name:</strong> {{ data['name'] }} | <strong>Authenticated:</strong> {{ data['authenticated'] }} | <strong>Date:</strong> {{ data['date'] }} | <strong>Time:</strong> {{ data['time'] }}</li>
        {% endfor %}
        </ul>
    {% else %}
        <p>No data received yet.</p>
    {% endif %}
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(html_template, received_data_list=received_data_list)

@app.route('/receive_data', methods=['POST'])
def receive_data():
    data = request.get_json()
    received_data_list.append(data)
    print(f"Received data: {data}")
    return jsonify({"status": "success"}), 200

def load_model():
    with open('face_recognition_model.pickle', 'rb') as f:
        model = pickle.load(f)
    return model

def recognize_faces(frame, model):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    results = []

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_face)

        if encodings:
            encoding = encodings[0]
            predictions = model.predict_proba([encoding])
            predicted_class = model.predict([encoding])[0]
            confidence = np.max(predictions)

            if confidence >= CONFIDENCE_THRESHOLD:
                name = predicted_class
                authenticated = True
            else:
                name = "Unknown"
                authenticated = False

            # label persons
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"{name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            now = datetime.now()
            date_str = now.strftime('%Y-%m-%d')
            time_str = now.strftime('%H.%M.%S')

            results.append({
                'name': name,
                'authenticated': authenticated,
                'date': date_str,
                'time': time_str
            })

    return frame, results

def send_data_to_server():
    while True:
        data = data_queue.get()
        if data is None:
            break
        try:
            response = requests.post('http://localhost:5000/receive_data', json=data)
            print(f"Sent to server: {data}, Response: {response.status_code}")
        except requests.RequestException as e:
            print(f"Error sending data to server: {e}")
        finally:
            data_queue.task_done()

def send_command_to_esp32():
    while True:
        command = command_queue.get()
        if command is None:
            break
        try:
            if command['type'] == 'led_on':
                pin = command['pin']
                url = f"http://{esp32_ip}/pin?number={pin}&state=on"
            elif command['type'] == 'led_off':
                pin = command['pin']
                url = f"http://{esp32_ip}/pin?number={pin}&state=off"
            else:
                continue

            response = requests.get(url)
            print(f"Sent {command['type']} command to ESP32, Response: {response.status_code}")
        except requests.RequestException as e:
            print(f"Error sending command to ESP32: {e}")
        finally:
            command_queue.task_done()

def run_flask_app():
    app.run(host='localhost', port=5000)

# run in seperate threads
flask_thread = threading.Thread(target=run_flask_app)
flask_thread.start()

data_thread = threading.Thread(target=send_data_to_server)
data_thread.start()

command_thread = threading.Thread(target=send_command_to_esp32)
command_thread.start()

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((server_ip, server_port))
server_socket.listen(5)
print("Server is listening...")

client_socket, addr = server_socket.accept()
print(f"Connection from {addr}")

model = load_model()

# Delay time setting
last_known_action_time = datetime.now() - timedelta(seconds=ACTION_WAIT_TIME_SECONDS)
last_unknown_action_time = datetime.now() - timedelta(seconds=UNKNOWN_ACTION_WAIT_TIME_SECONDS)


# Receive data from the client
try:
    while True:
        
        data = b''
        while len(data) < 4:
            data += client_socket.recv(4 - len(data))
        packed_msg_size = data
        msg_size = struct.unpack('!I', packed_msg_size)[0]

        data = b''
        while len(data) < msg_size:
            data += client_socket.recv(msg_size - len(data))
        frame_data = data

        np_arr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        current_time = datetime.now()

        frame, results = recognize_faces(frame, model)

        for result in results:
            if result['authenticated']:
                if (current_time - last_known_action_time).total_seconds() >= ACTION_WAIT_TIME_SECONDS:
                    command_queue.put({'type': 'led_on', 'pin': 19})  
                    data_queue.put(result)  
                    last_known_action_time = current_time

            else:  
                if (current_time - last_unknown_action_time).total_seconds() >= UNKNOWN_ACTION_WAIT_TIME_SECONDS:
                    data_queue.put(result)  
                    command_queue.put({'type': 'led_on', 'pin': 2})  
                    last_unknown_action_time = current_time

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    client_socket.close()
    server_socket.close()
    cv2.destroyAllWindows()

    data_queue.put(None)
    command_queue.put(None)

    data_thread.join()
    command_thread.join()
