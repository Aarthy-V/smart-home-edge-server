import cv2
import socket
import struct
import pickle

# Server configuration
server_ip = '10.102.11.86'  # Replace with the server's IP address
server_port = 9997

# Create a socket object
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((server_ip, server_port))

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
        break

    # Encode the frame
    encoded_frame = cv2.imencode('.jpg', frame)[1]
    data = encoded_frame.tobytes()
    
    # Send the size of the data first
    msg_size = struct.pack('!I', len(data))
    client_socket.sendall(msg_size)
    
    # Send the actual data
    client_socket.sendall(data)

    # Display the frame locally (optional)
    cv2.imshow("Client Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
client_socket.close()
cv2.destroyAllWindows()