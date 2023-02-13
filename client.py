
import cv2
import utils.utils as utils
import cv2
import websocket
import _thread
import socket

# initialize the camera
cap = cv2.VideoCapture(0)

def on_message(ws, message):
    print(f"Message: {message}")

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws):
    print("Connection closed.")

def on_open(ws):
    client_ip = socket.gethostbyname(socket.gethostname())

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read the frame from the camera.")
            break

        # encode the frame in JPEG format
        _, jpg_image = cv2.imencode('.jpg', frame)

        # send the IP address and the frame to the server
        message = client_ip + jpg_image.tobytes()
        ws.send(message, opcode=websocket.ABNF.OPCODE_BINARY)

if __name__ == '__main__':
    websocket.enableTrace(True)
    network = utils.config_parse(utils.current_path(), 'NETWORK')
    ip = network['server_ip']
    port = network['server_port']
    url = f"ws://{ip}:{port}/video_feed"
    ws = websocket.WebSocketApp(url,
                              on_message=on_message,
                              on_error=on_error,
                              on_close=on_close)
    ws.on_open = on_open
    _thread.start_new_thread(ws.run_forever, ())
