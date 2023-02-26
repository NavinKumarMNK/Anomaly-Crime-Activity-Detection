import asyncio
import cv2
import numpy as np
import websockets

async def send_frames_to_websocket():
    async with websockets.connect("ws://localhost:8765") as websocket:
        print(f"Connected to websocket server at {websocket.remote_address}")
        cap = cv2.VideoCapture(0)
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % 5 == 0:
                    frame_bytes = cv2.imencode(".jpg", frame)[1].tobytes()
                    await websocket.send(frame_bytes)
                frame_count += 1        

        finally:
            cap.release()

if __name__ == "__main__":
    asyncio.run(send_frames_to_websocket())
