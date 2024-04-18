import json
from channels.generic.websocket import WebsocketConsumer, AsyncWebsocketConsumer  # type: ignore
import os
from datetime import datetime
from django.conf import settings
import cv2
import numpy as np
import face_recognition
import shutil
import time
import imutils
from eye_blink_detection import f_detector


# Define the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Media files
MEDIA_ROOT = os.path.join(BASE_DIR, "media")
MEDIA_URL = "/media/"


class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_group_name = "Test-Room"

        await self.channel_layer.group_add(self.room_group_name, self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)
        print("Disconnected!")

    async def receive(self, text_data):
        receive_dict = json.loads(text_data)
        message = receive_dict["message"]
        print(message)
        action = receive_dict["action"]
        if action == "new-offer" or action == "new-answer":
            receiver_channel_name = receive_dict["message"]["receiver_channel_name"]
            receive_dict["message"]["receiver_channel_name"] = self.channel_name

            await self.channel_layer.send(
                receiver_channel_name,
                {"type": "send.sdp", "receive_dict": receive_dict},
            )
            return

        receive_dict["message"]["receiver_channel_name"] = self.channel_name

        await self.channel_layer.group_send(
            self.room_group_name, {"type": "send.sdp", "receive_dict": receive_dict}
        )

    async def sdp(self, event):
        receive_dict = event["receive_dict"]

        await self.send(text_data=json.dumps(receive_dict))


class SignalingConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()
        print("WebSocket connection established")

    def disconnect(self, close_code):
        print(f"WebSocket connection closed with code: {close_code}")

    def receive(self, text_data=None, bytes_data=None):
        print("Received signal")
        # print(text_data)
        # print(bytes_data)
        if bytes_data:
            self.save_image(bytes_data)
        if text_data:
            text_data = json.loads(text_data)
            print(text_data)
            if text_data.get("message") == "END":
                # print(text_data)
                frame_paths = self.handle_end_session()
                # decoded_frames = self.decode_frames(frames)
                id_image = os.path.join(MEDIA_ROOT, "my_id.jpg")
                for path in frame_paths:
                    status = self.ID_verification(id_image, path)
                    print(status)

                today = datetime.now().strftime("%Y-%m-%d")
                dir_path = os.path.join(MEDIA_ROOT, "received_images", today)
                self.delete_directory(dir_path)

        # data = json.loads(text_data)

        # Handle WebRTC signaling messages
        # if data.get("type") == "offer":
        #     self.handle_offer(data)
        # elif data.get("type") == "answer":
        #     self.handle_answer(data)
        # elif data.get("type") == "ice-candidate":
        #     self.handle_ice_candidate(data)
        # else:
        #     print("Unknown message type:", data.get("type"))

    def save_image(self, image_bytes):
        # Create a directory for today if it doesn't exist
        today = datetime.now().strftime("%Y-%m-%d")
        save_path = os.path.join(MEDIA_ROOT, "received_images", today)
        os.makedirs(save_path, exist_ok=True)

        # Construct a file name
        file_name = f'image_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpeg'
        full_path = os.path.join(save_path, file_name)

        # Write the image to a file
        with open(full_path, "wb") as file:
            file.write(image_bytes)
        print(f"Saved image to {full_path}")

    def handle_offer(self, offer):
        # print("Received offer:", offer)

        # Normally, you would store the offer and send it to the other peer
        # Here, for simplicity, we just echo it back
        self.send(
            text_data=json.dumps({"type": offer["type"], "sdp": offer["offer"]["sdp"]})
        )

    def handle_end_session(self):
        # Load images from today's directory
        today = datetime.now().strftime("%Y-%m-%d")
        directory_path = os.path.join(MEDIA_ROOT, "received_images", today)
        image_paths = []
        for filename in os.listdir(directory_path):
            if filename.endswith(".jpeg"):
                file_path = os.path.join(directory_path, filename)
                print(file_path)
                image_paths.append(file_path)

        print(f"Loaded {len(image_paths)} images for processing.")
        return image_paths

    def decode_frames(self, raw_frames):
        frames = []
        for raw_frame in raw_frames:
            np_arr = np.frombuffer(
                raw_frame, np.uint8
            )  # Convert binary data to numpy array
            frame = cv2.imdecode(
                np_arr, cv2.IMREAD_COLOR
            )  # Decode image data to OpenCV format
            if frame is not None:
                frames.append(frame)
            else:
                print("Failed to decode frame")
        return frames

    # Function to detect and crop the face using Haar cascades

    def find_and_crop_face(self, image_path):
        # Initialize the face classifier
        face_classifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        # Read the image
        image = cv2.imread(image_path)
        # Convert to grayscale for face detection
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_classifier.detectMultiScale(
            gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )

        # Sort the detected faces by size (largest first)
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)

        # Attempt to confirm if the largest detected region is a face
        for x, y, w, h in faces:
            # Extract the potential face ROI from the original (colored) image
            potential_face_roi = image[y : y + h, x : x + w]
            # Convert to RGB for face_recognition compatibility
            rgb_face_roi = cv2.cvtColor(potential_face_roi, cv2.COLOR_BGR2RGB)

            # Check if face_recognition can find a face in the ROI
            face_encodings = face_recognition.face_encodings(rgb_face_roi)

            # If face_recognition found a face, it's likely to be a real face
            if face_encodings:
                # Return the first real face and its encoding
                return rgb_face_roi, face_encodings[0]

        # If no faces pass the check, return None
        return None, None

    def ID_verification(self, id_image, user_image):

        id_face_crop, id_face_encoding = self.find_and_crop_face(id_image)
        person_image = face_recognition.load_image_file(user_image)
        # Check if a face was found and cropped from the ID card
        if id_face_crop is not None and id_face_encoding is not None:
            # Encode the face from the person's image
            person_face_encodings = face_recognition.face_encodings(person_image)
            if person_face_encodings:
                person_face_encoding = person_face_encodings[0]
                # Compare the faces and calculate face distance
                results = face_recognition.compare_faces(
                    [id_face_encoding], person_face_encoding
                )
                face_distances = face_recognition.face_distance(
                    [id_face_encoding], person_face_encoding
                )

                # Print the results
                print(f"Are the two faces of the same person? {results[0]}")
                print(f"Face distance: {face_distances[0]}")
                if results[0]:
                    return True
                else:
                    return False

            else:
                print("No face found in the person's image.")
                return False
        else:
            print(
                "No face found in the ID card image, or the detected face was not clear enough."
            )
            return False

    def delete_directory(self, dir_path):
        # Check if the directory exists
        if os.path.exists(dir_path):
            # Iterate over all the files and subdirectories in the directory
            for filename in os.listdir(dir_path):
                file_path = os.path.join(dir_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # Remove each file
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # Remove each subdirectory
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")

            # After all files and subdirectories have been removed, delete the empty directory
            try:
                os.rmdir(dir_path)
                print(f"Successfully deleted the directory: {dir_path}")
            except Exception as e:
                print(f"Failed to delete the directory: {dir_path}. Reason: {e}")
        else:
            print(f"The directory {dir_path} does not exist")

    def handle_answer(self, answer):
        # print("Received answer:", answer)

        # print("Sending answer:", answer)

        # Similarly, process and/or forward the answer to the initiating peer
        self.send(
            text_data=json.dumps(
                {"type": answer["type"], "sdp": answer["answer"]["sdp"]}
            )
        )

    def handle_ice_candidate(self, ice_candidate):
        # print("Received ICE candidate:", ice_candidate)
        # ICE candidates are usually forwarded to the other peer
        self.send(
            text_data=json.dumps(
                {
                    "type": ice_candidate["type"],
                    "candidate": ice_candidate["candidate"]["candidate"],
                    "sdpMLineIndex": ice_candidate["candidate"]["sdpMLineIndex"],
                    "sdpMid": ice_candidate["candidate"]["sdpMid"],
                }
            )
        )
