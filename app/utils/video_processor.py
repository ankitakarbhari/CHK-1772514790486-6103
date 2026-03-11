import cv2
import numpy as np
import tempfile
from PIL import Image


def extract_video_frames(video_bytes, max_frames=20):

    frames = []

    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(video_bytes)
        video_path = temp.name

    cap = cv2.VideoCapture(video_path)

    count = 0

    while cap.isOpened() and count < max_frames:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(frame)

        frames.append(image)

        count += 1

    cap.release()

    return frames