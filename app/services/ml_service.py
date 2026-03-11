import io
import requests
import numpy as np
from PIL import Image
import torch

from app.utils.image_processor import preprocess_image
from app.utils.video_processor import extract_video_frames
from app.utils.audio_processor import preprocess_audio


# =========================
# Load PyTorch Models
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

efficientnet_model = torch.load("models/efficientnet_model.pt", map_location=device)
mobilenet_model = torch.load("models/mobilenet_model.pt", map_location=device)
ensemble_model = torch.load("models/ensemble_model.pt", map_location=device)
audio_model = torch.load("models/audio_deepfake_model.pt", map_location=device)
text_model = torch.load("models/text_detection_model.pt", map_location=device)

efficientnet_model.eval()
mobilenet_model.eval()
ensemble_model.eval()
audio_model.eval()
text_model.eval()


# =========================
# Helper
# =========================

def format_result(score):
    label = "fake" if score > 0.5 else "real"

    return {
        "prediction": label,
        "confidence": float(score)
    }


# =========================
# Image Detection
# =========================

def detect_image_deepfake(image_bytes):

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    tensor = preprocess_image(image).to(device)

    with torch.no_grad():

        pred1 = efficientnet_model(tensor)
        pred2 = mobilenet_model(tensor)

        combined = torch.cat((pred1, pred2), dim=1)

        score = torch.sigmoid(ensemble_model(combined)).item()

    return format_result(score)


# =========================
# Video Detection
# =========================

def detect_video_deepfake(video_bytes):

    frames = extract_video_frames(video_bytes)

    scores = []

    for frame in frames:

        tensor = preprocess_image(frame).to(device)

        with torch.no_grad():

            pred1 = efficientnet_model(tensor)
            pred2 = mobilenet_model(tensor)

            combined = torch.cat((pred1, pred2), dim=1)

            score = torch.sigmoid(ensemble_model(combined)).item()

            scores.append(score)

    avg_score = np.mean(scores)

    return format_result(avg_score)


# =========================
# Audio Detection
# =========================

def detect_audio_deepfake(audio_bytes):

    features = preprocess_audio(audio_bytes)

    tensor = torch.tensor(features).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = torch.sigmoid(audio_model(tensor)).item()

    return format_result(prediction)


# =========================
# Live Frame Detection
# =========================

def detect_live_frame(frame_bytes):

    image = Image.open(io.BytesIO(frame_bytes)).convert("RGB")

    tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        score = torch.sigmoid(efficientnet_model(tensor)).item()

    return format_result(score)


# =========================
# Text Detection
# =========================

def detect_ai_generated_text(text):

    # very simple tokenizer example
    tokens = [ord(c) for c in text[:512]]

    tensor = torch.tensor(tokens).unsqueeze(0).float().to(device)

    with torch.no_grad():
        score = torch.sigmoid(text_model(tensor)).item()

    return format_result(score)


# =========================
# URL Verification
# =========================

def verify_url_content(url):

    response = requests.get(url)

    content_type = response.headers.get("content-type", "")

    if "image" in content_type:
        return detect_image_deepfake(response.content)

    elif "video" in content_type:
        return detect_video_deepfake(response.content)

    elif "audio" in content_type:
        return detect_audio_deepfake(response.content)

    else:
        return {
            "prediction": "unknown",
            "confidence": 0.0
        }