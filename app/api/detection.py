from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.ml_service import (
    detect_image_deepfake,
    detect_video_deepfake,
    detect_audio_deepfake
)

router = APIRouter()


@router.post("/image")
async def detect_image(file: UploadFile = File(...)):
    """
    Detect deepfake in an uploaded image
    """
    try:
        contents = await file.read()
        result = detect_image_deepfake(contents)

        return {
            "type": "image",
            "filename": file.filename,
            "result": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/video")
async def detect_video(file: UploadFile = File(...)):
    """
    Detect deepfake in an uploaded video
    """
    try:
        contents = await file.read()
        result = detect_video_deepfake(contents)

        return {
            "type": "video",
            "filename": file.filename,
            "result": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/audio")
async def detect_audio(file: UploadFile = File(...)):
    """
    Detect deepfake in uploaded audio
    """
    try:
        contents = await file.read()
        result = detect_audio_deepfake(contents)

        return {
            "type": "audio",
            "filename": file.filename,
            "result": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))