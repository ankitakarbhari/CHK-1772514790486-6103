from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.ml_service import detect_live_frame

router = APIRouter()


@router.post("/frame")
async def analyze_live_frame(file: UploadFile = File(...)):
    """
    Analyze a single frame from webcam for deepfake detection
    """
    try:
        frame_bytes = await file.read()

        result = detect_live_frame(frame_bytes)

        return {
            "type": "live_frame",
            "result": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))