from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.ml_service import detect_ai_generated_text

router = APIRouter()


class TextRequest(BaseModel):
    text: str


@router.post("/analyze")
async def analyze_text(request: TextRequest):
    """
    Detect if the given text is AI generated or manipulated
    """
    try:
        result = detect_ai_generated_text(request.text)

        return {
            "type": "text",
            "result": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))