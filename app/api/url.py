from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.ml_service import verify_url_content

router = APIRouter()


class URLRequest(BaseModel):
    url: str


@router.post("/verify")
async def verify_url(request: URLRequest):
    """
    Verify authenticity of media from a given URL
    """
    try:
        result = verify_url_content(request.url)

        return {
            "type": "url_verification",
            "url": request.url,
            "result": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))