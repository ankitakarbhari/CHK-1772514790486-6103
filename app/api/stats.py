from fastapi import APIRouter, HTTPException
from app.services.stats_service import get_system_stats

router = APIRouter()


@router.get("/")
async def stats():
    """
    Return system detection statistics
    """
    try:
        stats = get_system_stats()

        return {
            "status": "success",
            "data": stats
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))