import logging
from fastapi import APIRouter, Body

from surogate import huggingface

logging.basicConfig(level=logging.ERROR)

router = APIRouter(tags=["model"])

@router.get(path="/model/download_size")
def get_download_size(model_id: str, allow_patterns: list = []):
    try:
        download_size_in_bytes = huggingface.get_huggingface_download_size(model_id, allow_patterns)
    except Exception as e:
        logging.error(f"Error in get_model_download_size: {type(e).__name__}: {e}")
        return {"status": "error", "message": "An internal error has occurred."}

    return {"status": "success", "data": download_size_in_bytes}

@router.get(path="/model/details")
async def get_model_details(model_id: str):
    try:
        model_details = await huggingface.get_model_details_from_huggingface(model_id)
    except Exception as e:
        logging.error(f"Error in get_model_details: {type(e).__name__}: {e}")
        return {"status": "error", "message": "An internal error has occurred."}

    return {"status": "success", "data": model_details}
