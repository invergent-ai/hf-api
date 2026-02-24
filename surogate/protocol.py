from enum import IntEnum

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    code: int

class ErrorCode(IntEnum):
    VALIDATION_TYPE_ERROR = 40001

    INVALID_MODEL = 41001

    INTERNAL_ERROR = 50000