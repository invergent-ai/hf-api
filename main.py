import argparse
from contextlib import asynccontextmanager

import fastapi
import urllib3
import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse

from surogate.protocol import ErrorResponse, ErrorCode
from surogate.routers import model, dataset

urllib3.disable_warnings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    print("hf-api LIFESPAN: Complete")

tags_metadata = [
    {
        "name": "dataset",
        "description": "Actions used to manage the datasets.",
    },
    {
        "name": "model",
        "description": "Actions for interacting with huggingface models",
    }
]

app = fastapi.FastAPI(
    title="Surogate Studio HF API",
    summary="An API for working with LLMs.",
    version="1.0",
    lifespan=lifespan,
    openapi_tags=tags_metadata,
)

def create_error_response(code: int, message: str) -> JSONResponse:
    return JSONResponse(ErrorResponse(message=message, code=code).model_dump(), status_code=400)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return create_error_response(ErrorCode.VALIDATION_TYPE_ERROR, str(exc))

app.include_router(model.router)
app.include_router(dataset.router)

def parse_args():
    parser = argparse.ArgumentParser(description="Surogate Studio HF API server.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="host name")
    parser.add_argument("--port", type=int, default=9090, help="port number")
    return parser.parse_args()

def run():
    args = parse_args()
    print(f"args: {args}")
    uvicorn.run("main:app", host=args.host, port=args.port, log_level="warning")

if __name__ == "__main__":
    run()
