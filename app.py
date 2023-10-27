import uvicorn
from fastapi import FastAPI, UploadFile, FastAPI, File, UploadFile, Form 

from fastapi.responses import JSONResponse, FileResponse
from typing import Annotated, Optional, List

from fastapi import FastAPI, File, UploadFile, Form 

from redis import Redis
import json
from rq import Queue
from dotenv import load_dotenv

import os, requests

app = FastAPI()

redis_conn = Redis(
    host=os.getenv('REDIS_HOST'),
    port=os.getenv('REDIS_PORT'),
    password=os.getenv('REDIS_PASSWORD')
)

task_queue = Queue(connection=redis_conn, default_timeout=36000)

def background_process():
    pass

@app.get("/")
async def root_route():
    return "Application working"

@app.post("/ml_extraction")
async def ml_extraction(
    is_training: Annotated[bool, Form()],
    contract_id: Annotated[str, Form()],
    c_pk: Annotated[str, Form()],
    text_file: Annotated[UploadFile, File()],
    categorial_keys: Annotated[str, Form()],
):
    print(is_training)
    print(contract_id)
    print(c_pk)
    print(text_file)
    print(categorial_keys)
    return JSONResponse(
        status_code=200,
        content=f"Processing in Queue"
    )

if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
