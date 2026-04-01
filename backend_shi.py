from fastapi import FastAPI, UploadFile, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import green_channel
from pathlib import Path
import os

app = FastAPI()

app.add_middleware(CORSMiddleware,
                   allow_origins = ["*"],
                   allow_methods = ["*"],
                   allow_headers = ["*"]
                )

@app.post("/uploadfile")
async def file_upload(file: UploadFile, request: Request):
    with open(f"./{file.filename}", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    video_path = Path(file.filename)

    try:
        bpm = green_channel.find_green_channel(video_path)

        if request.is_disconnected():
            return {"Disconnected" : "Client has been disconneted"}
        
        return {"Calculated BPM" : bpm}
    finally:
        if video_path.exists():
            os.remove(video_path)  