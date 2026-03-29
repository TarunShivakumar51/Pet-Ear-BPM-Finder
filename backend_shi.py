from fastapi import FastAPI, UploadFile
import shutil
import green_channel
import pathlib as Path
import os

app = FastAPI() 

@app.post("/uploadfile/")
async def file_upload(file: UploadFile):
    with open(f"./{file.filename}", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    video_path = Path(file.filename)

    try:
        bpm = green_channel.find_green_channel(video_path)
        return {"Calculated BPM" : bpm}
    finally:
        if video_path.exists():
            os.remove(video_path)  