from fastapi import FastAPI, UploadFile
import shutil
import green_channel
from pathlib import Path

app = FastAPI() 

@app.post("/uploadfile/")
async def file_upload(file: UploadFile):
    with open(f"./{file.filename}", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    bpm = green_channel.fine_green_channel(file.filename)
    return {"Calculated BPM" : bpm}    

