import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sts_backend.audio.splitter import splitter
import os
import numpy as np
from sts_backend.ml_logic.registry import load_model
from sts_backend.ml_logic.preprocessor import preprocess_features
from sts_backend.utils import delete_files_in_directory

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

UPLOAD_DIRECTORY = os.path.join("sts_backend","audio","uploads")
SPLITS_DIRECTORY = os.path.join("sts_backend","audio","splits")

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

# http://127.0.0.1:8000/predict
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
    print(file_location)
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())
    splitter(file.filename, file_location)

    # Load model
    model = load_model()
    assert model is not None
    model.summary() # Keep for debugging purpose
    X_processed = preprocess_features()
    y_pred = model.predict(X_processed)
    print(f"predict:{y_pred}") # Keep for debugging purpose

    # Response contents
    threshold = 0.6 # TO DISCUSS
    if any(y > threshold for y in y_pred):
        params = {"isStutter" : 1}
    else:
        params = {"isStutter" : 0}
    print(params)

    # Remove audio files
    delete_files_in_directory(UPLOAD_DIRECTORY)
    delete_files_in_directory(SPLITS_DIRECTORY)

    return params

@app.get("/")
def root():
    params = {'greeting': 'Hello'}
    return params
