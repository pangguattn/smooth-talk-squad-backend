import numpy as np
import os
from tqdm import tqdm
import librosa

def preprocess_features() -> np.ndarray:

    mfcc_features = []
    directory = os.path.join(os.pardir,'audio','splits')

    for clip in tqdm(os.listdir(directory)):
        audio, sample_rate = librosa.load(os.path.join(directory, clip) , sr=8000)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
        if mfccs.shape == (20,47):
            mfcc_features.append(mfccs)
        else:
            print(f"{clip}:{mfccs.shape}")

    x_mfcc = np.stack(mfcc_features)

    return x_mfcc
