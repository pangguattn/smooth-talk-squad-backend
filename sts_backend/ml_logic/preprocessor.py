import numpy as np
import os
from tqdm import tqdm
from phonet import Phonet
import librosa

def preprocess_features() -> np.ndarray:

    phon=Phonet(["consonantal",
                        "back",
                        "anterior",
                        "open",
                        "close",
                        "nasal",
                        "stop",
                        "continuant",
                        "lateral",
                        "flap",
                        "trill",
                        "voice",
                        "strident",
                        "labial",
                        "dental",
                        "velar",
                        "pause",
                        "vocalic"])

    mfcc_features = []
    phon_list = []
    directory = os.path.join(os.pardir,'audio','splits')

    for clip in tqdm(os.listdir(directory)):
        audio, sample_rate = librosa.load(os.path.join(directory, clip) , sr=8000)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
        phon_data = phon.get_phon_wav(os.path.join(directory, clip), "" , False)
        if phon_data.shape == (299,20) and mfccs.shape == (20,47):
            # x mfcc
            mfcc_features.append(mfccs)
            # x phoneme
            phon_list.append(phon_data)
        else:
            print(f"{clip}:{mfccs.shape}")

    x_mfcc = np.stack(mfcc_features)
    phon_np = np.stack(phon_list)

    return x_mfcc, phon_np
