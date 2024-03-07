import numpy as np
import os
from tqdm import tqdm
import librosa
import pandas as pd

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

def preprocess_csv_data() -> None:
    # :oad the raw data from csv file
    data_sep28k_labels = pd.read_csv('../raw_data/SEP-28k_labels.csv', dtype={'EpId':str})
    fluencybank_labels = pd.read_csv('../raw_data/fluencybank_labels.csv', dtype={'EpId':str})
    fluencybank_labels["EpId"] = fluencybank_labels["EpId"].str.strip()
    data_sep28k_labels = pd.concat([data_sep28k_labels, fluencybank_labels], axis=0)

    # Create unique 'Name' column
    data_sep28k_labels['Name'] = data_sep28k_labels[data_sep28k_labels.columns[0:3]].apply(
        lambda x: '_'.join(x.dropna().astype(str)),axis=1)

    # Create isStutter column
    # When NoStutteredWords equals 2 or 3, set isStutter to 0
    # When NoStutteredWords equals 1 or 0, set isStutter to 1
    data_sep28k_labels['isStutter_by_2_more_reviewers'] = ""
    data_sep28k_labels.loc[data_sep28k_labels["NoStutteredWords"] <= 1.0, "isStutter_by_2_more_reviewers"] = 1
    data_sep28k_labels.loc[data_sep28k_labels["NoStutteredWords"] >= 2.0, "isStutter_by_2_more_reviewers"] = 0

    # Encode stutter features
    data_sep28k_labels['ProlongationEncoded'] = ""
    data_sep28k_labels['BlockEncoded'] = ""
    data_sep28k_labels['SoundRepEncoded'] = ""
    data_sep28k_labels['WordRepEncoded'] = ""

    encodeStutterFeatureFunction = lambda x : 1 if x > 1.0 else 0
    data_sep28k_labels["ProlongationEncoded"] = data_sep28k_labels["Prolongation"].map(encodeStutterFeatureFunction)
    data_sep28k_labels["BlockEncoded"] = data_sep28k_labels["Block"].map(encodeStutterFeatureFunction)
    data_sep28k_labels["SoundRepEncoded"] = data_sep28k_labels["SoundRep"].map(encodeStutterFeatureFunction)
    data_sep28k_labels["WordRepEncoded"] = data_sep28k_labels["WordRep"].map(encodeStutterFeatureFunction)

    # Keep rows in which Unsure, PoorAudioQuality, DifficultToUnderstand, NoSpeech, MUsic is 0
    data_sep28k_labels = data_sep28k_labels[data_sep28k_labels["Unsure"] == 0]
    data_sep28k_labels = data_sep28k_labels[data_sep28k_labels["PoorAudioQuality"] == 0]
    data_sep28k_labels = data_sep28k_labels[data_sep28k_labels["DifficultToUnderstand"] == 0]
    data_sep28k_labels = data_sep28k_labels[data_sep28k_labels["NoSpeech"] == 0]
    data_sep28k_labels = data_sep28k_labels[data_sep28k_labels["Music"] == 0]

    # Drop unused columns
    data_sep28k_labels.drop(['Unsure', 'PoorAudioQuality', 'DifficultToUnderstand', 'NoSpeech'], axis=1, inplace=True)
    data_sep28k_labels.drop(['Show', 'EpId', 'ClipId', 'Start', 'Stop'], axis=1, inplace=True)

    # Drop data from StrongVoices and StutteringIsCool (we don't have their audo)
    data_sep28k_labels = data_sep28k_labels[data_sep28k_labels["Name"].str.contains('StrongVoices') == False]
    data_sep28k_labels = data_sep28k_labels[data_sep28k_labels["Name"].str.contains('StutteringIsCool') == False]

    # Output Processed data_sep28k_labels into csv file
    output_data_directory = os.path.join(os.pardir, 'output_data') # You have to create this folder manually
    data_sep28k_labels.to_csv(os.path.join(output_data_directory, 'processed_sep28k_fluencybank_labels.csv'))
