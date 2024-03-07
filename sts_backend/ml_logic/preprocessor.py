import numpy as np
import os
from tqdm import tqdm
import librosa
import pandas as pd

# You have to create these folders manually
OUT_DATA_FILEPATH =  os.path.join(os.pardir, 'output_data', 'processed_sep28k_fluencybank_labels.csv')
AUDIO_DIRECTORY = os.path.join(os.pardir,os.pardir, 'clips_test') # Clips folder should not in Smooth-Talk-Squad folder. It might mess up git because it is too big.

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
    # Load the raw data from csv file
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
    data_sep28k_labels.to_csv(OUT_DATA_FILEPATH)


def preprocess_training_data() :
    preprocessed_csv = pd.read_csv(OUT_DATA_FILEPATH)
    clips_list = list(preprocessed_csv["Name"])

    features=[]
    y_two_reviewer_list =[]
    y_stutter_feature_list = pd.DataFrame([])
    clip_name_list = []

    for filename in tqdm(os.listdir(AUDIO_DIRECTORY)):
        filename = filename[:-4] # Remove file extension (.wav in our situation)
        if clips_list.count(filename) > 0:
                audio, sample_rate = librosa.load(os.path.join(AUDIO_DIRECTORY, filename) + '.wav', sr=8000)
                mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)

                # Some mfccs.shape is not (20, 47) because the file is rather smaller
                # For future development, maybe we can add padding (maybe ??)
                if mfccs.shape == (20,47):
                    # mfcc
                    features.append(mfccs)

                    # stutter flag
                    y_two_reviewer = preprocessed_csv[preprocessed_csv["Name"] == filename]["isStutter_by_2_more_reviewers"]
                    y_two_reviewer_list.append(y_two_reviewer.astype(int))

                    # stutter features
                    y_stutter_feature = preprocessed_csv[preprocessed_csv["Name"] == filename][['ProlongationEncoded', 'BlockEncoded', 'SoundRepEncoded', 'WordRepEncoded']]
                    y_stutter_feature_list = pd.concat([y_stutter_feature_list, y_stutter_feature])

                    # clip names
                    clip_name = preprocessed_csv[preprocessed_csv["Name"] == filename]["Name"]
                    clip_name_list.append(clip_name)
                else:
                    print(f"{filename}:{mfccs.shape}")

    x_mfcc = np.stack(features)
    y_two_reviewer_np = np.array(y_two_reviewer_list)
    y_stutter_feature_np = np.array(y_stutter_feature_list)
    clip_name_list_np = np.array(clip_name_list)

    return x_mfcc, y_two_reviewer_np, y_stutter_feature_np, clip_name_list_np
