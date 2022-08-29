import json
import os
import math
import librosa
import pandas as pd
from sklearn.preprocessing import LabelEncoder

SAMPLE_RATE = 22050
TRACK_DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def process_metadata():
    tracks = pd.read_csv("fma_metadata/tracks.csv", skiprows=1)
    tracks["track"] = tracks["Unnamed: 0"].apply(lambda x: str(x).zfill(6))
    tracks["subdir"] = tracks["Unnamed: 0"].apply(lambda x: str(x)[:-3].zfill(3))
    tracks["path"] = "fma_small" + "/" + tracks["subdir"] + "/" + tracks["track"] + ".mp3"
    tracks = tracks.iloc[1:, :]
    lb_make = LabelEncoder()
    tracks["genre_label"] = lb_make.fit_transform(tracks["genre_top"])

    return tracks


def save_mfcc(num_mfcc: int = 13, n_fft: int = 2048, hop_length: int = 512, num_segments: int = 5):
    """
    Extracts MFCCs from music dataset and saves them into a json file along witgh genre labels.
    :param num_mfcc: Number of coefficients to extract (int)
    :param n_fft: Interval we consider to apply FFT. Measured in # of samples
    :param hop_length: Sliding window for FFT. Measured in # of samples
    :param num_segments: Number of segments we want to divide sample tracks into
    :return:
    """

    tracks = process_metadata()

    # dictionary to store mapping, labels, and MFCCs

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    for split in tracks["split"].unique():
        data = {
            "mapping": [],
            "labels": [],
            "mfcc": []
        }

        df_i = tracks[tracks["split"] == split].reset_index()

        for i, row in df_i.iloc[0: 100, :].iterrows():
            # semantic_label = row["genre_top"]
            # data["mapping"].append(semantic_label)

            try:

                signal, sample_rate = librosa.load(row["path"], sr=SAMPLE_RATE)

                # process all segments of audio file
                for d in range(num_segments):

                    # calculate start and finish sample for current segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # extract mfcc, transpose
                    mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                                hop_length=hop_length).T

                    # store only mfcc feature with expected number of vectors
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(row["genre_label"])

            except FileNotFoundError:
                continue

        json_path = f"{split}_data_fma_small.json"

        # save MFCCs to json file
        with open(json_path, "w") as fp:
            json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(num_segments=10)
