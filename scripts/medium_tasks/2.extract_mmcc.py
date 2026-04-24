import librosa
import numpy as np
import os
from tqdm import tqdm

AUDIO_DIR = "datasets/audio"
OUT_FILE = "features/mfcc_features.npy"

TARGET_FRAMES = 130
N_MFCC = 40

mfcc_list = []
ids = []

def extract_mfcc(file):
    y, sr = librosa.load(file, sr=22050)

    # force 30 seconds
    target_len = 30 * sr
    if len(y) > target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    # enforce fixed time dimension
    if mfcc.shape[1] < TARGET_FRAMES:
        pad_width = TARGET_FRAMES - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)))
    else:
        mfcc = mfcc[:, :TARGET_FRAMES]

    return mfcc


for file in tqdm(os.listdir(AUDIO_DIR)):
    if file.endswith(".wav"):
        song_id = int(file.split(".")[0])
        path = os.path.join(AUDIO_DIR, file)

        try:
            mfcc = extract_mfcc(path)

            # sanity check
            assert mfcc.shape == (N_MFCC, TARGET_FRAMES)

            mfcc_list.append(mfcc)
            ids.append(song_id)

        except Exception as e:
            print(f"Failed: {file}, {e}")

mfcc_array = np.stack(mfcc_list)

np.save(OUT_FILE, mfcc_array)

print("Final MFCC shape:", mfcc_array.shape)
