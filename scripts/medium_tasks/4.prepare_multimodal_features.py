import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# =========================
# CONFIG
# =========================
DATASET_CSV = "datasets/final_music_dataset.csv"
MFCC_FILE = "features/mfcc_features.npy"
LYRICS_EMB_FILE = "features/lyrics_embeddings.npy"   # you should have this saved earlier

OUTPUT_FEATURES = "features/X_multimodal.npy"
OUTPUT_DF = "datasets/aligned_dataset.csv"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATASET_CSV)

mfcc = np.load(MFCC_FILE)              # (N_audio, 40, 130)
lyrics_emb = np.load(LYRICS_EMB_FILE)  # (N_total, 384)

print("Original dataset:", df.shape)
print("MFCC shape:", mfcc.shape)
print("Lyrics shape:", lyrics_emb.shape)

# =========================
# STEP 1: GET AVAILABLE AUDIO IDS
# =========================
# Assumption: MFCC was extracted in order of audio filenames (id.wav)

import os

AUDIO_DIR = "datasets/audio"

audio_ids = []
for file in os.listdir(AUDIO_DIR):
    if file.endswith(".wav"):
        audio_ids.append(int(file.split(".")[0]))

audio_ids = sorted(audio_ids)

print("Audio files found:", len(audio_ids))

# =========================
# STEP 2: FILTER DATASET
# =========================
df = df[df["id"].isin(audio_ids)].copy()

# sort to ensure consistent order
df = df.sort_values("id").reset_index(drop=True)

print("Filtered dataset:", df.shape)

# =========================
# STEP 3: ALIGN MFCC WITH IDS
# =========================
# IMPORTANT: we must reorder MFCC according to sorted IDs

id_to_index = {id_: i for i, id_ in enumerate(audio_ids)}

mfcc_aligned = np.array([mfcc[id_to_index[id_]] for id_ in df["id"]])

print("Aligned MFCC:", mfcc_aligned.shape)

# =========================
# STEP 4: ALIGN LYRICS EMBEDDINGS
# =========================
# assuming lyrics_emb is in original df order

lyrics_aligned = lyrics_emb[df.index]

print("Aligned lyrics:", lyrics_aligned.shape)

# =========================
# STEP 5: FLATTEN MFCC
# =========================
mfcc_flat = mfcc_aligned.reshape(len(mfcc_aligned), -1)

print("MFCC flattened:", mfcc_flat.shape)

# =========================
# STEP 6: COMBINE FEATURES
# =========================
X_combined = np.concatenate([mfcc_flat, lyrics_aligned], axis=1)

print("Combined shape:", X_combined.shape)

# =========================
# STEP 7: NORMALIZE
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# =========================
# STEP 8: SAVE OUTPUT
# =========================
np.save(OUTPUT_FEATURES, X_scaled)
df.to_csv(OUTPUT_DF, index=False)

print("Saved:", OUTPUT_FEATURES)
print("Saved:", OUTPUT_DF)
