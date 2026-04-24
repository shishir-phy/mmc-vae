import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

# =========================
# LOAD
# =========================
Z_audio = np.load("features/latent_audio.npy")          # (600, 32)
lyrics_all = np.load("features/lyrics_embeddings.npy") # (~600, 384)

df_all = pd.read_csv("datasets/final_music_dataset.csv")
df_aligned = pd.read_csv("datasets/aligned_dataset.csv")

AUDIO_DIR = "datasets/audio"

print("Audio latent:", Z_audio.shape)
print("Lyrics full:", lyrics_all.shape)
print("Aligned df:", df_aligned.shape)

# =========================
# STEP 1: GET AUDIO IDS
# =========================
audio_ids = []
for file in os.listdir(AUDIO_DIR):
    if file.endswith(".wav"):
        audio_ids.append(int(file.split(".")[0]))

audio_ids = sorted(audio_ids)

print("Audio IDs:", len(audio_ids))

# =========================
# STEP 2: MAP AUDIO → ID
# =========================
id_to_audio = {id_: i for i, id_ in enumerate(audio_ids)}

# =========================
# STEP 3: ALIGN EVERYTHING USING ID
# =========================
Z_audio_aligned = []
lyrics_aligned = []

for _, row in df_aligned.iterrows():
    song_id = row["id"]

    if song_id in id_to_audio:
        idx = id_to_audio[song_id]

        Z_audio_aligned.append(Z_audio[idx])
        lyrics_aligned.append(lyrics_all[row.name])  # row.name = original index

# convert
Z_audio_aligned = np.array(Z_audio_aligned)
lyrics_aligned = np.array(lyrics_aligned)

print("Aligned audio:", Z_audio_aligned.shape)
print("Aligned lyrics:", lyrics_aligned.shape)

# =========================
# FINAL CHECK
# =========================
assert Z_audio_aligned.shape[0] == lyrics_aligned.shape[0], "Still mismatch!"

# =========================
# COMBINE
# =========================
Z_combined = np.concatenate([Z_audio_aligned, lyrics_aligned], axis=1)

Z_combined = StandardScaler().fit_transform(Z_combined)

np.save("features/Z_cnn_multimodal.npy", Z_combined)

print("Final shape:", Z_combined.shape)
