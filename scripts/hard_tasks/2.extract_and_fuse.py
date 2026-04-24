import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder

Z_audio = np.load("features/latent_audio_beta.npy")
lyrics = np.load("features/lyrics_embeddings.npy")

df_all = pd.read_csv("datasets/final_music_dataset.csv")
df = pd.read_csv("datasets/aligned_dataset.csv")

# audio ids
audio_ids = sorted([int(f.split(".")[0]) for f in os.listdir("datasets/audio") if f.endswith(".wav")])
id_map = {id_:i for i,id_ in enumerate(audio_ids)}

Z_audio_aligned=[]
lyrics_aligned=[]

for _,row in df.iterrows():
    sid=row["id"]
    if sid in id_map:
        Z_audio_aligned.append(Z_audio[id_map[sid]])
        lyrics_aligned.append(lyrics[row.name])

Z_audio_aligned=np.array(Z_audio_aligned)
lyrics_aligned=np.array(lyrics_aligned)

# genre
enc = OneHotEncoder(sparse_output=False)
genre = enc.fit_transform(df[["tag"]])

Z = np.concatenate([Z_audio_aligned, lyrics_aligned, genre], axis=1)
Z = StandardScaler().fit_transform(Z)

np.save("features/Z_beta_multimodal.npy",Z)

print("Saved:", Z.shape)
