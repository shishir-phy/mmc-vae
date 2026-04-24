## Clean and Normalize text
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)  # remove [chorus]
    text = re.sub(r'[^a-zA-Z\u0980-\u09FF\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df["lyrics"] = df["lyrics"].apply(clean_text)


## Multilingual Embedding
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
lyrics_embeddings = model.encode(
    df["lyrics"].tolist(),
    batch_size=32,
    show_progress_bar=True
)


## Metadata Feature Engineering

# Genre
le = LabelEncoder()
df["tag_encoded"] = le.fit_transform(df["tag"].astype(str))

# Year
scaler = MinMaxScaler()
df["year_scaled"] = scaler.fit_transform(df[["year"]])

# Views
df["views_log"] = np.log1p(df["views"])
df["views_scaled"] = scaler.fit_transform(df[["views_log"]])


## Combine Features
# select metadata
meta_features = df[[
    "tag_encoded",
    "year_scaled",
    "views_scaled"
]].values

# concatenate
X = np.concatenate([lyrics_embeddings, meta_features], axis=1)


## Normalize final features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

## Dimensity Reduction(PCA)
pca = PCA(n_components=128)
X_reduced = pca.fit_transform(X_scaled)

## Save feature matrix
np.save("features/X_features.npy", X_scaled)


