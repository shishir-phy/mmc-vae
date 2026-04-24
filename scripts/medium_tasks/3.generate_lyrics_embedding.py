import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# =========================
# CONFIG
# =========================
DATASET_CSV = "datasets/final_music_dataset.csv"
OUTPUT_FILE = "features/lyrics_embeddings.npy"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATASET_CSV)

print("Dataset:", df.shape)

# =========================
# LOAD MODEL
# =========================
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# =========================
# GENERATE EMBEDDINGS
# =========================
lyrics_list = df["lyrics"].fillna("").tolist()

embeddings = model.encode(
    lyrics_list,
    batch_size=32,
    show_progress_bar=True
)

embeddings = np.array(embeddings)

print("Embeddings shape:", embeddings.shape)

# =========================
# SAVE
# =========================
np.save(OUTPUT_FILE, embeddings)

print("Saved:", OUTPUT_FILE)
