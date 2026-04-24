import numpy as np
import pandas as pd
import os

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from sklearn.decomposition import PCA

# =========================
# LOAD DATA
# =========================
Z = np.load("features/Z_beta_multimodal.npy")
df = pd.read_csv("datasets/aligned_dataset.csv")

print("Z shape:", Z.shape)
print("DF shape:", df.shape)

# =========================
# METRICS
# =========================
def purity(y_true, y_pred):
    total = 0
    for c in set(y_pred):
        idx = np.where(y_pred == c)[0]
        total += y_true.iloc[idx].value_counts().max()
    return total / len(y_true)


def evaluate(name, Z_data, labels):
    return [
        name,
        silhouette_score(Z_data, labels),
        davies_bouldin_score(Z_data, labels),
        adjusted_rand_score(df["language"], labels),
        normalized_mutual_info_score(df["language"], labels),
        purity(df["language"], labels)
    ]


results = []

# =========================
# β-VAE CLUSTERING
# =========================
labels_km = KMeans(n_clusters=2, random_state=42).fit_predict(Z)
labels_agg = AgglomerativeClustering(n_clusters=2).fit_predict(Z)

results.append(evaluate("β-VAE+KMeans", Z, labels_km))
results.append(evaluate("β-VAE+Agglomerative", Z, labels_agg))

# =========================
# PCA BASELINE (ALIGNED)
# =========================
X_pca = PCA(n_components=64).fit_transform(Z)
labels_pca = KMeans(n_clusters=2, random_state=42).fit_predict(X_pca)

results.append(evaluate("PCA+KMeans", X_pca, labels_pca))

# =========================
# MFCC BASELINE (ALIGNED BY ID)
# =========================
mfcc = np.load("features/mfcc_features.npy")

# get audio ids
audio_ids = sorted([
    int(f.split(".")[0])
    for f in os.listdir("datasets/audio")
    if f.endswith(".wav")
])

id_map = {id_: i for i, id_ in enumerate(audio_ids)}

mfcc_aligned = []

for _, row in df.iterrows():
    sid = row["id"]
    if sid in id_map:
        mfcc_aligned.append(mfcc[id_map[sid]])

mfcc_aligned = np.array(mfcc_aligned)

print("MFCC aligned:", mfcc_aligned.shape)

# flatten
mfcc_flat = mfcc_aligned.reshape(len(mfcc_aligned), -1)

labels_mfcc = KMeans(n_clusters=2, random_state=42).fit_predict(mfcc_flat)

results.append(evaluate("MFCC+KMeans", mfcc_flat, labels_mfcc))

# =========================
# RESULTS TABLE
# =========================
res_df = pd.DataFrame(
    results,
    columns=["Method", "Silhouette", "DB", "ARI", "NMI", "Purity"]
)

print("\n=== FINAL RESULTS ===")
print(res_df)

res_df.to_csv("results/hard_results.csv", index=False)

print("\nSaved results to hard_results.csv")
