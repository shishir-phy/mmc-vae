import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import umap

# =========================
# CONFIG
# =========================
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================
Z = np.load("features/Z_beta_multimodal.npy")
df = pd.read_csv("datasets/aligned_dataset.csv")

print("Z shape:", Z.shape)

# =========================
# CLUSTERING (for coloring)
# =========================
labels = KMeans(n_clusters=2, random_state=42).fit_predict(Z)

# =========================
# 1. t-SNE
# =========================
print("Running t-SNE...")
tsne = TSNE(n_components=2, random_state=42)
Z_tsne = tsne.fit_transform(Z)

plt.figure(figsize=(8,6))
plt.scatter(Z_tsne[:,0], Z_tsne[:,1], c=labels, cmap='viridis', s=10)
plt.title("t-SNE: β-VAE Multimodal Latent Space")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.colorbar(label="Cluster")

tsne_path = os.path.join(PLOT_DIR, "hard_tasks_tsne_clusters.png")
plt.savefig(tsne_path, dpi=300)
plt.close()

print("Saved:", tsne_path)

# =========================
# 2. UMAP
# =========================
print("Running UMAP...")
reducer = umap.UMAP(n_components=2, random_state=42)
Z_umap = reducer.fit_transform(Z)

plt.figure(figsize=(8,6))
plt.scatter(Z_umap[:,0], Z_umap[:,1], c=labels, cmap='plasma', s=10)
plt.title("UMAP: β-VAE Multimodal Latent Space")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.colorbar(label="Cluster")

umap_path = os.path.join(PLOT_DIR, "hard_tasks_umap_clusters.png")
plt.savefig(umap_path, dpi=300)
plt.close()

print("Saved:", umap_path)

# =========================
# 3. Language Distribution Plot
# =========================
df["cluster"] = labels

lang_counts = df.groupby(["cluster", "language"]).size().unstack(fill_value=0)

lang_counts.plot(kind="bar", stacked=True, figsize=(8,6))
plt.title("Cluster Distribution by Language")
plt.xlabel("Cluster")
plt.ylabel("Count")

lang_path = os.path.join(PLOT_DIR, "language_distribution.png")
plt.savefig(lang_path, dpi=300)
plt.close()

print("Saved:", lang_path)

# =========================
# 4. Genre Distribution (Top 10)
# =========================
genre_counts = df.groupby(["cluster", "tag"]).size().reset_index(name="count")
top_genres = genre_counts.sort_values("count", ascending=False).head(10)

plt.figure(figsize=(10,6))
for cluster in top_genres["cluster"].unique():
    subset = top_genres[top_genres["cluster"] == cluster]
    plt.bar(subset["tag"], subset["count"], alpha=0.6, label=f"Cluster {cluster}")

plt.xticks(rotation=45)
plt.title("Top Genre Distribution per Cluster")
plt.legend()

genre_path = os.path.join(PLOT_DIR, "genre_distribution.png")
plt.savefig(genre_path, dpi=300)
plt.close()

print("Saved:", genre_path)

print("\nAll plots saved in 'plots/' directory")
