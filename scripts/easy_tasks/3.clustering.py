## Load Data
import numpy as np

Z = np.load("features/latent_vectors.npy")   # VAE output
X = np.load("features/X_features.npy")       # original features


## Choose K
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

Ks = range(2, 10)

for k in Ks:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(Z)

    score = silhouette_score(Z, labels)
    print(f"K={k}, Silhouette={score:.4f}")

## K-mean Clustering
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
labels_kmeans = kmeans.fit_predict(Z)

## Cluster insight
df["cluster"] = labels_kmeans
print(df.groupby(["cluster", "language"]).size())

for c in df["cluster"].unique():
    subset = df[df["cluster"] == c]
    counts = subset["language"].value_counts(normalize=True)
    print(f"Cluster {c}:\n{counts}\n")


