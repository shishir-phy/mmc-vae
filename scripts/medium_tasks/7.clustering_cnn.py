import os
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

# =========================
# LOAD DATA
# =========================
Z = np.load("features/Z_cnn_multimodal.npy")
df = pd.read_csv("datasets/aligned_dataset.csv")

print("Latent shape:", Z.shape)

# =========================
# STEP 1: CHECK COLLAPSE
# =========================
latent_std = np.std(Z, axis=0).mean()
print("Mean latent std:", latent_std)

if latent_std < 0.01:
    print("⚠️ WARNING: Latent space appears collapsed!")

# =========================
# STEP 2: NORMALIZE LATENT
# =========================
scaler = StandardScaler()
Z = scaler.fit_transform(Z)

# =========================
# STEP 3: AUTO SELECT K
# =========================
def find_best_k(Z, k_range=range(2, 8)):
    best_k = None
    best_score = -1

    for k in k_range:
        try:
            labels = KMeans(n_clusters=k, random_state=42).fit_predict(Z)

            if len(set(labels)) < 2:
                continue

            score = silhouette_score(Z, labels)

            if score > best_score:
                best_score = score
                best_k = k

        except:
            continue

    return best_k if best_k else 2


k = find_best_k(Z)
print("Selected K:", k)

# =========================
# METRIC FUNCTION
# =========================
def evaluate(Z, labels, name):
    unique_labels = set(labels)

    if len(unique_labels) <= 1 or (len(unique_labels) == 2 and -1 in unique_labels):
        print(f"{name}: Invalid clustering (1 cluster or noise only)")
        return [name, None, None, None]

    try:
        sil = silhouette_score(Z, labels)
    except:
        sil = None

    try:
        dbi = davies_bouldin_score(Z, labels)
    except:
        dbi = None

    try:
        ari = adjusted_rand_score(df["language"], labels)
    except:
        ari = None

    return [name, sil, dbi, ari]

# =========================
# K-MEANS
# =========================
try:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels_kmeans = kmeans.fit_predict(Z)
except:
    labels_kmeans = np.zeros(len(Z))

# =========================
# AGGLOMERATIVE
# =========================
try:
    agg = AgglomerativeClustering(n_clusters=k)
    labels_agg = agg.fit_predict(Z)
except:
    labels_agg = np.zeros(len(Z))

# =========================
# DBSCAN (auto eps search)
# =========================
def run_dbscan(Z):
    for eps in [0.5, 1.0, 1.5, 2.0, 3.0]:
        db = DBSCAN(eps=eps, min_samples=5)
        labels = db.fit_predict(Z)

        if len(set(labels)) > 1:
            print(f"DBSCAN worked with eps={eps}")
            return labels

    print("DBSCAN failed → returning single cluster")
    return np.zeros(len(Z))


labels_db = run_dbscan(Z)

# =========================
# EVALUATE
# =========================
results = []

results.append(evaluate(Z, labels_kmeans, "VAE+KMeans"))
results.append(evaluate(Z, labels_agg, "VAE+Agglomerative"))
results.append(evaluate(Z, labels_db, "VAE+DBSCAN"))

res_df = pd.DataFrame(results, columns=["Method", "Silhouette", "DB Index", "ARI"])

print("\n=== RESULTS ===")
print(res_df)


os.makedirs('results', exist_ok=True)
res_df.to_csv("results/medium_results.csv", index=False)

# =========================
# EXTRA ANALYSIS
# =========================
print("\n=== Cluster Distribution (KMeans) ===")

df["cluster"] = labels_kmeans

print(df.groupby(["cluster", "language"]).size())
print("\nTop genres per cluster:")
print(df.groupby(["cluster", "tag"]).size().sort_values(ascending=False).head(10))
