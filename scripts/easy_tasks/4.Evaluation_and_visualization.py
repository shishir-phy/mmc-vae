import os
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

Z = np.load("features/latent_vectors.npy")  
X = np.load("features/X_features.npy")   
labels_kmeans = np.load("features/labels_kmeans.npy")

k=3
## Silhouette Score
sil_kmeans = silhouette_score(Z, labels_kmeans)

## Calinski-Harabasz index
ch_kmeans = calinski_harabasz_score(Z, labels_kmeans)

## Davies-Bouldin index
db_kmeans = davies_bouldin_score(Z, labels_kmeans)

## Apply PCA
pca = PCA(n_components=16)
X_pca = pca.fit_transform(X)

## K-means on PCA
kmeans_pca = KMeans(n_clusters=k, random_state=42)
labels_pca = kmeans_pca.fit_predict(X_pca)

## Evaluate Baseline
sil_pca = silhouette_score(X_pca, labels_pca)
ch_pca = calinski_harabasz_score(X_pca, labels_pca)
db_pca = davies_bouldin_score(X_pca, labels_pca)


## Compare Results
results = pd.DataFrame({
    "Method": ["VAE+KMeans", "PCA+KMeans"],
    "Silhouette": [sil_kmeans, sil_pca],
    "CH Index": [ch_kmeans, ch_pca],
    "DB Index": [db_kmeans, db_pca]
})

print(results)

os.makedirs('results', exist_ok=True)
results.to_csv("results/easy_results.csv", index=False)

####### Visualization ############
print("Generating t-SNE plot")
### t-SNE
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2, random_state=42)
Z_2d = tsne.fit_transform(Z)

plt.figure(figsize=(8,6))
plt.scatter(Z_2d[:,0], Z_2d[:,1], c=labels_kmeans)
#plt.scatter(Z_2d[:,0], Z_2d[:,1], c=df["language"].map({"English":0,"Bangla":1}))
plt.title("VAE Latent Space Clustering")
plt.savefig("plots/easy_tasks_tSNE_plot.png")
print("Plot Generated: plots/easy_tasks_tSNE_plot.png")
plt.close()



print("Generating UMAP plot")
## UMAP
import umap

reducer = umap.UMAP()
Z_umap = reducer.fit_transform(Z)

plt.scatter(Z_umap[:,0], Z_umap[:,1], c=labels_kmeans)
plt.title("UMAP Clusters")
plt.savefig("plots/easy_tasks_UMAP_plot.png")
print("Plot Generated: plots/easy_tasks_UMAP_plot.png")
plt.close()


