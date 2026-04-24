# 🎵 Multimodal Music Clustering using VAE

This project explores unsupervised clustering of hybrid-language music (Bangla + English) using a combination of **audio features, lyrics embeddings, and genre information**. The goal is to learn meaningful representations of songs and group them based on similarity.

We implemented multiple models including **VAE, CNN-VAE, and β-VAE**, and compared them with baseline methods such as PCA and raw MFCC clustering.

---

## 🚀 Key Highlights

- Multimodal learning (audio + lyrics + genre)
- Deep generative models (VAE, CNN-VAE, β-VAE)
- Multiple clustering methods (KMeans, Agglomerative, DBSCAN)
- Comprehensive evaluation (Silhouette, ARI, NMI, Purity)
- Visualization using t-SNE and UMAP

---

## 📊 Final Results Summary

| Method | Silhouette | DB | ARI | NMI | Purity |
|--------|-----------|----|-----|-----|--------|
| β-VAE + Agglomerative | 0.256 | 1.554 | **0.954** | **0.920** | **0.988** |
| β-VAE + KMeans | 0.263 | 1.501 | 0.774 | 0.728 | 0.940 |
| PCA + KMeans | 0.296 | 1.377 | 0.780 | 0.734 | 0.942 |
| MFCC + KMeans | 0.320 | 1.165 | ~0 | ~0 | 0.516 |

👉 **Key insight:** High silhouette does not always mean meaningful clustering.  
MFCC shows good separation but fails to align with labels.

---

## 🧠 Project Structure
.
├── datasets/
├── features/
├── plots/
├── results/
├── scripts/
│ ├── easy_tasks/
│ ├── medium_tasks/
│ └── hard_tasks/
├── requirements.txt
└── README.md


---

## ⚙️ Installation

```bash
git clone <your-repo-url>
cd mmc-vae
pip install -r requirements.txt


## 🧪 How to Run
### 🔹 Easy Tasks (Text-based VAE)
python scripts/easy_tasks/1.feature_engineering.py
python scripts/easy_tasks/2.vae_training.py
python scripts/easy_tasks/3.clustering.py
python scripts/easy_tasks/4.Evaluation_and_visualization.py

### 🔹 Medium Tasks (Audio + Text, CNN-VAE)
python scripts/medium_tasks/1.download_audio.py
python scripts/medium_tasks/2.extract_mmcc.py
python scripts/medium_tasks/3.generate_lyrics_embedding.py
python scripts/medium_tasks/4.prepare_multimodal_features.py
python scripts/medium_tasks/5.train_cnn_vae.py
python scripts/medium_tasks/6.fusion_and_clustering.py
python scripts/medium_tasks/7.clustering_cnn.py

### 🔹 Hard Tasks (β-VAE + Multimodal)
python scripts/hard_tasks/1.train_beta_vae.py
python scripts/hard_tasks/2.extract_and_fuse.py
python scripts/hard_tasks/3.clustering_and_matrics.py
python scripts/hard_tasks/4.visualisation.py



