# Deep Learning-Based Scientific Figure Clustering for Materials Informatics

An end-to-end pipeline for understanding, clustering, and retrieving scientific figures from materials-science publications using modern vision foundation models.

**Repository:** [Robin-Chetry/Deep-Learning-Based-Scientific-Figure-Clustering-for-Materials-Informatics](https://github.com/Robin-Chetry/Deep-Learning-Based-Scientific-Figure-Clustering-for-Materials-Informatics)

---

## Overview

Materials-science publications contain diverse figure types — SEM micrographs, TEM images, EBSD maps, elemental mappings, diffraction patterns, spectroscopy plots, mechanical testing graphs, schematics, and multi-panel figures. Manually organizing tens of thousands of these images is impractical.

This project builds an AI system to:

1. **Understand** scientific figures via deep embeddings
2. **Discover** visually similar image groups without labels
3. **Organize** large scientific image collections
4. **Enable** similarity-based scientific image retrieval

**Long-term goal:** upload a query image and retrieve visually similar figures from a large scientific image database.

---

## Current Progress

| Phase | Status | Summary |
|-------|--------|---------|
| Dataset preparation | **Complete** | ~51k images cleaned, deduplicated, resized with aspect-ratio padding |
| Phase 1 — EfficientNet-B0 | **Complete** | 1280-d embeddings → PCA → UMAP → HDBSCAN clustering |
| Phase 2 — DINO | **Complete** | Self-supervised ViT embeddings; improved ARI/NMI over EfficientNet |
| Phase 3 — CLIP | **Explored** | Evaluated on cluster-0 sub-clustering; transitioned to SigLIP |
| Phase 4 — SigLIP | **Primary model** | Best embedding quality for classification & retrieval |
| Supervised evaluation | **Complete** | KNN ~60%, XGBoost ~74% on 18-category benchmark |
| Clustering evaluation | **Complete** | ARI ≈ 0.21, NMI ≈ 0.36 on scientific labels |
| Visualization subtypes | **Complete** | 138 subtypes; ARI ≈ 0.18, NMI ≈ 0.50 |
| Similarity retrieval | **In progress** | Retrieval-oriented workflow identified as most practical path |

### Key finding

Clustering separates images by **visual appearance**, not always by **scientific methodology**. Labels like *Raman Spectrum*, *Confusion Matrix*, and *Flowchart* form highly pure clusters, while labels like *DFT Result*, *Phase-Field Simulation*, and *MD Trajectory* contain many visually different image types. **Similarity retrieval** is therefore more practical than strict category classification for heterogeneous scientific datasets.

---

## Results at a Glance

| Metric | Value | Context |
|--------|-------|---------|
| KNN classification accuracy | ~60% | 18 scientific figure categories |
| XGBoost classification accuracy | ~74% | SigLIP embeddings + downstream classifier |
| Clustering ARI | ≈ 0.21 | Embeddings vs. scientific labels |
| Clustering NMI | ≈ 0.36 | Embeddings vs. scientific labels |
| Subtype clustering NMI | ≈ 0.50 | 138 visualization subtypes |
| High-purity subtypes | near-perfect | SHAP Plot, Confusion Matrix, Raman Spectrum, Radar Chart |

---

## Project Structure

```
.
├── notebooks/          # End-to-end analysis notebooks (numbered pipeline)
├── src/                # Source modules (feature extraction utilities)
├── data/
│   ├── metadata/       # Dataset metadata (committed)
│   ├── balanced_subset.csv   # Labeled evaluation subset (committed)
│   └── README.md       # Data layout & download instructions
├── embeddings/         # Saved feature vectors (.npy)
├── outputs/            # Cluster reports, plots, labelled CSV exports
├── requirements.txt
└── README.md
```

> **Note:** Raw images (~13 GB) and the largest embedding file (`efficientnet_b0_features.npy`, 244 MB) are excluded from Git. See [data/README.md](data/README.md) for local setup.

---

## Notebooks

### Phase 1 — Data preparation & EfficientNet pipeline

| Notebook | Description |
|----------|-------------|
| `01_dataset_exploration.ipynb` | Corruption detection, duplicate removal, dataset statistics |
| `02_resolution_analysis.ipynb` | Resolution and aspect-ratio distribution analysis |
| `03_resize_images.ipynb` | Batch resize to 224×224 with padding |
| `04_resize_validation.ipynb` | Visual validation of preprocessing |
| `05_efficientnet_feature_extraction.ipynb` | EfficientNet-B0 embedding extraction (1280-d) |
| `06_pca_analysis.ipynb` | PCA dimensionality reduction |
| `07_umap_visualization.ipynb` | Static 2D UMAP visualization |
| `08_hdbscan_clustering.ipynb` | HDBSCAN density-based clustering |
| `09_cluster_sample_grids.ipynb` | Per-cluster image grid visualization |
| `10_interactive_umap.ipynb` | Interactive Plotly UMAP explorer |

### Phase 2 — DINO clustering & evaluation

| Notebook | Description |
|----------|-------------|
| `11_dino_cluster_evaluation.ipynb` | Unsupervised cluster quality metrics (silhouette, DB, CH) |
| `12_dino_cluster_inspection.ipynb` | Manual cluster inspection with image grids |
| `13_apply_cluster_labels.ipynb` | Map cluster IDs to human-readable labels |
| `14_cluster0_kmeans_subclustering.ipynb` | K-Means sub-clustering of cluster 0 (plots & charts) |
| `15_dino_pca_umap_hdbscan.ipynb` | End-to-end DINO → PCA → UMAP → HDBSCAN pipeline |
| `16_optuna_clustering_tuning.ipynb` | Hyperparameter search for clustering (Optuna) |

### Phase 3 & 4 — CLIP, SigLIP, supervised learning (Kaggle exports)

| Notebook | Description |
|----------|-------------|
| `17_clip_cluster0_reclustering.ipynb` | CLIP ViT-L-14 re-clustering of plot/chart cluster |
| `18_dino_supervised_cluster_eval.ipynb` | Supervised ARI/NMI evaluation with labeled subset |
| `19_supervised_contrastive_training.ipynb` | Supervised contrastive embedding fine-tuning |
| `99_scratch_balanced_subset_explore.ipynb` | Scratch notebook for labeled data exploration |

---

## Pipeline Architecture

```
Scientific Image
      ↓
SigLIP Embedding Extraction
      ↓
Nearest Neighbor Search
      ↓
Similarity-Based Retrieval
      ↓
Return Relevant Scientific Figures
```

### Embedding evolution

```
EfficientNet-B0  →  DINO  →  CLIP  →  SigLIP (primary)
     (1280-d)      (ViT)    (V+L)     (best retrieval)
```

### Clustering stack

```
Embeddings → PCA → UMAP → HDBSCAN / K-Means
                              ↓
                    ARI / NMI / Cluster Purity
```

---

## Technologies

Python · PyTorch · Transformers · SigLIP · DINO · CLIP · EfficientNet · Scikit-learn · XGBoost · PCA · UMAP · HDBSCAN · Optuna · Plotly · Pandas · NumPy

---

## Setup

```bash
# Clone the repository
git clone https://github.com/Robin-Chetry/Deep-Learning-Based-Scientific-Figure-Clustering-for-Materials-Informatics.git
cd Deep-Learning-Based-Scientific-Figure-Clustering-for-Materials-Informatics

# Create environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

Place the raw image dataset under `data/raw/` following [data/README.md](data/README.md), then run notebooks starting from `01_dataset_exploration.ipynb`.

---

## Outputs

Key artifacts are saved under `outputs/` and `embeddings/`:

- `outputs/labelled_dataset.csv` — cluster labels mapped to image paths
- `outputs/cluster_quality_report.csv` — per-cluster statistics
- `embeddings/dino_features.npy` — DINO embedding matrix
- `embeddings/pca_features.npy` — PCA-reduced features
- `embeddings/hdbscan_labels.npy` — cluster assignments

---

## Future Work

- [ ] Deploy SigLIP-based similarity retrieval API
- [ ] Build web UI for figure upload and nearest-neighbor search
- [ ] Scale retrieval index to full 50k+ image collection
- [ ] Explore hybrid retrieval (visual + caption/text embeddings)

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Author

Robin Rawat — IIT Roorkee Internship Project (Materials Informatics)
