# Dataset Layout

The full materials-science figure dataset (~51,000 images, ~13 GB) is **not committed to GitHub** due to size limits. Place your local copy under the paths below before running the notebooks.

## Directory structure

| Path | Description |
|------|-------------|
| `data/raw/` | Original scientific figures from publications |
| `data/resized/` | 224×224 padded images for EfficientNet preprocessing |
| `data/cleaned/` | Images after corruption filtering |
| `data/duplicates_removed/` | Duplicate figures removed during cleaning |
| `data/crop_alloy/` | Cropped alloy figure subset |
| `data/cluster0_images/` | Subset of cluster-0 plot/chart images (~26k) for sub-clustering |
| `data/metadata/metadata.csv` | Image metadata (paths, captions, categories) |
| `data/balanced_subset.csv` | Labeled evaluation subset (18 categories, 138 visualization subtypes) |

## Included in the repository

- `metadata/metadata.csv` — full dataset metadata
- `balanced_subset.csv` — labeled benchmark subset used for KNN, XGBoost, ARI/NMI evaluation

## Regenerating processed data

Run the numbered notebooks in order:

1. `01_dataset_exploration.ipynb` — corruption & duplicate detection
2. `02_resolution_analysis.ipynb` — resolution statistics
3. `03_resize_images.ipynb` — padded resize to 224×224
4. `04_resize_validation.ipynb` — visual QA of resizing
