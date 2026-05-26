# HyperClusterAI

Universal Spatial-Spectral Intelligent Segmentation System for RGB and hyperspectral image clustering.

HyperClusterAI combines classical clustering, dimensionality reduction, spatial-spectral patch features, dense autoencoder embeddings, and Deep Embedded Clustering (DEC) inside an interactive Streamlit dashboard.

## Key Features

- RGB and hyperspectral image support
- `.jpg`, `.png`, `.jpeg`, `.npy`, and `.mat` input support
- Optional ground truth upload for evaluation
- KMeans clustering baseline
- PCA + KMeans clustering
- Spatial-spectral clustering with patch extraction
- Dense autoencoder clustering
- Deep Embedded Clustering (DEC)
- Intelligent method recommendations
- Method comparison dashboard
- Accuracy, Kappa, NMI, confusion matrix, and class-wise accuracy
- Runtime, memory, and CPU/GPU backend display
- Downloadable cluster maps, metrics, graphs, and reports

## System Architecture

```text
Input Image
    |
    v
File Loader (.jpg/.png/.npy/.mat)
    |
    v
RGB / HSI Detection
    |
    v
Preprocessing + Scaling
    |
    +--> KMeans
    |
    +--> PCA Reduction --> KMeans
    |
    +--> Patch Extraction --> PCA Reduction --> KMeans
    |
    +--> Patch Extraction --> PCA Reduction --> Dense Autoencoder --> KMeans
    |
    +--> Patch Extraction --> PCA Reduction --> Dense Autoencoder --> DEC Refinement --> KMeans
    |
    v
Cluster Map
    |
    v
Evaluation Metrics + Visualization + Export
```

## Method Summary

| Method           | Best For                       | Strength                         | Tradeoff                       |
| ---------------- | ------------------------------ | -------------------------------- | ------------------------------ |
| KMeans           | Fast baseline segmentation     | Simple and quick                 | No spatial context             |
| PCA + KMeans     | High-dimensional RGB/HSI data  | Reduces dimensionality and noise | Linear feature reduction       |
| Spatial-Spectral | Context-aware segmentation     | Uses neighborhood patches        | Higher memory usage            |
| Autoencoder      | Nonlinear feature learning     | Learns compact latent embeddings | Retrains per uploaded image    |
| DEC              | Research-style deep clustering | Refines clusters in latent space | Most computationally expensive |

## App Workflow

1. Upload an RGB or hyperspectral image.
2. Optionally upload a ground truth mask.
3. Choose single-method mode or comparison mode.
4. Configure cluster count, patch size, PCA components, and latent dimension.
5. Run clustering.
6. Review original image, PCA preview, cluster map, metrics, comparison results, and performance.
7. Export maps, metrics, graphs, or text reports.

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Supported Inputs

| Input Type   | Supported Extensions    | Notes                                    |
| ------------ | ----------------------- | ---------------------------------------- |
| RGB image    | `.jpg`, `.jpeg`, `.png` | Converted to NumPy array                 |
| NumPy cube   | `.npy`                  | Supports RGB or hyperspectral arrays     |
| MATLAB cube  | `.mat`                  | Loads the first valid non-metadata array |
| Ground truth | `.npy`, `.mat`          | Optional, required only for metrics      |

## Evaluation Metrics

When ground truth is provided, HyperClusterAI computes:

- Accuracy
- Cohen's Kappa
- Normalized Mutual Information (NMI)
- Confusion matrix
- Class-wise accuracy

Cluster labels are aligned with ground truth labels using Hungarian matching before accuracy and Kappa are calculated.

## Project Structure

```text
HyperClusterAI/
|-- app.py
|-- clustering/
|   |-- autoencoder_clustering.py
|   |-- dec_clustering.py
|   |-- kmeans_clustering.py
|   |-- pca_clustering.py
|   `-- spatial_clustering.py
|-- models/
|   `-- autoencoder_model.py
|-- utils/
|   |-- metrics.py
|   |-- patches.py
|   |-- preprocessing.py
|   `-- visualization.py
|-- requirements.txt
`-- README.md
```

## Technologies Used

- Python
- Streamlit
- TensorFlow
- Scikit-learn
- NumPy
- SciPy
- Matplotlib
- Pandas
- Pillow
- psutil

## Recommended Next Polish

- Add dashboard screenshots to this README.
- Add a short demo GIF showing upload, clustering, comparison, and export.
- Add sample datasets or links to public hyperspectral datasets.
- Add optional result caching keyed by image and parameter settings.
- Add spectral signature visualization for selected pixels/classes.
