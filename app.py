# =========================================
# app.py
# HyperClusterAI
# =========================================

import io
import importlib.util
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy.io import loadmat
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

try:
    import psutil
except ImportError:
    psutil = None

from clustering.kmeans_clustering import run_kmeans
from clustering.pca_clustering import run_pca_kmeans
from clustering.spatial_clustering import run_spatial
from utils.metrics import calculate_metrics
from utils.preprocessing import detect_image_type


# =========================================
# EXPORT HELPERS
# =========================================

def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf


def create_metrics_csv(results_dict):
    df = pd.DataFrame(results_dict).T
    return df.to_csv().encode("utf-8")


# =========================================
# IMAGE HELPERS
# =========================================

def load_mat_array(file_obj):
    data = loadmat(file_obj)
    possible_keys = [k for k in data.keys() if not k.startswith("__")]

    if not possible_keys:
        st.error("No valid array found in the MAT file.")
        st.stop()

    return data[possible_keys[0]]


def ensure_three_dimensional(image):
    if image.ndim == 2:
        return np.stack([image] * 3, axis=-1)

    if image.ndim != 3:
        st.error("Uploaded image must be a 2D grayscale image or a 3D image cube.")
        st.stop()

    return image


def normalize_for_display(image):
    image = image.astype(np.float32)
    image_min = image.min()
    image_max = image.max()

    if image_max == image_min:
        return np.zeros_like(image)

    return (image - image_min) / (image_max - image_min)


def make_rgb_preview(image, image_type):
    if image_type == "RGB":
        return normalize_for_display(image[:, :, :3])

    bands = image.shape[2]
    band_indices = [
        min(10, bands - 1),
        min(20, bands - 1),
        min(30, bands - 1),
    ]

    return normalize_for_display(image[:, :, band_indices])


def make_pca_preview(reduced, height, width):
    preview = reduced[:, :3]

    if preview.shape[1] < 3:
        padding = np.zeros((preview.shape[0], 3 - preview.shape[1]))
        preview = np.hstack([preview, padding])

    preview = normalize_for_display(preview)
    return preview.reshape(height, width, 3)


def parse_band_indices(raw_indices, band_count):
    indices = []

    for item in raw_indices.split(","):
        item = item.strip()

        if item.isdigit():
            indices.append(int(item))

    if not indices:
        indices = [min(10, band_count - 1), min(20, band_count - 1), min(30, band_count - 1)]

    while len(indices) < 3:
        indices.append(indices[-1])

    return [max(0, min(index, band_count - 1)) for index in indices[:3]]


def run_single_band_kmeans(image, band_index, n_clusters):
    selected_band = image[:, :, band_index].reshape(-1, 1)
    scaled = StandardScaler().fit_transform(selected_band)
    labels = run_kmeans(scaled, n_clusters)

    return labels, scaled


def run_false_color_kmeans(image, band_indices, n_clusters):
    false_color = image[:, :, band_indices].reshape(-1, len(band_indices))
    scaled = StandardScaler().fit_transform(false_color)
    labels = run_kmeans(scaled, n_clusters)

    return labels, scaled


def plot_spectral_signatures(image, cluster_map, max_clusters=8):
    bands = np.arange(image.shape[2])
    fig, ax = plt.subplots(figsize=(9, 4))

    for cluster_id in np.unique(cluster_map)[:max_clusters]:
        mask = cluster_map == cluster_id

        if np.any(mask):
            signature = image[mask].mean(axis=0)
            ax.plot(bands, signature, label=f"Cluster {cluster_id}")

    ax.set_title("Mean Spectral Signature by Cluster")
    ax.set_xlabel("Band Index")
    ax.set_ylabel("Mean Reflectance / Intensity")
    ax.legend(loc="best", fontsize=8)

    return fig


def plot_tsne_features(features, labels, max_samples=2000):
    if features is None or features.shape[1] < 2:
        return None

    sample_count = min(max_samples, features.shape[0])

    if sample_count < 6:
        return None

    sample_indices = np.linspace(0, features.shape[0] - 1, sample_count, dtype=int)
    sampled_features = features[sample_indices]
    sampled_labels = labels[sample_indices]
    perplexity = max(5, min(30, sample_count - 1))

    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42,
    ).fit_transform(sampled_features)

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=sampled_labels,
        cmap="nipy_spectral",
        s=8,
        alpha=0.8,
    )
    ax.set_title("t-SNE Feature Space Visualization")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.colorbar(scatter, ax=ax, label="Cluster")

    return fig


def calculate_silhouette(features, labels, max_samples=3000):
    if features is None or len(np.unique(labels)) < 2:
        return None

    sample_count = min(max_samples, features.shape[0])

    if sample_count < 3:
        return None

    sample_indices = np.linspace(0, features.shape[0] - 1, sample_count, dtype=int)

    return silhouette_score(
        features[sample_indices],
        labels[sample_indices],
    )


METHOD_INFO = {
    "Single Band KMeans": {
        "Best For": "Single-wavelength baseline",
        "Strength": "Very fast and easy to interpret",
        "Limitation": "Uses only one spectral band",
        "Runtime": "Very fast",
    },
    "False Color KMeans": {
        "Best For": "Three-band HSI visualization baseline",
        "Strength": "Uses more spectral information than grayscale",
        "Limitation": "Still discards most HSI bands",
        "Runtime": "Fast",
    },
    "KMeans": {
        "Best For": "Fast baseline segmentation",
        "Strength": "Simple, quick, and easy to compare",
        "Limitation": "Uses spectral values without spatial context",
        "Runtime": "Fast",
    },
    "PCA + KMeans": {
        "Best For": "High-dimensional RGB/HSI data",
        "Strength": "Reduces spectral noise and dimensionality",
        "Limitation": "Linear reduction may miss nonlinear patterns",
        "Runtime": "Fast",
    },
    "Spatial-Spectral": {
        "Best For": "Context-aware segmentation",
        "Strength": "Uses neighboring pixels through patch features",
        "Limitation": "Patch extraction increases memory use",
        "Runtime": "Medium",
    },
    "Autoencoder": {
        "Best For": "Nonlinear feature learning",
        "Strength": "Learns compact latent representations",
        "Limitation": "Retrains per input image and needs more time",
        "Runtime": "Slow",
    },
    "CNN Autoencoder": {
        "Best For": "Spatial-spectral deep feature learning",
        "Strength": "Preserves local patch structure with convolutions",
        "Limitation": "Slower and more memory intensive than dense AE",
        "Runtime": "Very slow",
    },
    "DEC": {
        "Best For": "Research-style deep clustering",
        "Strength": "Refines latent clusters with target distribution learning",
        "Limitation": "Most computationally expensive method",
        "Runtime": "Slowest",
    },
}


METHOD_PIPELINES = {
    "Single Band KMeans": [
        "Image upload",
        "Band selection",
        "Single-band feature vector",
        "Standard scaling",
        "KMeans clustering",
        "Cluster map",
    ],
    "False Color KMeans": [
        "Image upload",
        "Three-band selection",
        "False-color feature vector",
        "Standard scaling",
        "KMeans clustering",
        "Cluster map",
    ],
    "KMeans": [
        "Image upload",
        "RGB/HSI detection",
        "Pixel flattening",
        "Standard scaling",
        "KMeans clustering",
        "Cluster map",
    ],
    "PCA + KMeans": [
        "Image upload",
        "RGB/HSI detection",
        "Pixel flattening",
        "Standard scaling",
        "PCA reduction",
        "KMeans clustering",
        "Cluster map",
    ],
    "Spatial-Spectral": [
        "Image upload",
        "RGB/HSI detection",
        "Patch extraction",
        "Patch scaling",
        "PCA reduction",
        "KMeans clustering",
        "Cluster map",
    ],
    "Autoencoder": [
        "Image upload",
        "RGB/HSI detection",
        "Patch extraction",
        "Patch scaling",
        "PCA reduction",
        "Dense autoencoder training",
        "Latent embedding",
        "KMeans clustering",
        "Cluster map",
    ],
    "CNN Autoencoder": [
        "Image upload",
        "RGB/HSI detection",
        "Pixel scaling",
        "PCA band reduction",
        "Patch cube extraction",
        "CNN autoencoder training",
        "Latent embedding",
        "KMeans clustering",
        "Cluster map",
    ],
    "DEC": [
        "Image upload",
        "RGB/HSI detection",
        "Patch extraction",
        "Patch scaling",
        "PCA reduction",
        "Dense autoencoder pretraining",
        "Latent embedding",
        "DEC refinement",
        "Final KMeans clustering",
        "Cluster map",
    ],
}


DEEP_METHODS = {"Autoencoder", "CNN Autoencoder", "DEC"}


def tensorflow_available():
    return importlib.util.find_spec("tensorflow") is not None


def get_compute_backend(load_tensorflow=False):
    if not tensorflow_available():
        return "TensorFlow unavailable"

    if not load_tensorflow:
        return "TensorFlow available"

    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        return f"GPU ({len(gpus)} detected)"

    return "CPU"


def format_pipeline(method_name):
    return " -> ".join(METHOD_PIPELINES[method_name])


def run_selected_method(
    method_name,
    image,
    pixels_scaled,
    n_clusters,
    patch_size,
    pca_components,
    latent_dim,
    single_band_index,
    false_color_indices,
):
    if method_name == "Single Band KMeans":
        return run_single_band_kmeans(image, single_band_index, n_clusters)

    if method_name == "False Color KMeans":
        return run_false_color_kmeans(image, false_color_indices, n_clusters)

    if method_name == "KMeans":
        return run_kmeans(pixels_scaled, n_clusters), pixels_scaled

    if method_name == "PCA + KMeans":
        return run_pca_kmeans(pixels_scaled, n_clusters, pca_components)

    if method_name == "Spatial-Spectral":
        return run_spatial(image, patch_size, pca_components, n_clusters)

    if method_name == "Autoencoder":
        from clustering.autoencoder_clustering import run_autoencoder

        return run_autoencoder(
            image,
            patch_size,
            pca_components,
            latent_dim,
            n_clusters,
        )

    if method_name == "CNN Autoencoder":
        from clustering.cnn_autoencoder_clustering import run_cnn_autoencoder

        return run_cnn_autoencoder(
            image,
            patch_size,
            pca_components,
            latent_dim,
            n_clusters,
        )

    if method_name == "DEC":
        from clustering.dec_clustering import run_dec

        return run_dec(
            image,
            patch_size,
            pca_components,
            latent_dim,
            n_clusters,
        )

    st.error(f"Unknown method: {method_name}")
    st.stop()


# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(page_title="HyperClusterAI", layout="wide")

st.markdown(
    """
    <style>
    .main {
        background-color: #0E1117;
    }

    h1, h2, h3 {
        color: #4CAF50;
    }

    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("HyperClusterAI")
st.subheader("Universal Spatial-Spectral Intelligent Segmentation System")

# =====================================
# DASHBOARD TABS
# =====================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
    [
        "Original Image",
        "Feature View",
        "Cluster Results",
        "Metrics",
        "Comparison",
        "Spectral Signatures",
        "Latent Space",
        "Exports",
        "Performance",
    ]
)


# =========================================
# SIDEBAR
# =========================================

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/2909/2909763.png",
    width=120,
)
st.sidebar.title("HyperClusterAI")
st.sidebar.markdown(
    """
    **Workflow**

    1. Upload data
    2. Select method
    3. Configure parameters
    4. Run clustering
    5. Analyze results
    6. Export outputs
    """
)
st.sidebar.markdown("---")
st.sidebar.header("Settings")

mode = st.sidebar.radio(
    "Mode",
    [
        "Single Method",
        "Compare Methods",
    ],
)

method_options = [
    "Single Band KMeans",
    "False Color KMeans",
    "KMeans",
    "PCA + KMeans",
    "Spatial-Spectral",
    "Autoencoder",
    "CNN Autoencoder",
    "DEC",
]

if mode == "Single Method":
    method = st.sidebar.selectbox("Select Clustering Method", method_options)
    selected_methods = [method]
else:
    compare_methods = st.sidebar.multiselect(
        "Select Methods",
        method_options,
        default=["KMeans", "DEC"],
    )
    selected_methods = compare_methods

n_clusters = st.sidebar.slider("Number of Clusters", 2, 20, 5)
patch_size = st.sidebar.slider("Patch Size", 3, 9, 3, step=2)
latent_dim = st.sidebar.slider("Latent Dimension", 2, 50, 10)
pca_components = st.sidebar.slider("PCA Components", 2, 50, 10)
single_band_index = st.sidebar.number_input("Single Band Index", min_value=0, value=20)
false_color_bands = st.sidebar.text_input("False Color Bands", value="20,50,100")
run_button = st.sidebar.button("Run Clustering")

st.sidebar.markdown("---")
deep_method_selected = any(selected_method in DEEP_METHODS for selected_method in selected_methods)
st.sidebar.metric("Compute Backend", get_compute_backend(load_tensorflow=False))

if deep_method_selected:
    st.sidebar.caption("TensorFlow loads only when a deep method is run.")

st.sidebar.markdown("---")
st.sidebar.info(
    """
    HyperClusterAI

    Universal Spatial-Spectral
    Deep Clustering Platform

    Built for:
    - RGB Images
    - Hyperspectral Images
    - Research
    - Intelligent Segmentation
    """
)


# =========================================
# METHOD GUIDE
# =========================================

st.subheader("Method Guide")
st.dataframe(
    pd.DataFrame(METHOD_INFO).T,
    use_container_width=True,
)

if mode == "Single Method":
    st.info(f"Selected pipeline: {format_pipeline(method)}")
else:
    selected_for_summary = compare_methods if compare_methods else method_options

    for method_name in selected_for_summary:
        st.info(f"{method_name}: {format_pipeline(method_name)}")


# =========================================
# FILE UPLOAD
# =========================================

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "png", "jpeg", "npy", "mat"],
)

gt_file = st.file_uploader(
    "Upload Ground Truth (Optional)",
    type=["npy", "mat"],
)


# =========================================
# MAIN PIPELINE
# =========================================

start_time = time.time()

if uploaded_file is not None and run_button:
    file_name = uploaded_file.name.lower()
    process = psutil.Process(os.getpid()) if psutil is not None else None
    memory_before = process.memory_info().rss / 1024**2 if process is not None else None

    if file_name.endswith(("jpg", "png", "jpeg")):
        image = np.array(Image.open(uploaded_file))
        image = ensure_three_dimensional(image)
        image_type = "RGB"
    elif file_name.endswith("npy"):
        image = np.load(uploaded_file)
        image = ensure_three_dimensional(image)
        image_type = detect_image_type(image)
    elif file_name.endswith("mat"):
        image = load_mat_array(uploaded_file)
        image = ensure_three_dimensional(image)
        image_type = detect_image_type(image)
    else:
        st.error("Unsupported file type.")
        st.stop()

    gt = None

    if gt_file is not None:
        gt_name = gt_file.name.lower()

        if gt_name.endswith("npy"):
            gt = np.load(gt_file)
        elif gt_name.endswith("mat"):
            gt = load_mat_array(gt_file)

    st.success(f"Detected Image Type: {image_type}")
    st.write(f"Image Shape: {image.shape}")

    h, w, c = image.shape
    selected_single_band = max(0, min(int(single_band_index), c - 1))
    selected_false_color_bands = parse_band_indices(false_color_bands, c)

    # =====================================
    # AI RECOMMENDATION ENGINE
    # =====================================

    st.subheader("Intelligent Analysis")
    st.markdown("**Pipeline Summary**")

    if mode == "Single Method":
        st.info(format_pipeline(method))
    else:
        for method_name in compare_methods:
            st.info(f"{method_name}: {format_pipeline(method_name)}")

    recommendations = []

    if image_type == "HSI":
        recommendations.append("Hyperspectral image detected.")
        recommendations.append("Recommended Method: DEC, CNN Autoencoder, or Autoencoder.")
        recommendations.append("Spatial-spectral learning is highly beneficial for HSI.")

        if c > 100:
            recommendations.append("High spectral dimensionality detected.")
            recommendations.append("Recommended PCA Components: 10-30.")

    elif image_type == "RGB":
        recommendations.append("RGB image detected.")
        recommendations.append("Recommended Method: PCA + KMeans.")
        recommendations.append("Deep clustering may improve texture-based segmentation.")

    if h * w > 300000:
        recommendations.append("Large image detected.")
        recommendations.append("Use smaller patch sizes for faster computation.")
        recommendations.append("KMeans or PCA + KMeans recommended for efficiency.")

    if patch_size >= 7:
        recommendations.append("Large patch size selected.")
        recommendations.append("May increase memory usage and processing time.")

    for recommendation in recommendations:
        st.info(recommendation)

    if h * w > 300000:
        st.warning("Large image detected. Processing may be slow.")

    with tab1:
        st.subheader("Original Image")

        display_image = make_rgb_preview(image, image_type)
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        ax1.imshow(display_image)
        ax1.set_title("Input Image")
        ax1.axis("off")
        st.pyplot(fig1)

    pixels = image.reshape(-1, c)
    scaler = StandardScaler()
    pixels_scaled = scaler.fit_transform(pixels)

    if mode == "Single Method":
        labels, reduced = run_selected_method(
            method,
            image,
            pixels_scaled,
            n_clusters,
            patch_size,
            pca_components,
            latent_dim,
            selected_single_band,
            selected_false_color_bands,
        )

        if reduced is not None:
            with tab2:
                st.subheader("Feature Visualization")

                pca_vis = make_pca_preview(reduced, h, w)
                fig2, ax2 = plt.subplots(figsize=(6, 6))
                ax2.imshow(pca_vis)
                ax2.set_title("Feature Representation")
                ax2.axis("off")
                st.pyplot(fig2)
        else:
            with tab2:
                st.info("Feature visualization is available for methods that return feature vectors.")

        cluster_map = labels.reshape(h, w)

        with tab3:
            st.subheader("Clustered Output")

            fig3, ax3 = plt.subplots(figsize=(6, 6))
            ax3.imshow(cluster_map, cmap="nipy_spectral")
            ax3.set_title(f"{method} Cluster Map")
            ax3.axis("off")
            st.pyplot(fig3)

        with tab6:
            st.subheader("Spectral Signature Curves")

            if c > 1:
                fig_signature = plot_spectral_signatures(image, cluster_map)
                st.pyplot(fig_signature)
            else:
                st.info("Spectral signatures require at least two image bands.")

        with tab7:
            st.subheader("Latent / Feature Space Visualization")
            fig_tsne = plot_tsne_features(reduced, labels)

            if fig_tsne is not None:
                st.pyplot(fig_tsne)
            else:
                st.info("t-SNE visualization requires at least two feature dimensions.")

        metrics = None
        fig_cm = None
        fig_acc = None
        silhouette = calculate_silhouette(reduced, labels)

        if gt is not None:
            with tab4:
                st.subheader("Evaluation Metrics")

                acc, kappa, nmi, cm = calculate_metrics(gt, labels)
                metrics = {
                    "Accuracy": acc,
                    "Kappa": kappa,
                    "NMI": nmi,
                }

                if silhouette is not None:
                    metrics["Silhouette"] = silhouette

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{acc:.4f}")
                col2.metric("Kappa", f"{kappa:.4f}")
                col3.metric("NMI", f"{nmi:.4f}")
                col4.metric(
                    "Silhouette",
                    f"{silhouette:.4f}" if silhouette is not None else "N/A",
                )

                st.subheader("Confusion Matrix")

                fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                ax_cm.imshow(cm)
                ax_cm.set_title("Confusion Matrix")
                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("True")
                st.pyplot(fig_cm)

                st.subheader("Class-wise Accuracy")

                class_acc = np.divide(
                    cm.diagonal(),
                    cm.sum(axis=1),
                    out=np.zeros_like(cm.diagonal(), dtype=float),
                    where=cm.sum(axis=1) != 0,
                )

                fig_acc, ax_acc = plt.subplots(figsize=(8, 4))
                ax_acc.bar(range(len(class_acc)), class_acc)
                ax_acc.set_title("Class-wise Accuracy")
                ax_acc.set_xlabel("Class")
                ax_acc.set_ylabel("Accuracy")
                st.pyplot(fig_acc)
        else:
            with tab4:
                st.info("Upload ground truth to enable Accuracy, Kappa, and NMI.")

                if silhouette is not None:
                    st.metric("Silhouette Score", f"{silhouette:.4f}")
                else:
                    st.info("Silhouette Score could not be computed for this result.")

        with tab8:
            st.download_button(
                label="Download Cluster Map PNG",
                data=fig_to_png(fig3),
                file_name=f"{method}_cluster_map.png",
                mime="image/png",
            )

            if metrics is not None:
                csv_data = create_metrics_csv({method: metrics})

                st.download_button(
                    label="Download Metrics CSV",
                    data=csv_data,
                    file_name=f"{method}_metrics.csv",
                    mime="text/csv",
                )

                st.download_button(
                    label="Download Confusion Matrix PNG",
                    data=fig_to_png(fig_cm),
                    file_name=f"{method}_confusion_matrix.png",
                    mime="image/png",
                )

                st.download_button(
                    label="Download Class Accuracy Graph",
                    data=fig_to_png(fig_acc),
                    file_name=f"{method}_class_accuracy.png",
                    mime="image/png",
                )

                report_text = f"""
HyperClusterAI Report Summary
====================================

Method Used:
{method}

Image Type:
{image_type}

Number of Clusters:
{n_clusters}

Patch Size:
{patch_size}

PCA Components:
{pca_components}

Latent Dimension:
{latent_dim}

Evaluation Metrics
------------------------------------

Accuracy: {metrics['Accuracy']:.4f}
Kappa: {metrics['Kappa']:.4f}
NMI: {metrics['NMI']:.4f}
Silhouette: {f"{metrics['Silhouette']:.4f}" if 'Silhouette' in metrics else "N/A"}

Observations
------------------------------------

The clustering method segmented the image into regions based on the
selected spatial and spectral features.

Higher NMI indicates stronger clustering consistency. Kappa reflects
agreement beyond chance.
"""

                st.download_button(
                    label="Download Report Summary",
                    data=report_text,
                    file_name=f"{method}_report.txt",
                    mime="text/plain",
                )
            else:
                st.info("Upload ground truth to enable metrics exports and report downloads.")

    else:
        if not compare_methods:
            with tab5:
                st.warning("Select at least one method to compare.")
            st.stop()

        results = {}
        fig_bar = None

        with tab5:
            st.subheader("Method Comparison Dashboard")

            for current_method in compare_methods:
                st.write(f"Running: {current_method}")

                labels, features = run_selected_method(
                    current_method,
                    image,
                    pixels_scaled,
                    n_clusters,
                    patch_size,
                    pca_components,
                    latent_dim,
                    selected_single_band,
                    selected_false_color_bands,
                )

                cluster_map = labels.reshape(h, w)

                fig_compare, ax_compare = plt.subplots(figsize=(5, 5))
                ax_compare.imshow(cluster_map, cmap="nipy_spectral")
                ax_compare.set_title(current_method)
                ax_compare.axis("off")
                st.pyplot(fig_compare)

                if gt is not None:
                    acc, kappa, nmi, _ = calculate_metrics(gt, labels)
                    results[current_method] = {
                        "Accuracy": acc,
                        "Kappa": kappa,
                        "NMI": nmi,
                    }

                    silhouette = calculate_silhouette(features, labels)

                    if silhouette is not None:
                        results[current_method]["Silhouette"] = silhouette
                else:
                    silhouette = calculate_silhouette(features, labels)

                    if silhouette is not None:
                        results[current_method] = {
                            "Silhouette": silhouette,
                        }

            if results:
                st.subheader("Comparison Table")

                results_df = pd.DataFrame(results).T
                st.dataframe(results_df)

                methods = list(results.keys())
                metric_names = list(results_df.columns)
                x = np.arange(len(methods))
                width = 0.8 / max(len(metric_names), 1)

                fig_bar, ax_bar = plt.subplots(figsize=(10, 5))

                for metric_index, metric_name in enumerate(metric_names):
                    offset = (metric_index - (len(metric_names) - 1) / 2) * width
                    values = [results[m].get(metric_name, 0) for m in methods]
                    ax_bar.bar(x + offset, values, width, label=metric_name)

                ax_bar.set_xticks(x)
                ax_bar.set_xticklabels(methods, rotation=20, ha="right")
                ax_bar.set_title("Method Comparison")
                ax_bar.legend()
                st.pyplot(fig_bar)

                if gt is None:
                    st.info("Ground truth was not uploaded, so comparison uses Silhouette only.")
            else:
                st.info("Comparison metrics could not be computed for the selected methods.")

        if results:
            ranking_metric = "NMI" if gt is not None else "Silhouette"
            best_method = max(results, key=lambda x: results[x].get(ranking_metric, -1))

            comparison_report = f"""
HyperClusterAI Comparison Report
====================================

Compared Methods:
{methods}

Best Performing Method:
{best_method} based on {ranking_metric}

Detailed Results
------------------------------------
"""

            for method_name, method_metrics in results.items():
                comparison_report += f"""
Method: {method_name}

Accuracy: {method_metrics.get('Accuracy', 'N/A')}
Kappa: {method_metrics.get('Kappa', 'N/A')}
NMI: {method_metrics.get('NMI', 'N/A')}
Silhouette: {method_metrics.get('Silhouette', 'N/A')}

------------------------------------
"""

            comparison_report += f"""

Final Observation
====================================

{best_method} achieved the highest {ranking_metric} score among the compared methods.
"""

            with tab6:
                st.info("Spectral signature plots are available in Single Method mode.")

            with tab7:
                st.info("t-SNE latent-space visualization is available in Single Method mode.")

            with tab8:
                st.download_button(
                    label="Download Comparison Graph",
                    data=fig_to_png(fig_bar),
                    file_name="comparison_graph.png",
                    mime="image/png",
                )

                st.download_button(
                    label="Download Comparison Table CSV",
                    data=create_metrics_csv(results),
                    file_name="comparison_metrics.csv",
                    mime="text/csv",
                )

                st.download_button(
                    label="Download Comparison Report",
                    data=comparison_report,
                    file_name="comparison_report.txt",
                    mime="text/plain",
                )
        elif gt is None:
            with tab8:
                st.info("Upload ground truth to enable comparison exports.")

    # =====================================
    # PERFORMANCE DASHBOARD
    # =====================================

    end_time = time.time()
    runtime = end_time - start_time
    memory_after = process.memory_info().rss / 1024**2 if process is not None else None
    memory_used = (
        memory_after - memory_before
        if memory_after is not None and memory_before is not None
        else None
    )
    memory_display = f"{memory_used:.2f}" if memory_used is not None else "Install psutil"

    complexity = {
        "KMeans": "Low",
        "PCA + KMeans": "Low-Medium",
        "Spatial-Spectral": "Medium",
        "Autoencoder": "High",
        "CNN Autoencoder": "Very High",
        "DEC": "Very High",
    }

    if mode == "Single Method":
        processing_mode = (
            "Deep Clustering"
            if method in DEEP_METHODS
            else "Classical Clustering"
        )
        method_complexity = complexity.get(method, "Unknown")
    else:
        processing_mode = (
            "Hybrid Comparison"
            if any(selected_method in DEEP_METHODS for selected_method in compare_methods)
            else "Classical Comparison"
        )
        method_complexity = ", ".join(
            f"{selected_method}: {complexity.get(selected_method, 'Unknown')}"
            for selected_method in compare_methods
        )

    with tab9:
        st.subheader("Performance Dashboard")

        col1, col2, col3 = st.columns(3)
        col1.metric("Runtime (sec)", f"{runtime:.2f}")
        col2.metric("Memory Usage (MB)", memory_display)
        col3.metric("Image Size", f"{h} x {w}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Processing Mode", processing_mode)
        col5.metric("Method Complexity", method_complexity)
        col6.metric("Dataset Type", image_type)

        st.metric("Compute Backend", get_compute_backend(load_tensorflow=deep_method_selected))
        st.info(f"Estimated Computational Complexity: {method_complexity}")


# =========================================
# FOOTER
# =========================================

st.markdown("---")
st.markdown("HyperClusterAI (c) Universal Spatial-Spectral Intelligent Segmentation System")
