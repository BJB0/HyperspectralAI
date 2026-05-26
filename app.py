# =========================================
# app.py
# HyperClusterAI
# =========================================

import io
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler

try:
    import psutil
except ImportError:
    psutil = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

from clustering.autoencoder_clustering import run_autoencoder
from clustering.dec_clustering import run_dec
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


METHOD_INFO = {
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
    "DEC": {
        "Best For": "Research-style deep clustering",
        "Strength": "Refines latent clusters with target distribution learning",
        "Limitation": "Most computationally expensive method",
        "Runtime": "Slowest",
    },
}


METHOD_PIPELINES = {
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


def get_compute_backend():
    if tf is None:
        return "TensorFlow unavailable"

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
):
    if method_name == "KMeans":
        return run_kmeans(pixels_scaled, n_clusters), None

    if method_name == "PCA + KMeans":
        return run_pca_kmeans(pixels_scaled, n_clusters, pca_components)

    if method_name == "Spatial-Spectral":
        return run_spatial(image, patch_size, pca_components, n_clusters)

    if method_name == "Autoencoder":
        return run_autoencoder(
            image,
            patch_size,
            pca_components,
            latent_dim,
            n_clusters,
        )

    if method_name == "DEC":
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

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "Original Image",
        "PCA View",
        "Cluster Results",
        "Metrics",
        "Comparison",
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
    "KMeans",
    "PCA + KMeans",
    "Spatial-Spectral",
    "Autoencoder",
    "DEC",
]

if mode == "Single Method":
    method = st.sidebar.selectbox("Select Clustering Method", method_options)
else:
    compare_methods = st.sidebar.multiselect(
        "Select Methods",
        method_options,
        default=["KMeans", "DEC"],
    )

n_clusters = st.sidebar.slider("Number of Clusters", 2, 20, 5)
patch_size = st.sidebar.slider("Patch Size", 3, 9, 3, step=2)
latent_dim = st.sidebar.slider("Latent Dimension", 2, 50, 10)
pca_components = st.sidebar.slider("PCA Components", 2, 50, 10)
run_button = st.sidebar.button("Run Clustering")

st.sidebar.markdown("---")
st.sidebar.metric("Compute Backend", get_compute_backend())
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
        recommendations.append("Recommended Method: DEC or Autoencoder.")
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
        )

        if reduced is not None:
            with tab2:
                st.subheader("PCA Visualization")

                pca_vis = make_pca_preview(reduced, h, w)
                fig2, ax2 = plt.subplots(figsize=(6, 6))
                ax2.imshow(pca_vis)
                ax2.set_title("PCA Representation")
                ax2.axis("off")
                st.pyplot(fig2)
        else:
            with tab2:
                st.info("PCA visualization is available for methods that return reduced features.")

        cluster_map = labels.reshape(h, w)

        with tab3:
            st.subheader("Clustered Output")

            fig3, ax3 = plt.subplots(figsize=(6, 6))
            ax3.imshow(cluster_map, cmap="nipy_spectral")
            ax3.set_title(f"{method} Cluster Map")
            ax3.axis("off")
            st.pyplot(fig3)

        metrics = None
        fig_cm = None
        fig_acc = None

        if gt is not None:
            with tab4:
                st.subheader("Evaluation Metrics")

                acc, kappa, nmi, cm = calculate_metrics(gt, labels)
                metrics = {
                    "Accuracy": acc,
                    "Kappa": kappa,
                    "NMI": nmi,
                }

                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{acc:.4f}")
                col2.metric("Kappa", f"{kappa:.4f}")
                col3.metric("NMI", f"{nmi:.4f}")

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
                st.info("Upload ground truth to enable evaluation metrics.")

        with tab6:
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

                labels, _ = run_selected_method(
                    current_method,
                    image,
                    pixels_scaled,
                    n_clusters,
                    patch_size,
                    pca_components,
                    latent_dim,
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

            if gt is not None and results:
                st.subheader("Comparison Table")

                results_df = pd.DataFrame(results).T
                st.dataframe(results_df)

                methods = list(results.keys())
                accuracy_vals = [results[m]["Accuracy"] for m in methods]
                kappa_vals = [results[m]["Kappa"] for m in methods]
                nmi_vals = [results[m]["NMI"] for m in methods]

                x = np.arange(len(methods))
                width = 0.25

                fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
                ax_bar.bar(x - width, accuracy_vals, width, label="Accuracy")
                ax_bar.bar(x, kappa_vals, width, label="Kappa")
                ax_bar.bar(x + width, nmi_vals, width, label="NMI")
                ax_bar.set_xticks(x)
                ax_bar.set_xticklabels(methods)
                ax_bar.set_title("Method Comparison")
                ax_bar.legend()
                st.pyplot(fig_bar)
            elif gt is None:
                st.info("Upload ground truth to enable comparison metrics and reports.")

        if gt is not None and results:
            best_method = max(results, key=lambda x: results[x]["NMI"])

            comparison_report = f"""
HyperClusterAI Comparison Report
====================================

Compared Methods:
{methods}

Best Performing Method:
{best_method}

Detailed Results
------------------------------------
"""

            for method_name, method_metrics in results.items():
                comparison_report += f"""
Method: {method_name}

Accuracy: {method_metrics['Accuracy']:.4f}
Kappa: {method_metrics['Kappa']:.4f}
NMI: {method_metrics['NMI']:.4f}

------------------------------------
"""

            comparison_report += f"""

Final Observation
====================================

{best_method} achieved the highest NMI score among the compared methods.
"""

            with tab6:
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
            with tab6:
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
        "DEC": "Very High",
    }

    if mode == "Single Method":
        processing_mode = "Deep Clustering" if method in {"Autoencoder", "DEC"} else "Classical Clustering"
        method_complexity = complexity.get(method, "Unknown")
    else:
        deep_methods = {"Autoencoder", "DEC"}
        processing_mode = (
            "Hybrid Comparison"
            if any(selected_method in deep_methods for selected_method in compare_methods)
            else "Classical Comparison"
        )
        method_complexity = ", ".join(
            f"{selected_method}: {complexity.get(selected_method, 'Unknown')}"
            for selected_method in compare_methods
        )

    with tab7:
        st.subheader("Performance Dashboard")

        col1, col2, col3 = st.columns(3)
        col1.metric("Runtime (sec)", f"{runtime:.2f}")
        col2.metric("Memory Usage (MB)", memory_display)
        col3.metric("Image Size", f"{h} x {w}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Processing Mode", processing_mode)
        col5.metric("Method Complexity", method_complexity)
        col6.metric("Dataset Type", image_type)

        st.metric("Compute Backend", get_compute_backend())
        st.info(f"Estimated Computational Complexity: {method_complexity}")


# =========================================
# FOOTER
# =========================================

st.markdown("---")
st.markdown("HyperClusterAI (c) Universal Spatial-Spectral Intelligent Segmentation System")
