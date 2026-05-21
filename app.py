# =========================================
# app.py
# HyperClusterAI
# FINAL MODULAR VERSION
# =========================================

# =========================================
# IMPORTS
# =========================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from scipy.io import loadmat

from sklearn.preprocessing import StandardScaler

# =========================================
# IMPORT CUSTOM MODULES
# =========================================

from utils.preprocessing import detect_image_type

from utils.metrics import calculate_metrics

from clustering.kmeans import run_kmeans

from clustering.pca_kmeans import run_pca_kmeans

from clustering.spatial import run_spatial

from clustering.autoencoder import run_autoencoder

from clustering.dec import run_dec


# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="HyperClusterAI",
    layout="wide"
)

st.title("🌾 HyperClusterAI")

st.subheader(
    "Universal Spatial-Spectral Intelligent Segmentation System"
)

# =========================================
# SIDEBAR
# =========================================

st.sidebar.header("Settings")

# =========================================
# MODE
# =========================================

mode = st.sidebar.radio(
    "Mode",
    [
        "Single Method",
        "Compare Methods"
    ]
)

# =========================================
# SINGLE METHOD
# =========================================

if mode == "Single Method":

    method = st.sidebar.selectbox(
        "Select Clustering Method",
        [
            "KMeans",
            "PCA + KMeans",
            "Spatial-Spectral",
            "Autoencoder",
            "DEC"
        ]
    )

# =========================================
# COMPARE METHODS
# =========================================

else:

    compare_methods = st.sidebar.multiselect(
        "Select Methods",
        [
            "KMeans",
            "PCA + KMeans",
            "Spatial-Spectral",
            "Autoencoder",
            "DEC"
        ],
        default=["KMeans", "DEC"]
    )

# =========================================
# PARAMETERS
# =========================================

n_clusters = st.sidebar.slider(
    "Number of Clusters",
    2,
    20,
    5
)

patch_size = st.sidebar.slider(
    "Patch Size",
    3,
    9,
    3,
    step=2
)

latent_dim = st.sidebar.slider(
    "Latent Dimension",
    2,
    50,
    10
)

pca_components = st.sidebar.slider(
    "PCA Components",
    2,
    50,
    10
)

run_button = st.sidebar.button(
    "Run Clustering"
)

# =========================================
# FILE UPLOAD
# =========================================

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "png", "jpeg", "npy", "mat"]
)

gt_file = st.file_uploader(
    "Upload Ground Truth (Optional)",
    type=["npy", "mat"]
)

# =========================================
# MAIN PIPELINE
# =========================================

if uploaded_file is not None and run_button:

    file_name = uploaded_file.name

    # =====================================
    # LOAD RGB IMAGE
    # =====================================

    if file_name.endswith(("jpg", "png", "jpeg")):

        image = Image.open(uploaded_file)

        image = np.array(image)

        # grayscale safety
        if len(image.shape) == 2:

            image = np.stack(
                [image]*3,
                axis=-1
            )

        image_type = "RGB"

    # =====================================
    # LOAD NPY
    # =====================================

    elif file_name.endswith("npy"):

        image = np.load(uploaded_file)

        image_type = detect_image_type(image)

    # =====================================
    # LOAD MAT
    # =====================================

    elif file_name.endswith("mat"):

        data = loadmat(uploaded_file)

        possible_keys = [
            k for k in data.keys()
            if not k.startswith("__")
        ]

        image = data[possible_keys[0]]

        image_type = detect_image_type(image)

    else:

        st.error("Unsupported file type.")
        st.stop()

    # =====================================
    # LOAD GROUND TRUTH
    # =====================================

    gt = None

    if gt_file is not None:

        if gt_file.name.endswith("npy"):

            gt = np.load(gt_file)

        elif gt_file.name.endswith("mat"):

            gt_data = loadmat(gt_file)

            possible_keys = [
                k for k in gt_data.keys()
                if not k.startswith("__")
            ]

            gt = gt_data[possible_keys[0]]

    # =====================================
    # IMAGE INFO
    # =====================================

    st.success(
        f"Detected Image Type: {image_type}"
    )

    st.write(
        f"Image Shape: {image.shape}"
    )

    h, w, c = image.shape

    # =====================================
    # LARGE IMAGE WARNING
    # =====================================

    if h * w > 300000:

        st.warning(
            "Large image detected. Processing may be slow."
        )

    # =====================================
    # DISPLAY IMAGE
    # =====================================

    if image_type == "RGB":

        display_image = image

    else:

        bands = image.shape[2]

        b1 = min(10, bands-1)
        b2 = min(20, bands-1)
        b3 = min(30, bands-1)

        display_image = image[:, :, [b1, b2, b3]]

        display_image = (
            display_image - display_image.min()
        ) / (
            display_image.max() - display_image.min()
        )

    # =====================================
    # SHOW ORIGINAL IMAGE
    # =====================================

    st.subheader("Original Image")

    fig1, ax1 = plt.subplots(figsize=(6,6))

    ax1.imshow(display_image)

    ax1.set_title("Input Image")

    ax1.axis("off")

    st.pyplot(fig1)

    # =====================================
    # PREPROCESSING
    # =====================================

    pixels = image.reshape(-1, c)

    scaler = StandardScaler()

    pixels_scaled = scaler.fit_transform(
        pixels
    )

    # =====================================
    # SINGLE METHOD MODE
    # =====================================

    if mode == "Single Method":

        # =================================
        # KMEANS
        # =================================

        if method == "KMeans":

            labels = run_kmeans(
                pixels_scaled,
                n_clusters
            )

        # =================================
        # PCA + KMEANS
        # =================================

        elif method == "PCA + KMeans":

            labels, reduced = run_pca_kmeans(
                pixels_scaled,
                n_clusters,
                pca_components
            )

        # =================================
        # SPATIAL
        # =================================

        elif method == "Spatial-Spectral":

            labels, reduced = run_spatial(
                image,
                patch_size,
                pca_components,
                n_clusters
            )

        # =================================
        # AUTOENCODER
        # =================================

        elif method == "Autoencoder":

            labels, reduced = run_autoencoder(
                image,
                patch_size,
                pca_components,
                latent_dim,
                n_clusters
            )

        # =================================
        # DEC
        # =================================

        elif method == "DEC":

            labels, reduced = run_dec(
                image,
                patch_size,
                pca_components,
                latent_dim,
                n_clusters
            )

        # =================================
        # PCA VISUALIZATION
        # =================================

        if method != "KMeans":

            st.subheader("PCA Visualization")

            pca_vis = reduced[:, :3]

            pca_vis = (
                pca_vis - pca_vis.min()
            ) / (
                pca_vis.max() - pca_vis.min()
            )

            pca_vis = pca_vis.reshape(
                h,
                w,
                3
            )

            fig2, ax2 = plt.subplots(
                figsize=(6,6)
            )

            ax2.imshow(pca_vis)

            ax2.set_title(
                "PCA Representation"
            )

            ax2.axis("off")

            st.pyplot(fig2)

        # =================================
        # CLUSTER MAP
        # =================================

        cluster_map = labels.reshape(h, w)

        st.subheader("Clustered Output")

        fig3, ax3 = plt.subplots(
            figsize=(6,6)
        )

        ax3.imshow(
            cluster_map,
            cmap="nipy_spectral"
        )

        ax3.set_title(
            f"{method} Cluster Map"
        )

        ax3.axis("off")

        st.pyplot(fig3)

        # =================================
        # METRICS
        # =================================

        if gt is not None:

            st.subheader(
                "Evaluation Metrics"
            )

            acc, kappa, nmi, cm = calculate_metrics(
                gt,
                labels
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Accuracy",
                f"{acc:.4f}"
            )

            col2.metric(
                "Kappa",
                f"{kappa:.4f}"
            )

            col3.metric(
                "NMI",
                f"{nmi:.4f}"
            )

            # =============================
            # CONFUSION MATRIX
            # =============================

            st.subheader(
                "Confusion Matrix"
            )

            fig_cm, ax_cm = plt.subplots(
                figsize=(8,6)
            )

            ax_cm.imshow(cm)

            ax_cm.set_title(
                "Confusion Matrix"
            )

            ax_cm.set_xlabel(
                "Predicted"
            )

            ax_cm.set_ylabel(
                "True"
            )

            st.pyplot(fig_cm)

            # =============================
            # CLASS ACCURACY
            # =============================

            st.subheader(
                "Class-wise Accuracy"
            )

            class_acc = (
                cm.diagonal() /
                cm.sum(axis=1)
            )

            fig_acc, ax_acc = plt.subplots(
                figsize=(8,4)
            )

            ax_acc.bar(
                range(len(class_acc)),
                class_acc
            )

            ax_acc.set_title(
                "Class-wise Accuracy"
            )

            ax_acc.set_xlabel(
                "Class"
            )

            ax_acc.set_ylabel(
                "Accuracy"
            )

            st.pyplot(fig_acc)

    # =====================================
    # COMPARE METHODS MODE
    # =====================================

    else:

        st.subheader(
            "Method Comparison Dashboard"
        )

        results = {}

        # =================================
        # LOOP THROUGH METHODS
        # =================================

        for current_method in compare_methods:

            st.write(
                f"Running: {current_method}"
            )

            # =============================
            # KMEANS
            # =============================

            if current_method == "KMeans":

                labels = run_kmeans(
                    pixels_scaled,
                    n_clusters
                )

            # =============================
            # PCA + KMEANS
            # =============================

            elif current_method == "PCA + KMeans":

                labels, reduced = run_pca_kmeans(
                    pixels_scaled,
                    n_clusters,
                    pca_components
                )

            # =============================
            # SPATIAL
            # =============================

            elif current_method == "Spatial-Spectral":

                labels, reduced = run_spatial(
                    image,
                    patch_size,
                    pca_components,
                    n_clusters
                )

            # =============================
            # AE
            # =============================

            elif current_method == "Autoencoder":

                labels, reduced = run_autoencoder(
                    image,
                    patch_size,
                    pca_components,
                    latent_dim,
                    n_clusters
                )

            # =============================
            # DEC
            # =============================

            elif current_method == "DEC":

                labels, reduced = run_dec(
                    image,
                    patch_size,
                    pca_components,
                    latent_dim,
                    n_clusters
                )

            # =============================
            # CLUSTER MAP
            # =============================

            cluster_map = labels.reshape(h, w)

            fig_compare, ax_compare = plt.subplots(
                figsize=(5,5)
            )

            ax_compare.imshow(
                cluster_map,
                cmap="nipy_spectral"
            )

            ax_compare.set_title(
                current_method
            )

            ax_compare.axis("off")

            st.pyplot(fig_compare)

            # =============================
            # METRICS
            # =============================

            if gt is not None:

                acc, kappa, nmi, cm = calculate_metrics(
                    gt,
                    labels
                )

                results[current_method] = {
                    "Accuracy": acc,
                    "Kappa": kappa,
                    "NMI": nmi
                }

        # =================================
        # RESULTS TABLE
        # =================================

        if gt is not None and len(results) > 0:

            st.subheader(
                "Comparison Table"
            )

            st.dataframe(results)

            # =============================
            # COMPARISON GRAPH
            # =============================

            methods = list(results.keys())

            accuracy_vals = [
                results[m]["Accuracy"]
                for m in methods
            ]

            kappa_vals = [
                results[m]["Kappa"]
                for m in methods
            ]

            nmi_vals = [
                results[m]["NMI"]
                for m in methods
            ]

            x = np.arange(len(methods))

            width = 0.25

            fig_bar, ax_bar = plt.subplots(
                figsize=(10,5)
            )

            ax_bar.bar(
                x - width,
                accuracy_vals,
                width,
                label="Accuracy"
            )

            ax_bar.bar(
                x,
                kappa_vals,
                width,
                label="Kappa"
            )

            ax_bar.bar(
                x + width,
                nmi_vals,
                width,
                label="NMI"
            )

            ax_bar.set_xticks(x)

            ax_bar.set_xticklabels(
                methods
            )

            ax_bar.set_title(
                "Method Comparison"
            )

            ax_bar.legend()

            st.pyplot(fig_bar)

# =========================================
# FOOTER
# =========================================

st.markdown("---")

st.markdown(
    "HyperClusterAI © Universal Spatial-Spectral Intelligent Segmentation System"
)