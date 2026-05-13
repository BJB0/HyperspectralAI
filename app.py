
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from scipy.io import loadmat

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="HyperClusterAI",
    layout="wide"
)

st.title("🌾 HyperClusterAI")
st.subheader("Universal Spatial-Spectral Intelligent Segmentation System")

# =========================================
# SIDEBAR
# =========================================

st.sidebar.header("Settings")

method = st.sidebar.selectbox(
    "Select Clustering Method",
    [
        "KMeans",
        "PCA + KMeans",
        "Spatial-Spectral"
    ]
)

n_clusters = st.sidebar.slider(
    "Number of Clusters",
    2,
    20,
    5
)

pca_components = st.sidebar.slider(
    "PCA Components",
    2,
    50,
    10
)

patch_size = st.sidebar.slider(
  "Patch Size",
  3,
  9,
  3,
  step=2
)

# =========================================
# FILE UPLOAD
# =========================================

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "png", "jpeg", "npy", "mat"]
)

# =========================================
# IMAGE TYPE DETECTION
# =========================================

def detect_image_type(data):

    if len(data.shape) == 3:

        if data.shape[2] <= 4:
            return "RGB"

        else:
            return "HSI"

    return "UNKNOWN"
  
  
  
# =========================================
# PATCH EXTRACTION
# =========================================

def extract_patches(image, patch_size=3):

    pad = patch_size // 2

    h, w, c = image.shape

    padded = np.pad(
        image,
        ((pad, pad), (pad, pad), (0, 0)),
        mode='reflect'
    )

    patches = []

    for i in range(h):

        for j in range(w):

            patch = padded[
                i:i+patch_size,
                j:j+patch_size
            ]

            patches.append(patch.flatten())

    return np.array(patches)
  

# =========================================
# LOAD IMAGE
# =========================================

if uploaded_file is not None:

    file_name = uploaded_file.name

    # -------------------------------------
    # RGB IMAGE
    # -------------------------------------

    if file_name.endswith(("jpg", "png", "jpeg")):

        image = Image.open(uploaded_file)
        image = np.array(image)

        image_type = "RGB"

    # -------------------------------------
    # NPY FILE
    # -------------------------------------

    elif file_name.endswith("npy"):

        image = np.load(uploaded_file)

        image_type = detect_image_type(image)

    # -------------------------------------
    # MAT FILE
    # -------------------------------------

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
    # DISPLAY INFO
    # =====================================

    st.success(f"Detected Image Type: {image_type}")

    st.write(f"Image Shape: {image.shape}")

    # =====================================
    # RGB VISUALIZATION
    # =====================================

    if image_type == "RGB":

        display_image = image

    # =====================================
    # HSI VISUALIZATION
    # =====================================

    elif image_type == "HSI":

        bands = image.shape[2]

        # false color image
        b1 = min(10, bands-1)
        b2 = min(20, bands-1)
        b3 = min(30, bands-1)

        display_image = image[:, :, [b1, b2, b3]]

        # normalize for display
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

    h, w, c = image.shape

    pixels = image.reshape(-1, c)

    scaler = StandardScaler()

    pixels_scaled = scaler.fit_transform(pixels)

    # =====================================
    # METHOD 1: KMEANS
    # =====================================

    if method == "KMeans":

        st.subheader("Running KMeans Clustering...")

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )

        labels = kmeans.fit_predict(pixels_scaled)

    # =====================================
    # METHOD 2: PCA + KMEANS
    # =====================================

    elif method == "PCA + KMeans":

        st.subheader("Running PCA + KMeans...")

        pca = PCA(n_components=pca_components)

        reduced = pca.fit_transform(pixels_scaled)

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )

        labels = kmeans.fit_predict(reduced)
        
        
      # =====================================
      # METHOD 3: SPATIAL-SPECTRAL
      # =====================================

    elif method == "Spatial-Spectral":
      st.subheader("Running Spatial-Spectral Clustering...")

      # extract patches
      patches = extract_patches(
        image,
        patch_size=patch_size
      )

      # normalize
      scaler = StandardScaler()

      patches_scaled = scaler.fit_transform(patches)

      # PCA
      pca = PCA(n_components=pca_components)

      reduced = pca.fit_transform(patches_scaled)

      # clustering
      kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
      )

      labels = kmeans.fit_predict(reduced)
        
        
        
        
        

      # PCA visualization
      st.subheader("PCA Visualization")

      pca_vis = reduced[:, :3]

      pca_vis = (
        pca_vis - pca_vis.min()
        ) / (
            pca_vis.max() - pca_vis.min()
        )

      pca_vis = pca_vis.reshape(h, w, 3)

      fig2, ax2 = plt.subplots(figsize=(6,6))

      ax2.imshow(pca_vis)
      ax2.set_title("PCA Image")
      ax2.axis("off")

      st.pyplot(fig2)

    # =====================================
    # CLUSTER MAP
    # =====================================

    cluster_map = labels.reshape(h, w)

    # =====================================
    # DISPLAY RESULTS
    # =====================================

    st.subheader("Clustered Output")

    fig3, ax3 = plt.subplots(figsize=(6,6))

    ax3.imshow(cluster_map, cmap="nipy_spectral")
    ax3.set_title("Cluster Map")
    ax3.axis("off")

    st.pyplot(fig3)

    # =====================================
    # CLUSTER DISTRIBUTION
    # =====================================

    st.subheader("Cluster Distribution")

    unique, counts = np.unique(labels, return_counts=True)

    for u, c in zip(unique, counts):

        st.write(f"Cluster {u}: {c} pixels")

# =========================================
# FOOTER
# =========================================

st.markdown("---")
st.markdown("HyperClusterAI © Intelligent Spatial-Spectral Segmentation System")