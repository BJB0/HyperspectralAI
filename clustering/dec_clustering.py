# =========================================
# clustering/dec.py
# =========================================

import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from tensorflow.keras.optimizers import Adam

from utils.patches import extract_patches
from models.autoencoder_model import build_autoencoder


# =========================================
# SOFT ASSIGNMENT
# =========================================

def soft_assign(z, centers):

    dist = np.sum(
        (z[:, None] - centers) ** 2,
        axis=2
    )

    q = 1.0 / (1.0 + dist)

    q = q / np.sum(
        q,
        axis=1,
        keepdims=True
    )

    return q


# =========================================
# TARGET DISTRIBUTION
# =========================================

def target_distribution(q):

    weight = q ** 2 / np.sum(q, axis=0)

    return (
        weight.T / np.sum(weight, axis=1)
    ).T


# =========================================
# DEC PIPELINE
# =========================================

def run_dec(
    image,
    patch_size,
    pca_components,
    latent_dim,
    n_clusters,
    epochs=10,
    dec_iters=20
):

    # =====================================
    # PATCH EXTRACTION
    # =====================================

    patches = extract_patches(
        image,
        patch_size
    )

    # =====================================
    # NORMALIZATION
    # =====================================

    scaler = StandardScaler()

    patches_scaled = scaler.fit_transform(
        patches
    )

    # =====================================
    # PCA
    # =====================================

    pca = PCA(
        n_components=pca_components
    )

    reduced = pca.fit_transform(
        patches_scaled
    )

    # =====================================
    # BUILD AUTOENCODER
    # =====================================

    autoencoder, encoder = build_autoencoder(
        reduced.shape[1],
        latent_dim
    )

    # =====================================
    # PRETRAIN AUTOENCODER
    # =====================================

    autoencoder.fit(
        reduced,
        reduced,
        epochs=epochs,
        batch_size=256,
        verbose=0
    )

    # =====================================
    # INITIAL FEATURES
    # =====================================

    features = encoder.predict(
        reduced,
        batch_size=256
    )

    # =====================================
    # INITIAL KMEANS
    # =====================================

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(
        features
    )

    cluster_centers = kmeans.cluster_centers_

    # =====================================
    # DEC REFINEMENT
    # =====================================

    optimizer = Adam(0.0001)

    for ite in range(dec_iters):

        with tf.GradientTape() as tape:

            z = encoder(
                reduced,
                training=True
            )

            z_np = z.numpy()

            q = soft_assign(
                z_np,
                cluster_centers
            )

            p = target_distribution(q)

            q_tf = tf.convert_to_tensor(
                q,
                dtype=tf.float32
            )

            p_tf = tf.convert_to_tensor(
                p,
                dtype=tf.float32
            )

            loss = tf.keras.losses.KLDivergence()(
                p_tf,
                q_tf
            )

        grads = tape.gradient(
            loss,
            encoder.trainable_weights
        )

        optimizer.apply_gradients(
            zip(
                grads,
                encoder.trainable_weights
            )
        )

    # =====================================
    # FINAL FEATURES
    # =====================================

    final_features = encoder.predict(
        reduced,
        batch_size=256
    )

    # =====================================
    # FINAL KMEANS
    # =====================================

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    final_labels = kmeans.fit_predict(
        final_features
    )

    # =====================================
    # RETURN
    # =====================================

    return final_labels, reduced