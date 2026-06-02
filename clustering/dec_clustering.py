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

    if isinstance(z, np.ndarray):
        dist = np.sum(
            (z[:, None] - centers) ** 2,
            axis=2
        )
        q = 1.0 / (1.0 + dist)
        return q / np.sum(q, axis=1, keepdims=True)
    else:
        z_expanded = tf.expand_dims(z, 1)
        centers_expanded = tf.expand_dims(tf.convert_to_tensor(centers, dtype=tf.float32), 0)
        dist = tf.reduce_sum(tf.square(z_expanded - centers_expanded), axis=2)
        q = 1.0 / (1.0 + dist)
        return q / tf.reduce_sum(q, axis=1, keepdims=True)


# =========================================
# TARGET DISTRIBUTION
# =========================================

def target_distribution(q):

    if isinstance(q, np.ndarray):
        weight = q ** 2 / np.sum(q, axis=0)
        return (
            weight.T / np.sum(weight, axis=1)
        ).T
    else:
        weight = tf.square(q) / tf.reduce_sum(q, axis=0, keepdims=True)
        return weight / tf.reduce_sum(weight, axis=1, keepdims=True)


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
    # PRETRAIN AUTOENCODER (Subsampled for speed)
    # =====================================

    max_train_samples = 50000
    if len(reduced) > max_train_samples:
        indices = np.random.choice(len(reduced), max_train_samples, replace=False)
        reduced_train = reduced[indices]
    else:
        reduced_train = reduced

    autoencoder.fit(
        reduced_train,
        reduced_train,
        epochs=epochs,
        batch_size=512,  # Larger batch size for faster pretraining
        verbose=0
    )

    # =====================================
    # INITIAL FEATURES
    # =====================================

    features = encoder.predict(
        reduced,
        batch_size=512
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
    # DEC REFINEMENT (Subsampled to prevent OOM)
    # =====================================

    optimizer = Adam(0.0001)

    max_dec_samples = 25000
    if len(reduced) > max_dec_samples:
        dec_indices = np.random.choice(len(reduced), max_dec_samples, replace=False)
        reduced_dec = reduced[dec_indices]
    else:
        reduced_dec = reduced

    for ite in range(dec_iters):

        with tf.GradientTape() as tape:

            z = encoder(
                reduced_dec,
                training=True
            )

            q = soft_assign(
                z,
                cluster_centers
            )

            p = target_distribution(q)
            # Stop gradient on the target distribution to treat it as a constant target as per DEC algorithm
            p = tf.stop_gradient(p)

            loss = tf.keras.losses.KLDivergence()(
                p,
                q
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
        batch_size=512
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

    return final_labels, final_features
