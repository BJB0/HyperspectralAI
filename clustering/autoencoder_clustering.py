from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from clustering.kmeans_clustering import run_kmeans

from utils.patches import extract_patches

from models.autoencoder_model import (
    build_autoencoder
)


def run_autoencoder(
    image,
    patch_size,
    pca_components,
    latent_dim,
    n_clusters
):

    patches = extract_patches(
        image,
        patch_size
    )

    scaler = StandardScaler()

    patches_scaled = scaler.fit_transform(
        patches
    )

    pca = PCA(
        n_components=pca_components
    )

    reduced = pca.fit_transform(
        patches_scaled
    )

    autoencoder, encoder = build_autoencoder(
        reduced.shape[1],
        latent_dim
    )

    # Subsample if the data is very large to speed up neural network training significantly
    max_train_samples = 50000
    if len(reduced) > max_train_samples:
        import numpy as np
        indices = np.random.choice(len(reduced), max_train_samples, replace=False)
        reduced_train = reduced[indices]
    else:
        reduced_train = reduced

    autoencoder.fit(
        reduced_train,
        reduced_train,
        epochs=20,
        batch_size=512,  # Larger batch size for faster fitting
        verbose=0
    )

    features = encoder.predict(
        reduced,
        batch_size=512  # Predict all pixels/patches in larger batches
    )

    labels = run_kmeans(
        features,
        n_clusters
    )

    return labels, features
