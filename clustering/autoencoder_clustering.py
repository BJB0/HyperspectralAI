from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from clustering.kmeans import run_kmeans

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

    autoencoder.fit(
        reduced,
        reduced,
        epochs=20,
        batch_size=256,
        verbose=0
    )

    features = encoder.predict(
        reduced,
        batch_size=256
    )

    labels = run_kmeans(
        features,
        n_clusters
    )

    return labels, reduced