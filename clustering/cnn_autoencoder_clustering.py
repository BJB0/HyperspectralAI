from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from clustering.kmeans_clustering import run_kmeans
from models.autoencoder_model import build_cnn_autoencoder
from utils.patches import extract_patch_cubes


def run_cnn_autoencoder(
    image,
    patch_size,
    pca_components,
    latent_dim,
    n_clusters
):

    h, w, c = image.shape

    pixels = image.reshape(-1, c)
    pixels_scaled = StandardScaler().fit_transform(pixels)

    pca = PCA(
        n_components=pca_components
    )

    reduced_pixels = pca.fit_transform(
        pixels_scaled
    )

    reduced_cube = reduced_pixels.reshape(
        h,
        w,
        pca_components
    )

    patch_cubes = extract_patch_cubes(
        reduced_cube,
        patch_size
    )

    autoencoder, encoder = build_cnn_autoencoder(
        patch_cubes.shape[1:],
        latent_dim
    )

    autoencoder.fit(
        patch_cubes,
        patch_cubes,
        epochs=20,
        batch_size=256,
        verbose=0
    )

    features = encoder.predict(
        patch_cubes,
        batch_size=256
    )

    labels = run_kmeans(
        features,
        n_clusters
    )

    return labels, features
