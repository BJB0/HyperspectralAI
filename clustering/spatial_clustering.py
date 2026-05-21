from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from clustering.kmeans import run_kmeans

from utils.patches import extract_patches


def run_spatial(
    image,
    patch_size,
    pca_components,
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

    labels = run_kmeans(
        reduced,
        n_clusters
    )

    return labels, reduced