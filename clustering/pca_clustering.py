from sklearn.decomposition import PCA

from clustering.kmeans_clustering import run_kmeans


def run_pca_kmeans(
    data,
    n_clusters,
    pca_components
):

    pca = PCA(
        n_components=pca_components
    )

    reduced = pca.fit_transform(data)

    labels = run_kmeans(
        reduced,
        n_clusters
    )

    return labels, reduced
