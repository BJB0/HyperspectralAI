from sklearn.cluster import KMeans


def run_kmeans(data, n_clusters):

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(data)

    return labels