from sklearn.cluster import KMeans, MiniBatchKMeans


def run_kmeans(data, n_clusters):

    # If the number of samples is large (e.g. > 100,000), use MiniBatchKMeans for massive speedup
    if len(data) > 100000:
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            batch_size=2048,
            n_init=3
        )
    else:
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )

    labels = kmeans.fit_predict(data)

    return labels