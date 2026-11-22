from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def train_model(X):
    model = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=10
    )

    model.fit(X)

    return model


def predict(model, X):
    return model.predict(X)


def get_centroids(model):
    return model.cluster_centers_


def evaluate_model(X, labels):
    score = silhouette_score(X, labels)

    return score