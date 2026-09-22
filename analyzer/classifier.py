"""Unsupervised-then-supervised rep consistency labeling."""
import os

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURE_FIELDS = (
    "elbow_at_release",
    "knee_bend_at_setup",
    "wrist_follow_through",
    "arc_proxy",
)


def _mean_dist_to_centroid(X_scaled, cluster_ids, cluster_id, centroid):
    members = X_scaled[cluster_ids == cluster_id]
    if len(members) == 0:
        return float("inf")
    return float(np.mean(np.linalg.norm(members - centroid, axis=1)))


def train_and_label(reps, output_dir: str = "output", min_reps: int = 3):
    if len(reps) < min_reps:
        for rep in reps:
            rep["label"] = "unclassified"
        return reps

    X = np.array([[rep[field] for field in FEATURE_FIELDS] for rep in reps], dtype=float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
    cluster_ids = kmeans.fit_predict(X_scaled)

    if len(np.unique(cluster_ids)) < 2:
        # All reps landed in one cluster (e.g. near-identical metrics) — nothing to
        # meaningfully split, and LogisticRegression can't fit on a single class.
        for rep in reps:
            rep["label"] = "consistent"
        return reps

    count_0 = int(np.sum(cluster_ids == 0))
    count_1 = int(np.sum(cluster_ids == 1))

    if count_0 != count_1:
        consistent_cluster = 0 if count_0 > count_1 else 1
    else:
        dist_0 = _mean_dist_to_centroid(X_scaled, cluster_ids, 0, kmeans.cluster_centers_[0])
        dist_1 = _mean_dist_to_centroid(X_scaled, cluster_ids, 1, kmeans.cluster_centers_[1])
        consistent_cluster = 0 if dist_0 <= dist_1 else 1

    y = (cluster_ids == consistent_cluster).astype(int)

    model = LogisticRegression()
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)

    for rep, pred in zip(reps, y_pred):
        rep["label"] = "consistent" if pred == 1 else "inconsistent"

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(
        {"scaler": scaler, "kmeans": kmeans, "model": model},
        os.path.join(output_dir, "classifier.pkl"),
    )

    return reps
