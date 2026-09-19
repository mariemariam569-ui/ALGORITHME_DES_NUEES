import numpy as np


def inertie(X, labels, centres):
    X = np.asarray(X)
    total = 0.0
    for j in range(len(centres)):
        pts = X[labels == j]
        if len(pts) > 0:
            total += np.sum((pts - centres[j]) ** 2)
    return total


def silhouette_score_manuel(X, labels):
    X = np.asarray(X, dtype=float)
    n = len(X)
    labels = np.asarray(labels)
    classes = np.unique(labels)

    if len(classes) < 2:
        return 0.0

    diff = X[:, None, :] - X[None, :, :]
    distances = np.linalg.norm(diff, axis=2)

    scores = np.zeros(n)
    for i in range(n):
        masque_meme = (labels == labels[i])
        masque_meme[i] = False
        if np.sum(masque_meme) > 0:
            a = np.mean(distances[i, masque_meme])
        else:
            a = 0.0

        b = np.inf
        for c in classes:
            if c == labels[i]:
                continue
            masque_autre = (labels == c)
            if np.sum(masque_autre) > 0:
                d = np.mean(distances[i, masque_autre])
                if d < b:
                    b = d

        if max(a, b) > 0:
            scores[i] = (b - a) / max(a, b)

    return float(np.mean(scores))


def methode_coude(X, k_range, seed=42, init='kmeans++'):
    from kmeans_vectorise import kmeans_vectorise

    ks, inerties = [], []
    for k in k_range:
        _, _, inertia, _ = kmeans_vectorise(X, k, seed=seed, init=init)
        ks.append(k)
        inerties.append(inertia)
        print(f"K = {k:2d} | Inertie = {inertia:10.4f}")

    return ks, inerties