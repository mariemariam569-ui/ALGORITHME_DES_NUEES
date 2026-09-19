import numpy as np


def kmeans_vectorise(X, k, max_iter=100, tol=1e-4,
                     seed=None, init='kmeans++'):
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    rng = np.random.default_rng(seed)

    if init == 'kmeans++':
        centres = _kmeans_pp_vectorise(X, k, rng)
    else:
        idx = rng.choice(n, k, replace=False)
        centres = X[idx].copy()

    labels = np.zeros(n, dtype=int)
    n_iter = 0

    for iteration in range(max_iter):
        X_norm2 = np.sum(X ** 2, axis=1, keepdims=True)
        C_norm2 = np.sum(centres ** 2, axis=1)
        cross = X @ centres.T
        distances = X_norm2 - 2 * cross + C_norm2

        labels = np.argmin(distances, axis=1)

        nouveaux = np.zeros_like(centres)
        comptes = np.zeros(k)
        np.add.at(nouveaux, labels, X)
        np.add.at(comptes, labels, 1)
        comptes = np.where(comptes == 0, 1, comptes)
        nouveaux /= comptes[:, None]

        deplacement = np.linalg.norm(nouveaux - centres)
        centres = nouveaux
        n_iter = iteration + 1

        if deplacement < tol:
            break

    distances = np.linalg.norm(X[:, None, :] - centres[None, :, :], axis=2)
    labels = np.argmin(distances, axis=1)
    inertie = float(np.sum(np.min(distances, axis=1) ** 2))

    return centres, labels, inertie, n_iter


def _kmeans_pp_vectorise(X, k, rng):
    n = len(X)
    centres = [X[rng.integers(n)]]

    for _ in range(1, k):
        dists = np.array([
            np.sum((X - c) ** 2, axis=1) for c in centres
        ])
        d2 = np.min(dists, axis=0)
        proba = d2 / d2.sum()
        centres.append(X[rng.choice(n, p=proba)])

    return np.array(centres)