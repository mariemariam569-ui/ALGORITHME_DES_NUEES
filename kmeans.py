import numpy as np


class NueesDynamiques:
    def __init__(self, k=3, max_iter=100, tol=1e-4,
                 random_state=None, init='random', verbose=False):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init = init
        self.verbose = verbose
        self.centres = None
        self.labels = None
        self.inertie_ = None
        self.n_iter_ = 0
        self.historique_inertie = []

    def _initialiser_centres(self, X):
        rng = np.random.default_rng(self.random_state)
        if self.init == 'kmeans++':
            return self._kmeans_plus_plus(X, rng)
        indices = rng.choice(len(X), self.k, replace=False)
        return X[indices].copy()

    def _kmeans_plus_plus(self, X, rng):
        n = len(X)
        centres = [X[rng.integers(n)]]
        for _ in range(1, self.k):
            d2 = np.array([
                np.min([np.linalg.norm(x - c) ** 2 for c in centres])
                for x in X
            ])
            proba = d2 / d2.sum()
            centres.append(X[rng.choice(n, p=proba)])
        return np.array(centres)

    def _distance(self, X, centres):
        return np.linalg.norm(X[:, np.newaxis] - centres, axis=2)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]

        if self.k > n_samples:
            raise ValueError("k ne peut pas dépasser le nombre de points.")

        self.centres = self._initialiser_centres(X)
        self.historique_inertie = []

        for iteration in range(self.max_iter):
            distances = self._distance(X, self.centres)
            labels = np.argmin(distances, axis=1)

            nouveaux_centres = np.zeros_like(self.centres)
            for j in range(self.k):
                points_cluster = X[labels == j]
                if len(points_cluster) > 0:
                    nouveaux_centres[j] = points_cluster.mean(axis=0)
                else:
                    nouveaux_centres[j] = X[np.random.randint(n_samples)]

            inertie = sum(
                np.sum((X[labels == j] - nouveaux_centres[j]) ** 2)
                for j in range(self.k)
            )
            self.historique_inertie.append(inertie)

            deplacement = np.linalg.norm(nouveaux_centres - self.centres)
            self.centres = nouveaux_centres
            self.n_iter_ = iteration + 1

            if self.verbose:
                print(f"Itération {iteration + 1:2d} | "
                      f"Inertie = {inertie:10.4f} | "
                      f"Déplacement = {deplacement:.6f}")

            if deplacement < self.tol:
                break

        distances = self._distance(X, self.centres)
        self.labels = np.argmin(distances, axis=1)
        self.inertie_ = sum(
            np.sum((X[self.labels == j] - self.centres[j]) ** 2)
            for j in range(self.k)
        )
        return self

    def predict(self, X):
        if self.centres is None:
            raise RuntimeError("Le modèle doit être entraîné d'abord.")
        X = np.asarray(X, dtype=float)
        distances = self._distance(X, self.centres)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X):
        return self.fit(X).labels_

    def __repr__(self):
        return (f"NueesDynamiques(k={self.k}, max_iter={self.max_iter}, "
                f"tol={self.tol}, init='{self.init}')")