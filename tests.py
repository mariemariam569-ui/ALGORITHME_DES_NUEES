import numpy as np
import time
from kmeans import NueesDynamiques
from kmeans_vectorise import kmeans_vectorise
from evaluation import silhouette_score_manuel


def generer_donnees_synthetiques(n_par_cluster=100, seed=42):
    rng = np.random.default_rng(seed)
    c1 = rng.normal(loc=[0, 0], scale=1.0, size=(n_par_cluster, 2))
    c2 = rng.normal(loc=[5, 5], scale=1.0, size=(n_par_cluster, 2))
    c3 = rng.normal(loc=[0, 6], scale=1.0, size=(n_par_cluster, 2))
    X = np.vstack([c1, c2, c3])
    y_vrai = np.array([0] * n_par_cluster +
                      [1] * n_par_cluster +
                      [2] * n_par_cluster)
    return X, y_vrai


def test_basique():
    print("=" * 60)
    print("TEST 1 : Données synthétiques (K=3)")
    print("=" * 60)

    X, y_vrai = generer_donnees_synthetiques()
    print(f"Dimensions : {X.shape}")

    t0 = time.time()
    model = NueesDynamiques(k=3, max_iter=100, tol=1e-4,
                            random_state=42, verbose=True)
    model.fit(X)
    t1 = time.time()

    print(f"\nRésultats :")
    print(f"   Itérations      : {model.n_iter_}")
    print(f"   Inertie finale  : {model.inertie_:.4f}")
    print(f"   Temps           : {(t1-t0)*1000:.2f} ms")

    from sklearn.metrics import adjusted_rand_score
    score = adjusted_rand_score(y_vrai, model.labels)
    print(f"   ARI (précision) : {score:.4f}")

    return model, X


def test_comparaison_sklearn():
    print("\n" + "=" * 60)
    print("TEST 2 : Comparaison avec scikit-learn")
    print("=" * 60)

    from sklearn.cluster import KMeans
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = iris.data

    t0 = time.time()
    model1 = NueesDynamiques(k=3, random_state=42, init='kmeans++')
    model1.fit(X)
    t1 = time.time()
    temps1 = (t1 - t0) * 1000

    t0 = time.time()
    centres2, labels2, inertie2, n_iter2 = kmeans_vectorise(
        X, k=3, seed=42, init='kmeans++'
    )
    t1 = time.time()
    temps2 = (t1 - t0) * 1000

    t0 = time.time()
    km_sk = KMeans(n_clusters=3, random_state=42, n_init=10)
    km_sk.fit(X)
    t1 = time.time()
    temps3 = (t1 - t0) * 1000

    print(f"\n{'Méthode':<25} {'Inertie':>12} {'Temps (ms)':>12}")
    print("-" * 55)
    print(f"{'Notre (from scratch)':<25} {model1.inertie_:>12.4f} "
          f"{temps1:>12.2f}")
    print(f"{'Notre (vectorisée)':<25} {inertie2:>12.4f} "
          f"{temps2:>12.2f}")
    print(f"{'scikit-learn':<25} {km_sk.inertia_:>12.4f} "
          f"{temps3:>12.2f}")


def test_choix_k():
    print("\n" + "=" * 60)
    print("TEST 3 : Méthode du coude sur Iris")
    print("=" * 60)

    from sklearn.datasets import load_iris
    from evaluation import methode_coude

    X = load_iris().data
    ks, inerties = methode_coude(X, k_range=range(1, 9), seed=42)


def test_initialisation():
    print("\n" + "=" * 60)
    print("TEST 4 : Impact de l'initialisation")
    print("=" * 60)

    from sklearn.datasets import load_iris
    X = load_iris().data

    for init_method in ['random', 'kmeans++']:
        inerties = []
        for seed in range(10):
            model = NueesDynamiques(k=3, random_state=seed,
                                    init=init_method)
            model.fit(X)
            inerties.append(model.inertie_)

        print(f"\n{init_method.upper():<12} : "
              f"min = {min(inerties):.2f}, "
              f"max = {max(inerties):.2f}, "
              f"moy = {np.mean(inerties):.2f}, "
              f"écart-type = {np.std(inerties):.2f}")


def test_silhouette():
    print("\n" + "=" * 60)
    print("TEST 5 : Score de silhouette sur Iris")
    print("=" * 60)

    from sklearn.datasets import load_iris
    X = load_iris().data

    for k in [2, 3, 4, 5]:
        model = NueesDynamiques(k=k, random_state=42, init='kmeans++')
        model.fit(X)
        sil = silhouette_score_manuel(X, model.labels)
        print(f"K = {k} | Inertie = {model.inertie_:>8.2f} | "
              f"Silhouette = {sil:.4f}")


if __name__ == "__main__":
    model, X = test_basique()
    test_comparaison_sklearn()
    test_choix_k()
    test_initialisation()
    test_silhouette()